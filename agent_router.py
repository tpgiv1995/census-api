# agent_router.py — AI agent orchestration with safe tool-call loop & flexible /message body parsing
from fastapi import APIRouter, UploadFile, File, Request, Form, Body
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any, Optional
from openai import OpenAI
import os, io, json, uuid, requests, logging

router = APIRouter(prefix="/agent", tags=["agent"])

# ----------- Logging -----------
log = logging.getLogger("agent")
if not log.handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# ----------- OpenAI client -----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()
client = OpenAI(api_key=OPENAI_API_KEY)

# ----------- Simple in-memory session store -----------
_SESSIONS: Dict[str, Dict[str, Any]] = {}

def _new_session_id() -> str:
    return str(uuid.uuid4())

def _base_url(req: Request) -> str:
    # Use public URL Render gives to this request
    return str(req.base_url).rstrip("/")

# ----------- Agent instructions -----------
AGENT_SYSTEM_PROMPT = """You are the Census Data Greeter Agent.
Goal: transform a user-supplied census source file into the format required by a carrier template using three tools:
- profile_source() → analyze files, return recipe_draft + questions
- transform_census(recipe_json, auto_approve) → masked 20-row preview + stats
- export_census(recipe_json) → final Excel

Rules (binding):
• Never display raw PII. Mask SSN as ***-**-1234; mask phones/emails.
• Limit previews to ≤20 rows.
• Don’t fabricate; ask a concise clarifier if unsure.
• Export only after explicit user approval.

Workflow:
1) Wait until TWO files are present (source + template).
2) Call profile_source. If questions are returned, ask them one-by-one; merge answers.
3) Build a strict Recipe JSON (valid keys only). Normalize field names & coverage tiers (EE/ES/EC/EF/Waived).
4) Call transform_census(auto_approve:true), show a masked 20-row preview. Ask for approval.
5) If approved, call export_census and return completion.

Be concise. Summarize decisions. If a required field is missing, state the blocker and what is needed.
"""

# ----------- Tool specs advertised to the model -----------
TOOLS_SPEC = [
    {
        "type": "function",
        "function": {
            "name": "profile_source",
            "description": "Profiles the uploaded source and template to produce a recipe_draft and potential questions.",
            "parameters": { "type": "object", "properties": {}, "additionalProperties": False },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "transform_census",
            "description": "Transforms with a final recipe and returns a masked 20-row preview + stats.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_json": { "type": "string" },
                    "auto_approve": { "type": "boolean", "default": True }
                },
                "required": ["recipe_json"],
                "additionalProperties": False
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "export_census",
            "description": "Exports the full transformed dataset using the final recipe.",
            "parameters": {
                "type": "object",
                "properties": { "recipe_json": { "type": "string" } },
                "required": ["recipe_json"],
                "additionalProperties": False
            },
        },
    },
]

# ----------- Helpers: sanitize + tool execution -----------
def _sanitize_messages(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Enforce Chat Completions ordering:
    - Keep roles: system/user/assistant/tool
    - Drop stray 'tool' messages that aren't responding to immediately prior assistant tool_calls
    - Ensure first message is the system prompt
    """
    msgs: List[Dict[str, Any]] = []
    for m in raw:
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue

        if role == "tool":
            if not msgs:
                continue
            prev = msgs[-1]
            tcid = m.get("tool_call_id")
            if prev.get("role") == "assistant" and prev.get("tool_calls") and tcid:
                ids = [tc.get("id") for tc in prev["tool_calls"] if isinstance(tc, dict)]
                if tcid in ids:
                    content = m.get("content")
                    if not isinstance(content, str):
                        content = json.dumps(content or {})
                    msgs.append({"role": "tool", "tool_call_id": tcid, "content": content})
            continue

        keep = {"role": role, "content": m.get("content")}
        if role == "assistant" and isinstance(m.get("tool_calls"), list):
            keep["tool_calls"] = m["tool_calls"]
        msgs.append(keep)

    if not msgs or msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}] + msgs
    return msgs

def _file_tuple(name: str, data: bytes):
    return (name or "upload", io.BytesIO(data))

def _call_profile(req: Request, sess: Dict[str, Any]) -> Dict[str, Any]:
    base = _base_url(req)
    files = {
        "source_file": _file_tuple(sess["source"]["name"], sess["source"]["bytes"]),
        "template_file": _file_tuple(sess["template"]["name"], sess["template"]["bytes"]),
    }
    r = requests.post(f"{base}/profile", files=files, timeout=180)
    r.raise_for_status()
    return r.json()

def _call_transform(req: Request, sess: Dict[str, Any], recipe_json: str, auto_approve: bool = True) -> Dict[str, Any]:
    base = _base_url(req)
    files = {
        "source_file": _file_tuple(sess["source"]["name"], sess["source"]["bytes"]),
        "template_file": _file_tuple(sess["template"]["name"], sess["template"]["bytes"]),
    }
    data = { "recipe_json": recipe_json, "auto_approve": str(bool(auto_approve)).lower() }
    r = requests.post(f"{base}/transform?auto_approve={data['auto_approve']}", files=files, data=data, timeout=240)
    r.raise_for_status()
    return r.json()

def _call_export(req: Request, sess: Dict[str, Any], recipe_json: str) -> Dict[str, Any]:
    base = _base_url(req)
    files = {
        "source_file": _file_tuple(sess["source"]["name"], sess["source"]["bytes"]),
        "template_file": _file_tuple(sess["template"]["name"], sess["template"]["bytes"]),
    }
    data = { "recipe_json": recipe_json, "carrier_name": "" }
    r = requests.post(f"{base}/export", files=files, data=data, timeout=300)
    r.raise_for_status()
    sess["last_export"] = r.content
    return {"status": "export_ready", "download_url": f"/agent/download/{sess['id']}"}

def _dispatch_tool(req: Request, sess: Dict[str, Any], tool_call: Dict[str, Any]) -> Dict[str, Any]:
    fn = tool_call.get("function", {})
    name = fn.get("name")
    try:
        args = json.loads(fn.get("arguments") or "{}")
    except Exception:
        args = {}

    if name == "profile_source":
        return _call_profile(req, sess)
    if name == "transform_census":
        recipe_json = args.get("recipe_json", "")
        auto_approve = bool(args.get("auto_approve", True))
        return _call_transform(req, sess, recipe_json, auto_approve)
    if name == "export_census":
        recipe_json = args.get("recipe_json", "")
        return _call_export(req, sess, recipe_json)

    return {"error": f"Unknown tool '{name}'"}

def _run_agent(req: Request, sess: Dict[str, Any]) -> Dict[str, Any]:
    MAX_ROUNDS = 6
    messages = _sanitize_messages(sess["messages"])

    download_url: Optional[str] = None
    last_text: str = ""

    for _ in range(MAX_ROUNDS):
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=TOOLS_SPEC,
            tool_choice="auto",
            temperature=0.1,
        )
        choice = resp.choices[0]
        asst = choice.message

        asst_dict: Dict[str, Any] = {"role": asst.role, "content": asst.content}
        if getattr(asst, "tool_calls", None):
            tool_calls = []
            for tc in asst.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                })
            asst_dict["tool_calls"] = tool_calls

        messages.append(asst_dict)

        if "tool_calls" in asst_dict and asst_dict["tool_calls"]:
            for tc in asst_dict["tool_calls"]:
                result = _dispatch_tool(req, sess, tc)
                if isinstance(result, dict) and result.get("download_url"):
                    download_url = result["download_url"]

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result)
                })
            continue

        last_text = asst_dict.get("content") or ""
        break

    sess["messages"] = messages

    payload: Dict[str, Any] = {"text": last_text, "session_id": sess["id"]}
    if download_url:
        payload["download_url"] = download_url
    return payload

# ----------- Public endpoints -----------
@router.post("/start")
async def start(
    request: Request,
    source_file: UploadFile = File(...),
    template_file: UploadFile = File(...)
):
    src_bytes = await source_file.read()
    tpl_bytes = await template_file.read()

    sid = _new_session_id()
    sess = {
        "id": sid,
        "source": {"name": source_file.filename, "bytes": src_bytes},
        "template": {"name": template_file.filename, "bytes": tpl_bytes},
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": "Two files uploaded (source + template). Please profile them and propose the recipe. Then show a masked 20-row preview before export."}
        ],
    }
    _SESSIONS[sid] = sess
    log.info(f"[start] session {sid} created; files: {source_file.filename}, {template_file.filename}")
    return {"session_id": sid, "status": "ready"}

@router.post("/message")
async def message(
    request: Request,
    # Accept form *or* JSON
    session_id_form: Optional[str] = Form(None),
    text_form: Optional[str] = Form(None),
    payload: Optional[Dict[str, Any]] = Body(None),
):
    ctype = request.headers.get("content-type", "")
    log.info(f"[message] content-type: {ctype}")

    # Prefer JSON payload if present
    session_id = session_id_form
    text = text_form

    if payload:
        if session_id is None:
            session_id = payload.get("session_id")
        if text is None:
            text = payload.get("text")

    if not session_id or session_id not in _SESSIONS:
        return JSONResponse({"error": "invalid or missing session_id"}, status_code=400)

    sess = _SESSIONS[session_id]
    if text:
        sess["messages"].append({"role": "user", "content": str(text)})

    log.info(f"[message] session {session_id}; text={bool(text)}")
    result = _run_agent(request, sess)
    return result

@router.get("/download/{sid}")
async def download(sid: str):
    sess = _SESSIONS.get(sid)
    if not sess or "last_export" not in sess:
        return JSONResponse({"error": "no export available"}, status_code=404)
    return StreamingResponse(
        io.BytesIO(sess["last_export"]),
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": 'attachment; filename="Transformed_Census.xlsx"'},
    )
