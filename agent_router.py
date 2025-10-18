# agent_router.py — AI agent orchestration with safe tool-call loop & message sanitizer
from fastapi import APIRouter, UploadFile, File, Request
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any, Optional
from openai import OpenAI
import os, io, json, uuid, requests

router = APIRouter(prefix="/agent", tags=["agent"])

# ----------- OpenAI client -----------
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

# Model can be configured from Render env var; sensible default below
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini").strip()

client = OpenAI(api_key=OPENAI_API_KEY)

# ----------- Simple in-memory session store -----------
# NOTE: On Render free tiers, the process may restart; for production, persist sessions externally.
_SESSIONS: Dict[str, Dict[str, Any]] = {}

def _new_session_id() -> str:
    return str(uuid.uuid4())

def _base_url(req: Request) -> str:
    # Use public URL Render gives to the request, not localhost
    return str(req.base_url).rstrip("/")

# ----------- Agent instructions -----------
AGENT_SYSTEM_PROMPT = """You are the Census Data Greeter Agent.
Goal: transform a user-supplied census source file into the format required by a carrier template using three tools:
- profile_source() → analyze files, return recipe_draft + questions
- transform_census(recipe_json, auto_approve) → masked preview (≤20 rows) + stats
- export_census(recipe_json) → final Excel file

Rules (binding):
• Never display raw PII. Mask SSNs as ***-**-1234; mask phones and emails similarly.
• Previews must be capped at 20 rows.
• Do not fabricate fields/values. If unsure, ask a concise clarifier.
• Only export after explicit user approval.

Workflow:
1) Wait until TWO files are present (source + template).
2) Call profile_source. If questions are returned, ask them one-by-one; merge answers into the recipe.
3) Build a strict Recipe JSON (valid keys only). Normalize field names and tiers (EE/ES/EC/EF/Waived).
4) Call transform_census(auto_approve:true) and show a masked 20-row preview. Ask for approval.
5) If user approves, call export_census and return completion.

Be concise. Summarize decisions. If a field is missing and required, plainly state the blocker and what you need.
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
            "description": "Exports the full transformed dataset to the carrier’s template using the final recipe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_json": { "type": "string" }
                },
                "required": ["recipe_json"],
                "additionalProperties": False
            },
        },
    },
]

# ----------- Helpers: sanitize + tool execution -----------
def _sanitize_messages(raw: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Ensure messages obey Chat Completions ordering:
    - Keep only valid roles: system/user/assistant/tool
    - Drop stray 'tool' messages that aren't responding to the immediately prior assistant tool_calls
    - Ensure we always start with a system message (AGENT_SYSTEM_PROMPT)
    """
    msgs: List[Dict[str, Any]] = []
    for m in raw:
        role = m.get("role")
        if role not in ("system", "user", "assistant", "tool"):
            continue

        if role == "tool":
            # keep only if last message is assistant with tool_calls referencing this tool_call_id
            if not msgs:
                continue
            prev = msgs[-1]
            tcid = m.get("tool_call_id")
            if prev.get("role") == "assistant" and prev.get("tool_calls") and tcid:
                # verify tcid exists in prev tool_calls
                ids = [tc.get("id") for tc in prev["tool_calls"] if isinstance(tc, dict)]
                if tcid in ids:
                    # coerce content to string
                    content = m.get("content")
                    if not isinstance(content, str):
                        content = json.dumps(content or {})
                    msgs.append({"role": "tool", "tool_call_id": tcid, "content": content})
                # else: drop
            # else: drop stray tool msg
            continue

        keep = {"role": role, "content": m.get("content")}
        # pass through tool_calls if present on assistant
        if role == "assistant" and isinstance(m.get("tool_calls"), list):
            keep["tool_calls"] = m["tool_calls"]
        msgs.append(keep)

    # Ensure first message is system
    if not msgs or msgs[0]["role"] != "system":
        msgs = [{"role": "system", "content": AGENT_SYSTEM_PROMPT}] + msgs
    return msgs

def _file_tuple(name: str, data: bytes):
    # (filename, fileobj, content_type) — content_type omitted; server will infer
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
    # cache export bytes in session and return a download URL
    sess["last_export"] = r.content
    return {"status": "export_ready", "download_url": f"/agent/download/{sess['id']}"}

# Dispatch a single tool call from the model
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

# ----------- Agent loop -----------
def _run_agent(req: Request, sess: Dict[str, Any]) -> Dict[str, Any]:
    """
    Runs up to N tool rounds until the model stops calling tools, then returns the final assistant message.
    Also returns an optional download_url if export occurred.
    """
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
        asst = choice.message  # ChatCompletionMessage
        # Convert assistant message to dict for our own history
        asst_dict: Dict[str, Any] = {"role": asst.role, "content": asst.content}
        if getattr(asst, "tool_calls", None):
            # attach raw tool_calls for replay
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

        # If model requested tools, execute each and append tool results
        if "tool_calls" in asst_dict and asst_dict["tool_calls"]:
            for tc in asst_dict["tool_calls"]:
                result = _dispatch_tool(req, sess, tc)
                # capture export URL if present
                if isinstance(result, dict) and result.get("download_url"):
                    download_url = result["download_url"]

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result)
                })
            # continue loop so model can see the tool outputs
            continue

        # No tool calls → final assistant answer for this turn
        last_text = asst_dict.get("content") or ""
        break

    # Persist session messages (sanitized + new turns)
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
        # seed with system + a brief user nudge
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": "Two files uploaded (source + template). Please profile them and propose the recipe. Then show a masked 20-row preview before export."}
        ],
    }
    _SESSIONS[sid] = sess
    return {"session_id": sid, "status": "ready"}

@router.post("/message")
async def message(request: Request, payload: Dict[str, Any]):
    sid = (payload or {}).get("session_id")
    text = (payload or {}).get("text", "").strip()
    if not sid or sid not in _SESSIONS:
        return JSONResponse({"error": "invalid session_id"}, status_code=400)

    sess = _SESSIONS[sid]
    if text:
        sess["messages"].append({"role": "user", "content": text})

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
