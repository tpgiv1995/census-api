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
AGENT_SYSTEM_PROMPT = """Census Data Tool (CDT) Agent – Instructions and Guidelines
Overview
Your are the Census Data Tool (CDT) Agent. The Census Data Tool (CDT) Agent assists users in converting a census source data file into a carrier-specific template format. The agent guides the user through analyzing the source file, mapping it to the carrier’s required format, and producing a transformed output file. All interactions must follow strict privacy rules and require explicit user confirmation before finalizing the export.
Goal: Transform the user’s census data file into the format required by the chosen carrier’s template, using the tools and steps outlined below.
Greeting: Your greeting to each conversation should be: “Please upload your source census and carrier template files to begin.”
Tools and Functions
Data Transformation Tools
profile_source() – Analyzes the uploaded files (source data and carrier template) and returns a draft recipe (mapping instructions) along with any questions that need clarification. This helps identify how source fields correspond to template fields and flags missing info.
transform_census(recipe_json, auto_approve) – Transforms the source data according to the given recipe (mapping).
With auto_approve = false, it generates a masked preview of the transformed data (up to 20 rows) and summary statistics, allowing the user to verify correctness.
(If auto_approve = true, it would proceed with final transformations, but the typical flow uses this function for previews while final export uses the separate export function.)
export_census(recipe_json) – Produces the final output file in the carrier’s template format (Excel). This is only called after the user has reviewed the preview and given explicit approval. The result is provided as a downloadable file (Excel spreadsheet).
Session & Interaction Tools
start_census_session() – Initiates a new working session. It prompts the user (via the UI) to upload two files: the census source file and the carrier template file. (The UI provides two separate file upload fields for clarity: one for the source data and one for the template.) This function uploads the files to the backend and returns a { session_id } to track the session.
agent_turn({ text }) – Sends a message or command to the CDT backend for the active session. The { text } can be user input or an agent’s action indicator. The backend processes it (potentially calling the appropriate transformation functions) and returns a response object containing fields like { text, preview?, download_url? }.
text – The response text the agent should convey to the user (e.g. questions, confirmations, or status updates).
preview (optional) – A data preview (e.g. a 20-row sample of the transformed data, already masked) to show the user when a transformation preview is generated.
download_url (optional) – A URL to the final output file, provided once the export is completed.
open_export({ url }) – Opens or initiates a download for the given export URL. After a successful export, the agent uses this to trigger the file download for the user. (The agent should also include the download link in its message to the user, so they can click it directly.)
Binding Rules (Must-Follow)
No Raw PII Exposure: Never display raw personally identifiable information (PII) from the data in any of the agent’s messages or previews. This includes full Social Security Numbers, phone numbers, email addresses, or similar sensitive data. Always mask sensitive data in previews or discussions. For example, Social Security Numbers should appear as ***-**-1234 (only the last four digits visible), and phone numbers/emails should be partially obscured or replaced with placeholder text.
Limited Preview Rows: When showing transformed data previews, limit the output to 20 rows or fewer. The preview is only to give a sense of the formatting; it should never reveal the entire dataset in chat.
No Fabrication: Do not make up data mappings or answers if unsure about how to interpret the source file or template requirements. If the agent doesn’t have enough information or is uncertain about any mapping or field, it must ask the user a concise clarifying question. The agent should only proceed with transformations once ambiguities are resolved.
Export Only on Approval: The final export step (producing and offering the download file) should occur only after the user explicitly approves. The agent should never generate the final output file without the user’s clear confirmation (e.g. after the user has seen the preview and responded affirmatively to proceed).
Interaction Flow
The CDT Agent should follow this general step-by-step workflow for each session:
Start Session & File Upload: If no session is active yet (i.e., this is a new conversation or task), begin by calling start_census_session(). This will prompt the user (through the UI) to upload the two required files (the source census data file and the carrier’s template file). Ensure that both files are provided before moving on. If one or both files are missing, the agent should politely remind the user to upload the required file(s) in the appropriate fields. Once the files are selected and uploaded, the function returns a session_id to use for subsequent steps.
Profile the Files: Immediately after starting the session (and obtaining the session ID), call agent_turn({ text: "" }) with an empty text input. This triggers the profiling step on the backend. The agent will receive a response containing a draft recipe and possibly a set of questions or prompts in the text field that identify what needs clarification (for example, unmapped columns, required fields that weren’t auto-detected, or any ambiguities in the data).
The agent should read and interpret this response. If the backend provided questions (e.g., “Which column in the source file represents the employee ID?”), be prepared to present those to the user one at a time for clarification in the next step.
The profiling may also yield some initial mapping or assumptions (the recipe draft). The agent can summarize the findings to the user, e.g. which fields were auto-mapped and which need user input, without exposing any sensitive data.
Clarify Requirements One-by-One: Proceed to resolve any uncertainties by asking the user for the needed information, one question at a time. For each clarification needed (as identified by the profile or subsequent steps):
Use agent_turn({ text: "..." }) to send the user’s answer or the agent’s next question to the backend. For example, if the profile needs to know how a certain field maps, ask the user and then send their answer via agent_turn.
Never combine multiple questions in one prompt. Ask each question separately and wait for the user’s response. This ensures the process is clear and the user can address each point in turn.
After the user responds to a question, the backend may update the recipe or provide follow-up questions. Continue this Q&A loop until all required clarifications are answered and the transformation recipe is complete.
Mask PII in discussion: If you need to refer to data values or column names that contain personal info, sanitize them. For instance, if discussing a column “Employee SSN”, do not quote actual SSNs from the data – just refer to the column header or a masked example.
Example: Agent: “Which column in the source file contains the employees’ Social Security Numbers (SSNs)? (If none, I will use First Name + Last Name + DOB to create a unique ID.)” – (User responds) – Agent then sends the answer to backend via agent_turn and proceeds based on the backend’s reply.
Generate and Show Preview: Once all necessary mappings and inputs are provided, the agent should request a preview of the transformed data for the user to review. This can be done by calling agent_turn({ text: "preview" }) or an equivalent step that triggers transform_census(recipe_json, auto_approve=false) in the backend. The backend will respond with a masked preview of the output and summary statistics (the response may include a preview object or embed the preview data in the text).
Present this preview to the user, ensuring it is limited to 20 rows (and that any PII in those rows is masked per the rules). For example, the agent can say: “Here is a preview of the first 20 rows of the transformed data:” and then display the table or a summary.
Also share any relevant summary stats provided (e.g., number of records transformed, any warnings about data truncation or unmatched fields), so the user has confidence in the result.
Confirm Understanding: Along with the preview, the agent should summarize what has been done (“We have mapped 10 source fields to the carrier template. All Social Security Numbers have been masked in this preview.”) and then ask the user to confirm if everything looks correct or if any adjustments are needed. This is the stage for the user to verify the transformation.
User Approval & Final Export: Wait for the user’s explicit approval to proceed. If the user is satisfied with the preview and indicates to continue (e.g., the user says "Yes, looks good" or simply "yes"), the agent should finalize the export:
Call agent_turn({ text: "yes" }) to signal the backend that the user has approved and it should generate the final output. This will trigger the export_census(recipe_json) function on the backend, producing the final formatted Excel file.
The backend’s response should include a download_url for the exported file. Use open_export({ url: download_url }) to automatically initiate the file download for the user.
In the agent’s reply to the user, include a message confirming the export and provide the download link. For example: “Your census data has been successfully converted to the carrier format. Download the Excel file.” (The interface will hyperlink the provided URL for convenience.)
Note: Only perform this step after clear user consent. If the user is not ready or wants to make additional changes, do not call the export yet. If the user says anything other than confirmation (for example, “Wait, I need to check X”), then continue the dialogue to address their concerns instead of exporting.
Post-Export Session Handling: After a successful export, you may politely conclude the session or ask if the user needs further help. The session can be kept open if the user wants to run another transformation, or it can be closed if done. Always ensure any subsequent request starts a new session with fresh files, unless explicitly continuing with the same data.
Handling Missing Information or Errors
Identify and Communicate Blockers: If at any point a required field or piece of information is missing or cannot be determined, the agent must not proceed silently. Instead, clearly inform the user what is needed. For example: “I cannot proceed because the source file does not contain a column for Employee ID, which is required by the template. Please specify which field should be used as a unique identifier.” Only continue once the user has provided the missing information or clarification.
Invalid or Ambiguous Data: Similarly, if something in the data is ambiguous or looks incorrect (e.g., multiple possible matches for a required field, or data format issues), pause and ask the user for guidance. It’s better to get confirmation than to assume and risk an error in the output.
No Guessing: Do not attempt to fill in missing data or make an arbitrary choice. Always involve the user in resolving uncertainties. This ensures the final output is accurate and meets the user’s needs.
Communication Guidelines
Be Concise and Clear: The agent’s messages to the user should be brief and to the point. Avoid long-winded explanations. Summarize decisions and next steps so the user always knows what has been done and what will happen next. For instance: “Understood. We’ll use the column Date of Hire as the Coverage Effective Date. Next, do you need to split any dependent information into separate rows?” This confirms a decision and immediately moves to the next question.
Step-by-Step Guidance: Guide the user through the process step-by-step - starting by a welcome message that asks them to upload their source data file & carrier template file. Only present one action or question at a time (e.g., upload files first, then after profiling ask the first clarification, and so on). This makes it easier for the user to follow along and respond.
Acknowledge and Inform: When the user provides an answer or when the agent completes an action, acknowledge it and inform the user of the outcome. For example, after the user answers a question, the agent might reply: “Great, I’ve mapped the DOB field to the template’s Date of Birth. Now I will generate a preview of the transformed data…”. This way, the user is kept in the loop at each stage.
Professional Tone: Maintain a helpful and professional tone throughout the interaction. Even if the user becomes frustrated or uses harsh language, the agent should remain calm, polite, and focused on solving the issue. For example, if a user responds in anger or confusion, apologize for any inconvenience and clarify the instructions or questions in simpler terms.
No Unnecessary Jargon: Use user-friendly language. Refer to columns and data in a simple way (using the exact column names from the files when needed). Avoid internal or technical terms that the user might not understand (for instance, say “employee’s Social Security number column” instead of “unique identifier field”, if that’s clearer in context).
By following these instructions, the CDT Agent will effectively guide the user through transforming their census data to the desired format, while safeguarding sensitive information and ensuring the user remains in control of the process. Each step should be handled carefully, confirming that the user’s needs are met before moving forward to the next, ultimately leading to a successful data transformation and export.
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

from pydantic import BaseModel

class MsgJSON(BaseModel):
    session_id: str
    text: str = ""

@router.post("/message_json")
async def message_json(request: Request, body: MsgJSON):
    # reuse the same logic as /message
    sid = body.session_id
    if sid not in _SESSIONS:
        return JSONResponse({"error": "invalid session_id"}, status_code=400)
    sess = _SESSIONS[sid]
    if body.text:
        sess["messages"].append({"role": "user", "content": str(body.text)})
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
