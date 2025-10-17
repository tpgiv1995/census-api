# agent_router.py
# Full, drop-in FastAPI router that:
# - Loads knowledge files and builds an in-memory embedding index (RAG)
# - Drives an AI-centric flow via Chat Completions + function calling
# - Calls your existing /transform and /export endpoints for execution
#
# Endpoints exposed:
#   POST /agent/start         -> creates a session_id
#   POST /agent/message       -> chat turn; accepts text + (first turn) files
#   POST /agent/confirm       -> user approval to export final file
#
# Frontend calls /agent/message in a loop to run the whole flow.

import os, io, re, json, uuid, time, shutil, glob
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel
from openai import OpenAI

# ----------------- CONFIG -----------------
OPENAI_API_KEY      = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL        = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_EMBED_MODEL  = os.environ.get("OPENAI_EMBED_MODEL", "text-embedding-3-small")
TRANSFORM_API_BASE  = os.environ.get("TRANSFORM_API_BASE", "").rstrip("/")
ASSISTANT_NAME      = os.environ.get("ASSISTANT_NAME", "Census Data Agent")
CORS_ALLOW_ORIGINS  = os.environ.get("CORS_ALLOW_ORIGINS", "*")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

if not TRANSFORM_API_BASE:
    raise RuntimeError("TRANSFORM_API_BASE is not set (e.g. https://your-render-app.onrender.com)")

client = OpenAI(api_key=OPENAI_API_KEY)
router = APIRouter(prefix="/agent", tags=["agent"])

# ----------------- SIMPLE SESSION STORE -----------------
@dataclass
class Session:
    id: str
    created_at: float = field(default_factory=lambda: time.time())
    source_path: Optional[str] = None
    template_path: Optional[str] = None
    final_recipe: Optional[dict] = None
    preview: Optional[List[dict]] = None
    state: str = "await_files"  # await_files -> profiled -> asking -> ready_to_transform -> preview_ready -> awaiting_confirm -> exported
    messages: List[Dict[str, Any]] = field(default_factory=list)  # chat messages for continuity
    clarifying_questions: List[str] = field(default_factory=list)

SESSIONS: Dict[str, Session] = {}

# ----------------- KNOWLEDGE INDEX (RAG) -----------------
# Load + embed text chunks at startup using OpenAI embeddings.
# For brevity, we parse only text-like files; you can expand parsers for PDFs later.

class KnowledgeIndex:
    def __init__(self):
        self.docs: List[Dict[str, Any]] = []   # {id, source, text, vector: np.ndarray}
        self.ready = False

    def load(self, base_dirs: List[str]):
        texts = []
        for base in base_dirs:
            for path in glob.glob(os.path.join(base, "**", "*"), recursive=True):
                if not os.path.isfile(path):
                    continue
                ext = os.path.splitext(path)[1].lower()
                if ext in [".md", ".txt", ".json", ".csv"]:
                    try:
                        if ext == ".csv":
                            # only take header + first few lines as knowledge
                            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                                lines = f.read().splitlines()[:20]
                            txt = "\n".join(lines)
                        else:
                            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                                txt = f.read()
                        texts.append({"source": path, "text": txt})
                    except Exception:
                        pass

        # chunk simple by length
        chunks = []
        for item in texts:
            t = item["text"]
            source = item["source"]
            # chunk ~1200 characters
            step = 1200
            for i in range(0, len(t), step):
                chunk = t[i:i+step]
                if chunk.strip():
                    chunks.append({"source": source, "text": chunk})

        # embed
        embeddings = []
        batch = [c["text"] for c in chunks]
        # chunk batches of 100
        for i in range(0, len(batch), 100):
            slice_texts = batch[i:i+100]
            resp = client.embeddings.create(
                model=OPENAI_EMBED_MODEL,
                input=slice_texts
            )
            for j, d in enumerate(resp.data):
                vec = np.array(d.embedding, dtype=np.float32)
                self.docs.append({
                    "id": f"kb_{i}_{j}",
                    "source": chunks[i+j]["source"],
                    "text": chunks[i+j]["text"],
                    "vector": vec
                })
        self.ready = True
        print(f"[KB] Loaded {len(self.docs)} chunks.")

    def search(self, query: str, k: int = 4) -> List[Dict[str, str]]:
        if not self.ready or not self.docs:
            return []
        q_emb = client.embeddings.create(model=OPENAI_EMBED_MODEL, input=[query]).data[0].embedding
        q_vec = np.array(q_emb, dtype=np.float32)
        # cosine similarity
        sims = []
        for doc in self.docs:
            denom = (np.linalg.norm(doc["vector"]) * np.linalg.norm(q_vec)) + 1e-9
            sim = float(np.dot(doc["vector"], q_vec) / denom)
            sims.append((sim, doc))
        sims.sort(key=lambda x: x[0], reverse=True)
        out = []
        for _, doc in sims[:k]:
            out.append({
                "source": doc["source"],
                "content": doc["text"][:2000]  # cap
            })
        return out

KB = KnowledgeIndex()
KB.load(["knowledge/census_agent_kb", "knowledge/project_resources"])

# ----------------- HELPERS -----------------
def ensure_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def save_upload(u: UploadFile, dest_path: str):
    ensure_dir(dest_path)
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(u.file, f)
    return dest_path

def sniff_headers(path: str) -> List[str]:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path, nrows=0)
        return list(df.columns)
    elif ext in [".csv", ".txt"]:
        df = pd.read_csv(path, nrows=0)
        return list(df.columns)
    else:
        return []

SSN_PAT   = re.compile(r"\b\d{3}-\d{2}-\d{4}\b|\b\d{9}\b")
PHONE_PAT = re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?){2}\d{4}\b")
MAIL_PAT  = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")

def mask_value(v: str) -> str:
    if not v or not isinstance(v, str):
        return v
    v = SSN_PAT.sub(lambda m: f"--{m.group()[-4:]}", v)
    v = PHONE_PAT.sub(lambda m: "() -"+m.group()[-4:], v)
    v = MAIL_PAT.sub(lambda m: m.group()[0] + "@d*.com", v)
    return v

def mask_preview_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in rows[:20]:  # hard cap 20 rows
        out.append({k: mask_value(str(v)) for k, v in r.items()})
    return out

# ----------------- AI SYSTEM PROMPT -----------------
SYSTEM_PROMPT = f"""
You are {ASSISTANT_NAME}. You orchestrate transforming a user’s census source file into a carrier’s template using tools.
Follow these rules strictly:
• Never display raw PII (mask SSN/phone/email). Show max 20 preview rows.
• Ask clarifying questions only when needed.
• Build a strict Recipe JSON aligned with Recipe_Schema.json (use allowed ops only).
• Use knowledge via the search_docs tool. Canonical: files under knowledge/census_agent_kb. Supporting: knowledge/project_resources.
• If the template implies a different shape (dependents or plan columns vs rows), include reshape steps in the recipe.
• When ready, return your result by calling the function return_profile_result with:
  - recipe_json (string)
  - clarifying_questions (array of strings; empty if none)
"""

# ----------------- TOOL (FUNCTION) SCHEMAS -----------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_docs",
            "description": "Search the in-app knowledge base for schema, rules, synonyms, coverage codes, mapping guidance, etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": { "type": "string" }
                },
                "required": ["query"],
                "additionalProperties": False
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "return_profile_result",
            "description": "Call this once you have a draft or final recipe and any clarifying questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "recipe_json": { "type": "string" },
                    "clarifying_questions": {
                        "type": "array",
                        "items": { "type": "string" }
                    }
                },
                "required": ["recipe_json", "clarifying_questions"],
                "additionalProperties": False
            }
        }
    }
]

# ----------------- AI LOOP (FUNCTION CALLING) -----------------
def run_ai(messages: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Runs Chat Completions with tool-calling until the model returns return_profile_result.
    """
    while True:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto"
        )
        msg = resp.choices[0].message

        # New tools format uses msg.tool_calls; older "functions" used msg.function_call
        tool_calls = getattr(msg, "tool_calls", None)
        if tool_calls:
            for call in tool_calls:
                fname = call.function.name
                fargs = json.loads(call.function.arguments or "{}")
                if fname == "search_docs":
                    hits = KB.search(fargs.get("query", ""))
                    # Return minimal, prefixed with source for traceability
                    snippet = "\n\n".join([f"Source: {h['source']}\n{h['content']}" for h in hits]) or "No results."
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": "search_docs",
                        "content": snippet
                    })
                elif fname == "return_profile_result":
                    # Final handoff
                    return fargs
            # loop again with appended tool outputs
            continue

        # If model replied with plain text, append and loop (it should eventually call return_profile_result)
        messages.append({"role": "assistant", "content": msg.content or ""})
        # Safety net: if no tool calls and the model is just chatting, ask it to finalize.
        messages.append({
            "role": "user",
            "content": "Please finalize now by calling return_profile_result."
        })

# ----------------- HTTP MODELS -----------------
class MsgIn(BaseModel):
    session_id: Optional[str] = None
    text: Optional[str] = ""

class ConfirmIn(BaseModel):
    session_id: str

# ----------------- ROUTES -----------------
@router.post("/start")
def start_session():
    sid = str(uuid.uuid4())
    SESSIONS[sid] = Session(id=sid)
    return {"session_id": sid}

@router.post("/message")
async def message(
    request: Request,
    session_id: Optional[str] = Form(None),
    text: Optional[str] = Form(""),
    source_file: Optional[UploadFile] = File(None),
    template_file: Optional[UploadFile] = File(None),
):
    # establish session
    if not session_id:
        session_id = str(uuid.uuid4())
        SESSIONS[session_id] = Session(id=session_id)
    if session_id not in SESSIONS:
        SESSIONS[session_id] = Session(id=session_id)
    s = SESSIONS[session_id]

    # Save uploads if provided
    base_dir = f"/tmp/census_sessions/{session_id}/"
    if source_file:
        s.source_path = save_upload(source_file, os.path.join(base_dir, f"source{os.path.splitext(source_file.filename)[1]}"))
    if template_file:
        s.template_path = save_upload(template_file, os.path.join(base_dir, f"template{os.path.splitext(template_file.filename)[1]}"))

    # If first turn: require both files
    if s.state == "await_files":
        if not (s.source_path and s.template_path):
            return {"session_id": session_id, "agent": "Please upload BOTH files: the census source file and the carrier template."}
        # Build a concise user context for the model
        src_headers = sniff_headers(s.source_path)
        tpl_headers = sniff_headers(s.template_path)
        file_context = f"""
User uploaded two files.
Source file headers (count {len(src_headers)}): {', '.join(src_headers[:50])}
Template file headers (count {len(tpl_headers)}): {', '.join(tpl_headers[:50])}
"""
        s.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": file_context + "\nProfile the source vs template and prepare a draft recipe. Use search_docs if helpful."}
        ]
        result = run_ai(s.messages)
        # result: {recipe_json, clarifying_questions}
        try:
            recipe_obj = json.loads(result.get("recipe_json", "{}"))
        except Exception:
            recipe_obj = {}
        s.final_recipe = recipe_obj if not result.get("clarifying_questions") else None
        s.clarifying_questions = result.get("clarifying_questions", [])
        if s.clarifying_questions:
            s.state = "asking"
            return {
                "session_id": session_id,
                "agent": "I need a bit more information before transforming:",
                "questions": s.clarifying_questions
            }
        else:
            s.state = "ready_to_transform"
            # call preview right away
            preview = call_transform_preview(s, recipe_obj)
            s.preview = preview.get("preview", [])
            s.state = "preview_ready"
            return {
                "session_id": session_id,
                "agent": "Here’s a masked preview (max 20 rows). Does this look correct? Say 'yes' to export, or describe changes.",
                "stats": preview.get("stats", {}),
                "preview": mask_preview_rows(preview.get("preview", []))
            }

    # If asking clarifiers, the user's text must contain answers
    if s.state == "asking":
        # Give the model the Q&As to finalize the recipe
        qa_text = f"User provided answers to the prior questions:\n{text}\nPlease finalize the recipe and call return_profile_result."
        s.messages.append({"role": "user", "content": qa_text})
        result = run_ai(s.messages)
        try:
            recipe_obj = json.loads(result.get("recipe_json", "{}"))
        except Exception:
            recipe_obj = {}
        s.final_recipe = recipe_obj
        s.clarifying_questions = []
        s.state = "ready_to_transform"
        preview = call_transform_preview(s, recipe_obj)
        s.preview = preview.get("preview", [])
        s.state = "preview_ready"
        return {
            "session_id": session_id,
            "agent": "Thanks. Here’s the masked preview (max 20 rows). Approve with 'yes' to export, or describe adjustments.",
            "stats": preview.get("stats", {}),
            "preview": mask_preview_rows(preview.get("preview", []))
        }

    # If preview ready: look for an approval cue
    if s.state == "preview_ready":
        if text.strip().lower() in ["yes", "y", "ok", "export", "go", "go ahead", "looks good", "approved"]:
            s.state = "awaiting_confirm"
            # proceed to export now
            export_path = call_export(s, s.final_recipe or {})
            s.state = "exported"
            # expose a one-time download route
            return {
                "session_id": session_id,
                "agent": "✅ Export complete. Click to download your transformed file.",
                "download_url": f"/agent/download/{session_id}"
            }
        else:
            # non-approval: ask model to adjust?
            s.messages.append({"role": "user", "content": f"User feedback on preview: {text}. Update the recipe and return_profile_result."})
            result = run_ai(s.messages)
            try:
                recipe_obj = json.loads(result.get("recipe_json", "{}"))
            except Exception:
                recipe_obj = {}
            s.final_recipe = recipe_obj
            preview = call_transform_preview(s, recipe_obj)
            s.preview = preview.get("preview", [])
            s.state = "preview_ready"
            return {
                "session_id": session_id,
                "agent": "Updated preview. Approve with 'yes' to export.",
                "stats": preview.get("stats", {}),
                "preview": mask_preview_rows(preview.get("preview", []))
            }

    # Fallback
    return {"session_id": session_id, "agent": "I’m ready. Upload both files to begin, or say 'help'."}

@router.get("/download/{session_id}")
def download(session_id: str):
    s = SESSIONS.get(session_id)
    if not s or s.state != "exported":
        raise HTTPException(status_code=404, detail="No export available.")
    path = f"/tmp/census_sessions/{session_id}/export.xlsx"
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path, filename="Transformed_Census.xlsx", media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ----------------- INTERNAL: CALL BACKEND TOOLS -----------------
def call_transform_preview(s: Session, recipe: dict) -> dict:
    url = f"{TRANSFORM_API_BASE}/transform?auto_approve=true"
    files = {
        "source_file": open(s.source_path, "rb"),
        "template_file": open(s.template_path, "rb")
    }
    data = {
        "recipe": json.dumps(recipe)
    }
    try:
        r = requests.post(url, files=files, data=data, timeout=120)
        r.raise_for_status()
        payload = r.json()
        # expect payload {"preview":[...], "stats":{...}, "used_mapping":{...}}
        return payload
    finally:
        for f in files.values():
            try:
                f.close()
            except Exception:
                pass

def call_export(s: Session, recipe: dict) -> str:
    url = f"{TRANSFORM_API_BASE}/export"
    files = {
        "source_file": open(s.source_path, "rb"),
        "template_file": open(s.template_path, "rb")
    }
    data = {
        "recipe_json": json.dumps(recipe)
    }
    out_path = f"/tmp/census_sessions/{s.id}/export.xlsx"
    ensure_dir(out_path)
    try:
        with requests.post(url, files=files, data=data, timeout=300, stream=True) as r:
            r.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        return out_path
    finally:
        for f in files.values():
            try:
                f.close()
            except Exception:
                pass
