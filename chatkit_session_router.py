# chatkit_session_router.py
from fastapi import APIRouter, HTTPException
from openai import OpenAI
import os

router = APIRouter(prefix="/api/chatkit", tags=["chatkit"])

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")

# IMPORTANT: paste your published Agent Builder workflow id here (looks like wf_...')
WORKFLOW_ID = os.environ.get("OPENAI_WORKFLOW_ID", "").strip()
if not WORKFLOW_ID:
    # You can also hardcode here, but env var is nicer:
    # WORKFLOW_ID = "wf_abc123..."
    raise RuntimeError("OPENAI_WORKFLOW_ID is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

@router.post("/session")
def create_chatkit_session():
    """
    Create a ChatKit session for the published workflow and return the client_secret token.
    The browser uses this token to open the embedded chat.
    """
    try:
        # NOTE: The sessions.create call name/shape comes from the current OpenAI SDK.
        # If your SDK is older, upgrade `openai` per the requirements.txt step above.
        session = client.chatkit.sessions.create({
            "workflow_id": WORKFLOW_ID,
            # You can pass optional per-user identity/metadata here if you want.
            # "user": {"id": "pat-gordon"},
            # Enable file upload UI in the chat (good for your two-file flow)
            "chatkit_configuration": {"file_upload": {"enabled": True}}
        })
        return {"client_secret": session.client_secret}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ChatKit session error: {e}")
