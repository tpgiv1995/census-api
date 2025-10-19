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
    try:
        print("Creating ChatKit session with workflow:", WORKFLOW_ID)

        session = client.chatkit.sessions.create({
            "workflow_id": WORKFLOW_ID,
            "chatkit_configuration": {
                "file_upload": { "enabled": True }
            }
        })

        print("Session created:", session)
        return { "client_secret": session.client_secret }

    except Exception as e:
        print("ChatKit session error:", e)
        raise HTTPException(status_code=500, detail=f"ChatKit session error: {e}")


