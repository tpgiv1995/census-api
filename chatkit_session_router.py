from fastapi import APIRouter, HTTPException
from openai import OpenAI
import os
import traceback

router = APIRouter(prefix="/api/chatkit", tags=["chatkit"])

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
WORKFLOW_ID = os.environ.get("OPENAI_WORKFLOW_ID", "").strip()

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set")
if not WORKFLOW_ID:
    raise RuntimeError("OPENAI_WORKFLOW_ID is not set")

client = OpenAI(api_key=OPENAI_API_KEY)

@router.post("/session")
def create_chatkit_session():
    try:
        print("🔧 ChatKit session creation started")
        print("🔑 OPENAI_API_KEY starts with:", repr(OPENAI_API_KEY[:10]))
        print("🧬 OPENAI_WORKFLOW_ID:", repr(WORKFLOW_ID))

        session = client.chatkit.sessions.create({
            "workflow_id": WORKFLOW_ID,
            "chatkit_configuration": {
                "file_upload": {"enabled": True}
            }
        })

        print("✅ ChatKit session created:", session)
        return {"client_secret": session.client_secret}

    except Exception as e:
        print("❌ ChatKit session creation FAILED")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"ChatKit session error: {e}")
