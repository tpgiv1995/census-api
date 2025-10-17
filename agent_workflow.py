# agent_workflow.py
import os, json, requests
from typing import Any, Dict, List
from dotenv import load_dotenv
from openai import OpenAI

# Load .env (OPENAI_API_KEY, CENSUS_API_BASE, OPENAI_MODEL)
load_dotenv()

API_BASE = os.getenv("CENSUS_API_BASE", "http://127.0.0.1:8000")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

if not OPENAI_API_KEY:
    raise SystemExit("OPENAI_API_KEY not set. Put it in .env or set as environment variable.")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------- HTTP helpers --------
def post_multipart(url: str, files: Dict[str, Any], data: Dict[str, Any] = None):
    try:
        r = requests.post(url, files=files, data=data or {}, timeout=120)
    finally:
        # ensure file handles are closed
        for f in files.values():
            try:
                f.close()
            except Exception:
                pass
    r.raise_for_status()
    return r

def post_json(url: str, payload: Dict[str, Any]):
    r = requests.post(url, json=payload, timeout=60)
    r.raise_for_status()
    return r

# -------- Tools (call your FastAPI) --------
def tool_profile_source(source_path: str, template_path: str) -> Dict[str, Any]:
    url = f"{API_BASE}/profile"
    files = {
        "source_file": open(source_path, "rb"),
        "template_file": open(template_path, "rb"),
    }
    return post_multipart(url, files=files).json()

def tool_transform_preview(source_path: str, template_path: str, recipe: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{API_BASE}/transform"
    files = {
        "source_file": open(source_path, "rb"),
        "template_file": open(template_path, "rb"),
    }
    data = {"recipe_json": json.dumps(recipe), "auto_approve": "true"}
    return post_multipart(url, files=files, data=data).json()

def tool_export(source_path: str, template_path: str, recipe: Dict[str, Any], carrier_name: str = "") -> bytes:
    url = f"{API_BASE}/export"
    files = {
        "source_file": open(source_path, "rb"),
        "template_file": open(template_path, "rb"),
    }
    data = {"recipe_json": json.dumps(recipe), "carrier_name": carrier_name}
    return post_multipart(url, files=files, data=data).content

def tool_feedback(carrier_name: str, carrier_headers: List[str], used_mapping: Dict[str,str], recipe: Dict[str,Any]):
    url = f"{API_BASE}/feedback"
    payload = {
        "carrier_name": carrier_name,
        "carrier_headers": carrier_headers,
        "used_mapping": used_mapping,
        "recipe": recipe,
    }
    post_json(url, payload)
    return {"status":"ok"}

SYSTEM_PROMPT = """You are the Census Transform Agent.
Workflow:
1) Ask for TWO files: Source census + Carrier template.
2) Call profile_source. If needs_questions=true, ask ONLY those questions; otherwise skip.
3) Call transform_preview and show a short, masked 20-row preview summary (never paste raw PII).
4) Ask for approval. ONLY on “yes”, call export and save XLSX. Then call feedback.
Keep messages concise. Never exceed 20 preview rows in text.
"""

def run_cli():
    print("=== Census Agent (CLI) ===")
    print("Enter full paths to your two files.")
    src = input("Source census file path: ").strip('" ')
    tpl = input("Carrier template file path: ").strip('" ')
    carrier_name = input("Optional carrier label (for learning): ").strip()

    # 1) profile
    prof = tool_profile_source(src, tpl)
    recipe = prof.get("recipe_draft", {})
    needs = prof.get("needs_questions", False)
    qs = prof.get("questions", [])

    # 2) clarification loop (only if needed)
    answers = {}
    if needs and qs:
        print("\nThe system needs a couple clarifications:")
        for q in qs:
            qid = q.get("id")
            text = q.get("text")
            opts = q.get("options", [])
            print(f"- {text}")
            if opts:
                print(f"  Options: {', '.join(opts[:10])}{' ...' if len(opts)>10 else ''}")
            ans = input("Your answer: ").strip()
            answers[qid] = ans
        # merge naive: put answers into recipe for now
        for k, v in answers.items():
            recipe[k] = v

    # 3) transform preview
    tr = tool_transform_preview(src, tpl, recipe)
    rows_in = tr["stats"]["rows_in"]
    rows_out = tr["stats"]["rows_out"]
    preview = tr.get("preview", [])
    print(f"\nPreview ready. Rows in: {rows_in} | Rows out: {rows_out}")
    # show a very short summary (not raw table)
    if preview:
        cols = list(preview[0].keys())
        print("Columns:", ", ".join(cols[:12]), ("..." if len(cols)>12 else ""))
        for r in preview[:5]:
            snippet = ", ".join([str(r.get(c,"")) for c in cols[:6]])
            print(f"- {snippet}{' ...' if len(cols)>6 else ''}")
    else:
        print("(No preview rows)")

    ok = input("\nExport the full file? (yes/no): ").strip().lower()
    if ok not in ("y","yes"):
        print("Cancelled.")
        return

    # 4) export
    xls = tool_export(src, tpl, recipe, carrier_name=carrier_name)
    out_path = os.path.join(os.getcwd(), "Transformed_Census.xlsx")
    with open(out_path, "wb") as f:
        f.write(xls)
    print(f"Exported: {out_path}")

    # 5) feedback (best effort)
    try:
        tool_feedback(
            carrier_name or "",
            tr.get("carrier_headers", []),
            tr.get("used_mapping", {}),
            recipe
        )
    except Exception as e:
        print(f"(feedback warning) {e}")

if __name__ == "__main__":
    run_cli()
