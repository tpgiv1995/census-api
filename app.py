from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional, Tuple
import io, os, re, json
import pandas as pd

# ---------------- App, CORS & static/templates ----------------
app = FastAPI(title="Census Engine", version="1.4")

# CORS: allow your Netlify front-end (or * while testing)
allow_origins = [o.strip() for o in os.environ.get("CORS_ALLOW_ORIGINS", "*").split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ---- Mount the AI agent router (drives recipe creation & transform/export orchestration)
from agent_router import router as agent_router
app.include_router(agent_router)

# ---------------- RAG-lite knowledge store ----------------
KNOWLEDGE_PATH = "data/census_knowledge.json"
os.makedirs("data", exist_ok=True)

def _load_ks():
    if os.path.exists(KNOWLEDGE_PATH):
        with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "header_map_stats": {},     # "src||tgt" -> {"count": int}
        "synonyms": {},             # token -> [tokens]
        "templates": [],            # [{carrier_name, headers_tokens}]
        "recipes": []               # [{carrier_name, recipe}]
    }

def _save_ks(data):
    with open(KNOWLEDGE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

KS = _load_ks()

def _tokenize(s: str) -> List[str]:
    # normalize by removing non-alphanumeric (so "D.O.B." => "dob")
    cleaned = re.sub(r"[^a-z0-9]+", " ", (s or "").strip().lower())
    return [t for t in cleaned.split() if t]

def _jaccard(a:set,b:set)->float:
    return 1.0 if not a and not b else len(a & b)/max(1,len(a|b))

def ks_record_success(carrier_name:str, carrier_headers:List[str], used_mapping:Dict[str,str], recipe:Dict[str,Any]):
    for tgt, src in used_mapping.items():
        key = f"{src}||{tgt}"
        KS["header_map_stats"][key] = {"count": KS["header_map_stats"].get(key,{"count":0})["count"]+1}
        toks = set(_tokenize(src)) | set(_tokenize(tgt))
        for tok in toks:
            bucket = set(KS["synonyms"].get(tok, []))
            KS["synonyms"][tok] = sorted(bucket | toks)
    KS["templates"].append({
        "carrier_name": carrier_name or "unknown",
        "headers_tokens": sorted({t for h in carrier_headers for t in _tokenize(h)})
    })
    KS["recipes"].append({"carrier_name": carrier_name or "unknown", "recipe": recipe})
    _save_ks(KS)

def ks_suggest_mapping(src_cols:List[str], carrier_headers:List[str]) -> Dict[str,str]:
    per_tgt: Dict[str,Dict[str,int]] = {}
    for key, payload in KS.get("header_map_stats",{}).items():
        src, tgt = key.split("||",1)
        per_tgt.setdefault(tgt, {})
        per_tgt[tgt][src] = per_tgt[tgt].get(src,0) + payload.get("count",0)
    suggestions = {}
    src_set = set(src_cols)
    for tgt in carrier_headers:
        if tgt in per_tgt:
            # pick the most-frequent learned src that exists this run
            best_src = None; best_count = -1
            for s,count in sorted(per_tgt[tgt].items(), key=lambda kv: kv[1], reverse=True):
                if s in src_set:
                    best_src = s; best_count = count; break
            if best_src:
                suggestions[tgt] = best_src
                continue
        t_tokens = set(_tokenize(tgt))
        syn = set()
        for tok in list(t_tokens):
            syn |= set(KS.get("synonyms",{}).get(tok,[]))
        best, best_score = None, 0.0
        for s in src_cols:
            score = len((t_tokens|syn) & set(_tokenize(s))) / max(1,len(t_tokens|syn))
            if score > best_score:
                best, best_score = s, score
        if best and best_score >= 0.6:
            suggestions[tgt] = best
    return suggestions

def ks_template_boost(carrier_headers:List[str]) -> float:
    if not KS["templates"]:
        return 0.0
    tgt = set({t for h in carrier_headers for t in _tokenize(h)})
    best = max((_jaccard(tgt, set(t["headers_tokens"])) for t in KS["templates"]), default=0.0)
    return 0.15 * best  # up to +0.15 confidence

# ---------------- File IO ----------------
PREFERRED_ENCODINGS = ["utf-8-sig","utf-16","cp1252","latin1"]

def read_any_table(filename:str, data:bytes)->pd.DataFrame:
    lower = (filename or "").lower()
    if lower.endswith((".xlsx",".xls")):
        return pd.read_excel(io.BytesIO(data))
    last = None
    for enc in PREFERRED_ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(data), encoding=enc)
        except Exception as e:
            last = e
    try:
        return pd.read_excel(io.BytesIO(data))
    except Exception as e:
        raise RuntimeError(f"Unable to read: {filename}. Last error: {last or e}")

def norm_headers(df:pd.DataFrame)->pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+"," ",str(c)).strip() for c in df.columns]
    return df

# ---------------- Detection heuristics ----------------
SSN_PAT = re.compile(r"^\*{3}-\*{2}-\d{4}$|^\d{3}-\d{2}-\d{4}$|^\d{9}$")

def looks_like_ssn_series(s:pd.Series)->float:
    return (s.astype(str).str.strip().str.match(SSN_PAT)).mean()

def header_hit(name:str, keys:List[str])->float:
    n = (name or "").lower()
    return 1.0 if any(k in n for k in keys) else 0.0

def detect_employee_ssn(df:pd.DataFrame)->Tuple[Optional[str], float]:
    header_hits = [c for c in df.columns if header_hit(c, ["employee ssn","ee ssn","emp ssn","ssn (employee)"])>0]
    if header_hits:
        best = max(header_hits, key=lambda c: looks_like_ssn_series(df[c]))
        return best, 0.95
    cands = [(c, looks_like_ssn_series(df[c])) for c in df.columns]
    cands = [x for x in cands if x[1] >= 0.6]
    if not cands:
        return None, 0.0
    best = max(cands, key=lambda x: x[1])[0]
    return best, 0.8

def detect_second_ssn(df:pd.DataFrame, primary:str)->Optional[str]:
    # Look for "member/individual ssn" or generic "ssn" not equal to primary
    cand_names = [c for c in df.columns if c != primary and (
        "member ssn" in c.lower() or "individual ssn" in c.lower() or c.lower().strip() == "ssn"
    )]
    if not cand_names:
        ssn_cols = [c for c in df.columns if c != primary and looks_like_ssn_series(df[c]) >= 0.6]
        cand_names = ssn_cols
    return cand_names[0] if cand_names else None

def detect_relationship(df:pd.DataFrame)->Tuple[Optional[str], List[str], float]:
    rel_cols = [c for c in df.columns if any(k in c.lower() for k in ["relationship","rel"])]
    if rel_cols:
        vals = pd.unique(df[rel_cols[0]].astype(str).str.strip())
        tokens = [v for v in vals if v.lower() in ["employee","ee","emp","subscriber","staff"]]
        return rel_cols[0], (tokens or ["Employee","EE"]), 0.9 if tokens else 0.75
    return None, ["Employee","EE"], 0.0

def detect_dependents_structure(df:pd.DataFrame)->Tuple[str,float]:
    cols = [c.lower() for c in df.columns]
    dep_pattern = any(re.match(r"^dep[\s_\-]*\d+", cn) or cn.startswith("dependent") for cn in cols)
    if dep_pattern: return "column_based", 0.95
    return "row_based", 0.70

def detect_plan_per_row(df:pd.DataFrame, anchor_col:Optional[str])->Tuple[str,Optional[str],List[str],float]:
    if anchor_col and df[anchor_col].duplicated().any():
        pt_candidates = [c for c in df.columns if any(k in c.lower() for k in
            ["benefit","plan type","coverage type","product","line of coverage","lob"])]
        attrs = [c for c in df.columns if any(k in c.lower() for k in
            # Include election/coverage/tier in attributes
            ["election","coverage","tier","carrier","plan id","policy","effective","amount","premium","class","option"])]
        return "plan_per_row", (pt_candidates[0] if pt_candidates else None), attrs, 0.85
    return "plan_per_record", None, [], 0.70

def read_template_headers(template:UploadFile, raw:bytes)->List[str]:
    df = read_any_table(template.filename, raw)
    df = norm_headers(df)
    return [str(c).strip() for c in df.columns]

def simple_best_match(src_cols:List[str], tgt:str)->Optional[str]:
    # compare on token sets (punctuation-insensitive) for robust matches like DOB <-> D.O.B.
    t_tokens = set(_tokenize(tgt))
    best, best_score = None, 0.0
    for c in src_cols:
        score = len(t_tokens & set(_tokenize(c))) / max(1, len(t_tokens))
        if score > best_score:
            best, best_score = c, score
    return best if best_score >= 0.6 else None

def map_to_headers(df_out:pd.DataFrame, carrier_headers:List[str])->Tuple[pd.DataFrame, Dict[str,str]]:
    src_cols = list(df_out.columns)
    learned = ks_suggest_mapping(src_cols, carrier_headers)
    final_map = {}
    for h in carrier_headers:
        if h in learned:
            final_map[h] = learned[h]
        else:
            m = simple_best_match(src_cols, h)
            if m: final_map[h] = m
    mapped = {}
    for h in carrier_headers:
        mapped[h] = df_out[final_map[h]] if h in final_map else pd.Series([""]*len(df_out))
    return pd.DataFrame(mapped), {h:s for h,s in final_map.items()}

# ---------------- Normalization helpers ----------------
def normalize_election(val:str)->str:
    s = (val or "").strip().lower()
    if not s: return ""
    # canonical long-form
    if s in ("ee","employee","emp","employee only","single"): return "Employee"
    if s in ("es","ee+spouse","employee+spouse","employee + spouse","emp+spouse"): return "Employee+Spouse"
    if s in ("ec","ee+child","employee+child","employee + child","ee+children","employee+children","employee + children","emp+children","employee + child(ren)","employee+child(ren)"): 
        return "Employee+Children"
    if s in ("ef","ee+family","employee+family","employee + family","family"): return "Employee+Family"
    if "waiv" in s or s=="decline" or s=="declined": return "Waived"
    # fallback best-effort
    s2 = re.sub(r"\s+","",s)
    if s2.startswith("employee+spouse"): return "Employee+Spouse"
    if s2.startswith("employee+child"): return "Employee+Children"
    if s2.startswith("employee+children"): return "Employee+Children"
    if s2.startswith("employee+family"): return "Employee+Family"
    if s2.startswith("employee"): return "Employee"
    return (val or "").strip()

def pick_election_column(cols:List[str])->Optional[str]:
    # prefer explicit 'election', then 'coverage', then 'tier'
    order = ["election", "coverage", "tier"]
    lowered = [c.lower() for c in cols]
    for key in order:
        matches = [c for c in cols if key in c.lower()]
        if matches:
            # Prefer shortest, simplest label (often the intended signal)
            return sorted(matches, key=lambda x: (len(x), x.lower())).pop(0)
    return None

# ---------------- Masking ----------------
def mask_preview_value(col:str, val:str)->str:
    s = str(val or "")
    lc = (col or "").lower()
    if "ssn" in lc or "social" in lc:
        digits = re.sub(r"\D","",s)
        return f"***-**-{digits[-4:]}" if len(digits)>=4 else "***-**-****"
    if "phone" in lc or re.search(r"\d{3}[\s\-]?\d{3}[\s\-]?\d{4}", s):
        digits = re.sub(r"\D","",s)
        return f"(***) ***-{digits[-4:]}" if len(digits)>=4 else "***-***-****"
    if "@" in s:
        try:
            user, dom = s.split("@",1)
            dom_name, *tld = dom.split(".")
            return f"{(user[:1] or '*')}***@{(dom_name[:1] or '*')}******.{'.'.join(tld) if tld else 'com'}"
        except:
            return "p***@e******.com"
    return s

# ---------------- UI routes ----------------
@app.get("/", response_class=HTMLResponse)
async def home(request:Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/ui", response_class=HTMLResponse)
async def ui(request:Request):
    return templates.TemplateResponse("index.html", {"request": request})

# ---------------- PROFILE ----------------
CONF_CUTOFF = 0.80

@app.post("/profile")
async def profile(
    source_file: UploadFile = File(...),
    template_file: UploadFile = File(...)
):
    src_raw = await source_file.read()
    tmp_raw = await template_file.read()
    df = norm_headers(read_any_table(source_file.filename, src_raw))
    carrier_headers = read_template_headers(template_file, tmp_raw)

    emp_ssn_col, emp_conf = detect_employee_ssn(df)
    if emp_ssn_col:
        anchor = {"type":"employee_ssn","field":emp_ssn_col,"confidence":emp_conf}
        key = emp_ssn_col
    else:
        anchor = {"type":"composite","field":None,"fallback_composite":["FirstName","LastName","DOB"],"confidence":0.60}
        key = None

    rel_field, emp_vals, rel_conf = detect_relationship(df)
    dep_mode, dep_conf = detect_dependents_structure(df)
    plan_mode, plan_type, plan_attrs, plan_conf = detect_plan_per_row(df, key)

    recipe = {
        "version":"1.0",
        "anchor": anchor,
        "relationship": {"field": rel_field, "employee_values": emp_vals, "confidence": rel_conf},
        "structure": {
            "dependents": dep_mode,
            "plans": plan_mode,
            "plan_type_field": plan_type,
            "plan_attrs": plan_attrs[:6],
            "confidence": min(dep_conf, plan_conf)
        },
        "mapping": {"employee":{}, "dependent":{}, "plan_columns":{}},
        "export": {"layout":"Row-Based","include_dependents":True,"columns_order":[]}
    }

    boost = ks_template_boost(carrier_headers)
    recipe["anchor"]["confidence"] = min(0.98, recipe["anchor"]["confidence"]+boost)
    recipe["relationship"]["confidence"] = min(0.98, recipe["relationship"]["confidence"]+boost/2)
    recipe["structure"]["confidence"] = min(0.98, recipe["structure"]["confidence"]+boost)

    questions = []
    if recipe["anchor"]["type"]=="composite" and recipe["anchor"]["confidence"]<CONF_CUTOFF:
        questions.append({"id":"choose_anchor","text":"No confident unique ID found. Select the employee ID field or choose 'Use Composite: First+Last+DOB'.","options":list(df.columns)+["Use Composite: First+Last+DOB"]})
    if recipe["relationship"]["confidence"]<CONF_CUTOFF:
        questions.append({"id":"relationship_field","text":"Which field contains relationship (e.g., Employee/Spouse/Child)?","options":list(df.columns)})
        questions.append({"id":"employee_value","text":"Which value indicates the employee?","options":[]})
    if recipe["structure"]["plans"]=="plan_per_row" and recipe["structure"]["plan_type_field"] is None:
        questions.append({"id":"plan_type_field","text":"Select the field that identifies plan type (e.g., Benefit Type).","options":list(df.columns)})
        questions.append({"id":"plan_detail_fields","text":"Select plan detail fields (multi-select).","options":list(df.columns)})

    return JSONResponse({
        "recipe_draft": recipe,
        "needs_questions": len(questions)>0,
        "questions": questions,
        "carrier_headers": carrier_headers,
        "telemetry": {
            "rows": len(df),
            "anchor_conf": recipe["anchor"]["confidence"],
            "relationship_conf": recipe["relationship"]["confidence"],
            "structure_conf": recipe["structure"]["confidence"]
        }
    })

# ---------------- TRANSFORM (Form fields) ----------------
@app.post("/transform")
async def transform(
    source_file: UploadFile = File(...),
    template_file: UploadFile = File(...),
    recipe_json: str = Form(...),
    auto_approve: bool = Form(False),
):
    recipe = json.loads(recipe_json)

    df = norm_headers(read_any_table(source_file.filename, await source_file.read()))
    tmp_raw = await template_file.read()
    carrier_headers = read_template_headers(template_file, tmp_raw)

    emp_ssn_field = recipe.get("anchor",{}).get("field")
    member_ssn_field = detect_second_ssn(df, emp_ssn_field) if emp_ssn_field else None

    def apply_transform(df_in:pd.DataFrame, recipe:Dict[str,Any])->pd.DataFrame:
        # Anchor / key
        if recipe["anchor"]["type"]=="employee_ssn" and recipe["anchor"]["field"] in df_in.columns:
            key = recipe["anchor"]["field"]
        else:
            parts = recipe["anchor"].get("fallback_composite") or ["FirstName","LastName","DOB"]
            for p in parts:
                if p not in df_in.columns: df_in[p] = ""
            key = "_composite_key"
            df_in[key] = df_in[parts].astype(str).agg("|".join, axis=1)

        base = df_in.copy()
        relcol = recipe["relationship"].get("field")
        if recipe["structure"]["dependents"]=="row_based" and relcol:
            emp_values = {v.lower() for v in recipe["relationship"].get("employee_values",["employee","ee"])}
            base["_is_employee"] = base[relcol].astype(str).str.lower().isin(emp_values)
            base = base[base["_is_employee"]].copy()

        # add Employee SSN / Member SSN columns to base
        if emp_ssn_field and emp_ssn_field in df_in.columns:
            base["Employee SSN"] = df_in[emp_ssn_field]
        if member_ssn_field and member_ssn_field in df_in.columns:
            base["Member SSN"] = df_in[member_ssn_field]
        else:
            if emp_ssn_field and emp_ssn_field in df_in.columns:
                base["Member SSN"] = df_in[emp_ssn_field]

        out = base.copy()

        # Plans: plan_per_row -> wide by plan type, include normalized election/coverage/tier
        if recipe["structure"]["plans"]=="plan_per_row" and recipe["structure"].get("plan_type_field"):
            plan_col = recipe["structure"]["plan_type_field"]
            attrs_raw = recipe["structure"].get("plan_attrs") or []
            attrs = [a for a in attrs_raw if a in df_in.columns]

            # pick ONE best "election-like" column to normalize/display as Election
            election_candidates = [a for a in attrs if any(k in a.lower() for k in ["election","coverage","tier"])]
            election_col = pick_election_column(election_candidates) if election_candidates else None

            wide = base[[key]].drop_duplicates().set_index(key)
            plan_df = df_in[[key, plan_col]+attrs].copy()
            plan_df[plan_col] = plan_df[plan_col].astype(str)

            if election_col:
                plan_df[election_col] = plan_df[election_col].astype(str).map(normalize_election)

            for pval in sorted(plan_df[plan_col].dropna().unique()):
                pblock = plan_df[plan_df[plan_col]==pval].groupby(key).first()

                # rename all attrs under this plan; if we have election_col, map only that to "Election"
                rename_map = {}
                for a in attrs:
                    if election_col and a == election_col:
                        rename_map[a] = f"{pval} - Election"
                    else:
                        rename_map[a] = f"{pval} - {a}"

                pblock = pblock.rename(columns=rename_map)
                # drop the plan_col if present
                pblock = pblock.drop(columns=[c for c in pblock.columns if c == plan_col], errors="ignore")
                wide = wide.join(pblock, how="left")
            out = wide.reset_index()

        return out

    full = apply_transform(df, recipe)
    carrier_df, used_mapping = map_to_headers(full, carrier_headers)

    # preview (mask PII)
    prev = carrier_df.head(20).copy()
    for c in prev.columns:
        prev[c] = prev[c].astype(str).map(lambda v: mask_preview_value(c, v))

    stats = {"rows_in": len(df), "rows_out": len(full), "unique_people": full.shape[0]}
    confs = [
        recipe.get("anchor", {}).get("confidence", 0),
        recipe.get("relationship", {}).get("confidence", 0),
        recipe.get("structure", {}).get("confidence", 0),
    ]
    export_ready = (all(c >= CONF_CUTOFF for c in confs) and auto_approve)

    return JSONResponse({
        "preview": prev.to_dict(orient="records"),
        "stats": stats,
        "export_ready": export_ready,
        "carrier_headers": carrier_headers,
        "used_mapping": used_mapping
    })

# ---------------- EXPORT (Form fields) ----------------
@app.post("/export")
async def export_file(
    source_file: UploadFile = File(...),
    template_file: UploadFile = File(...),
    recipe_json: str = Form(...),
    carrier_name: str = Form("", description="Optional carrier label")
):
    recipe = json.loads(recipe_json)
    df = norm_headers(read_any_table(source_file.filename, await source_file.read()))
    tmp_raw = await template_file.read()
    carrier_headers = read_template_headers(template_file, tmp_raw)

    emp_ssn_field = recipe.get("anchor",{}).get("field")
    member_ssn_field = detect_second_ssn(df, emp_ssn_field) if emp_ssn_field else None

    def apply_transform(df_in:pd.DataFrame, recipe:Dict[str,Any])->pd.DataFrame:
        if recipe["anchor"]["type"]=="employee_ssn" and recipe["anchor"]["field"] in df_in.columns:
            key = recipe["anchor"]["field"]
        else:
            parts = recipe["anchor"].get("fallback_composite") or ["FirstName","LastName","DOB"]
            for p in parts:
                if p not in df_in.columns: df_in[p] = ""
            key = "_composite_key"
            df_in[key] = df_in[parts].astype(str).agg("|".join, axis=1)

        base = df_in.copy()
        relcol = recipe["relationship"].get("field")
        if recipe["structure"]["dependents"]=="row_based" and relcol:
            emp_values = {v.lower() for v in recipe["relationship"].get("employee_values",["employee","ee"])}
            base["_is_employee"] = base[relcol].astype(str).str.lower().isin(emp_values)
            base = base[base["_is_employee"]].copy()

        if emp_ssn_field and emp_ssn_field in df_in.columns:
            base["Employee SSN"] = df_in[emp_ssn_field]
        if member_ssn_field and member_ssn_field in df_in.columns:
            base["Member SSN"] = df_in[member_ssn_field]
        else:
            if emp_ssn_field and emp_ssn_field in df_in.columns:
                base["Member SSN"] = df_in[emp_ssn_field]

        out = base.copy()

        if recipe["structure"]["plans"]=="plan_per_row" and recipe["structure"].get("plan_type_field"):
            plan_col = recipe["structure"]["plan_type_field"]
            attrs_raw = recipe["structure"].get("plan_attrs") or []
            attrs = [a for a in attrs_raw if a in df_in.columns]
            election_candidates = [a for a in attrs if any(k in a.lower() for k in ["election","coverage","tier"])]
            election_col = pick_election_column(election_candidates) if election_candidates else None

            plan_df = df_in[[key, plan_col]+attrs].copy()
            plan_df[plan_col] = plan_df[plan_col].astype(str)
            if election_col:
                plan_df[election_col] = plan_df[election_col].astype(str).map(normalize_election)

            wide = base[[key]].drop_duplicates().set_index(key)
            for pval in sorted(plan_df[plan_col].dropna().unique()):
                pblock = plan_df[plan_df[plan_col]==pval].groupby(key).first()
                rename_map = {}
                for a in attrs:
                    if election_col and a == election_col:
                        rename_map[a] = f"{pval} - Election"
                    else:
                        rename_map[a] = f"{pval} - {a}"
                pblock = pblock.rename(columns=rename_map)
                pblock = pblock.drop(columns=[c for c in pblock.columns if c == plan_col], errors="ignore")
                wide = wide.join(pblock, how="left")
            out = wide.reset_index()

        return out

    full = apply_transform(df, recipe)
    carrier_df, used_mapping = map_to_headers(full, carrier_headers)

    output = io.BytesIO()
    carrier_df.to_excel(output, index=False)
    output.seek(0)

    ks_record_success(carrier_name, carrier_headers, used_mapping, recipe)

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="Transformed_Census.xlsx"'}
    )

# ---------------- FEEDBACK ----------------
@app.post("/feedback")
async def feedback(payload: Dict[str,Any]):
    carrier_name = payload.get("carrier_name") or "unknown"
    headers = payload.get("carrier_headers") or []
    used_mapping = payload.get("used_mapping") or {}
    recipe = payload.get("recipe") or {}
    if not headers or not isinstance(used_mapping, dict):
        return JSONResponse({"error":"missing carrier_headers or used_mapping"}, status_code=400)
    ks_record_success(carrier_name, headers, used_mapping, recipe)
    return {"status":"learned", "mappings_stored": len(used_mapping)}
