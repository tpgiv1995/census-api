from fastapi import FastAPI, UploadFile, File, Body, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import pandas as pd
import io, re

app = FastAPI()

# Pydantic data models for the transformation recipe
class Anchor(BaseModel):
    type: str                       # "employee_ssn" or "composite"
    field: str = None               # column name if using a direct ID field (like SSN)
    fallback_composite: list = None # list of fields for composite key if applicable
    confidence: float

class RelationshipCfg(BaseModel):
    field: str = None               # column name that indicates relationship (if any)
    employee_values: list = []      # values in that field that denote an Employee (e.g., "EE")
    confidence: float

class StructureCfg(BaseModel):
    dependents: str                 # "row_based" or "column_based"
    plans: str                      # "plan_per_row" or "plan_per_column"
    plan_type_field: str = None     # column name for plan type (if plans per row)
    plan_attrs: list = []           # other columns that vary per plan (for pivoting)
    confidence: float

class MappingCfg(BaseModel):
    field_map: dict = {}            # (optional) mapping of source fields to target (not heavily used here)

class ExportCfg(BaseModel):
    layout: str                     # e.g., "Row-Based" (for documentation purposes)
    include_dependents: bool
    columns_order: list = []        # final columns order if specified (we derive from template)

class Recipe(BaseModel):
    anchor: Anchor
    relationship: RelationshipCfg
    structure: StructureCfg
    mapping: MappingCfg
    export: ExportCfg

# Utility functions to handle data reading and normalization
def read_any_table(filename: str, content: bytes) -> pd.DataFrame:
    """Read CSV or Excel content into a DataFrame."""
    # Determine format by file extension
    fn = filename.lower()
    if fn.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(content))
    else:
        # Try CSV (comma-separated). If that fails, try tab-separated.
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception:
            df = pd.read_csv(io.BytesIO(content), sep="\t")
    return df

def normalize_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Strip and standardize column headers (remove spaces/punctuation)."""
    df = df.copy()
    df.columns = [re.sub(r"\s+", "", str(col)).strip() for col in df.columns]
    return df

def read_template_headers(file: UploadFile, content: bytes):
    """Extract column headers from the carrier template file."""
    fname = file.filename.lower()
    if fname.endswith((".xlsx", ".xls")):
        # Read only header (nrows=0 yields columns)
        temp_df = pd.read_excel(io.BytesIO(content), nrows=0)
    else:
        temp_df = pd.read_csv(io.BytesIO(content), nrows=0)
    headers = list(temp_df.columns)
    return headers

def map_to_carrier_headers(df: pd.DataFrame, carrier_headers: list) -> pd.DataFrame:
    """Map DataFrame columns to exactly match the carrier template headers (order and names)."""
    out_df = pd.DataFrame()
    # Normalize keys for matching (alphanumeric lowercase)
    df_cols_norm = {re.sub(r'[^A-Za-z0-9]', '', col).lower(): col for col in df.columns}
    for header in carrier_headers:
        norm = re.sub(r'[^A-Za-z0-9]', '', header).lower()
        if norm in df_cols_norm:
            # If a source column matches (by normalized name), use it
            out_df[header] = df[df_cols_norm[norm]]
        else:
            # No matching column in output, create an empty column
            out_df[header] = ""
    return out_df

# Heuristic detection functions (leveraging domain knowledge from docs)
def detect_employee_ssn(df: pd.DataFrame):
    """Detect a unique identifier column (SSN or similar)."""
    ssn_col = None
    confidence = 0.0
    for col in df.columns:
        # Check if column name suggests SSN or ID
        if re.search(r"ssn|social", col, re.IGNORECASE):
            ssn_col = col
            # If values look like SSNs (9-digit), boost confidence
            sample = df[col].astype(str).str.replace(r"\D", "", regex=True)
            if sample.str.match(r"^\d{9}$").all():
                confidence = 0.95
            else:
                confidence = 0.9
            break
        if re.search(r"emp.?id|employee.?id", col, re.IGNORECASE):
            # Potential employee ID field
            if df[col].nunique() == len(df):
                ssn_col = col
                confidence = 0.9
                break
    # If no explicit ID column found, no unique identifier detected confidently
    return ssn_col, confidence

def detect_relationship(df: pd.DataFrame):
    """Detect relationship column and the token indicating 'Employee'."""
    rel_col = None
    employee_tokens = []
    confidence = 0.0
    common_emp_terms = {"employee", "emp", "ee"}  # tokens that imply an employee
    for col in df.columns:
        # Check column name for relationship hints
        if re.search(r"relat|relationship|relation", col, re.IGNORECASE):
            rel_col = col
        # Check data values for clues (e.g., contains 'Spouse', 'Child', 'EE')
        values = {str(v).strip().lower() for v in df[col].dropna().unique()}
        if any(val in ("spouse", "child", "dependent") or val in common_emp_terms for val in values):
            rel_col = col
        if rel_col:
            # Identify what denotes an employee in this column
            vals = {str(v).strip().lower() for v in df[rel_col].dropna().unique()}
            for val in vals:
                if val in common_emp_terms or val.startswith("emp"):
                    employee_tokens.append(val)
            if not employee_tokens:
                # If no obvious token found, default to a generic value if present
                if "employee" in vals: employee_tokens.append("employee")
                if "ee" in vals: employee_tokens.append("ee")
            confidence = 0.9
            break
    return rel_col, employee_tokens, confidence

def detect_dependents_structure(df: pd.DataFrame):
    """Determine if dependents are listed in separate rows or extra columns."""
    # Clue for column-based dependents: columns like Dep1, Dependent Name, etc.
    col_based = any(re.search(r"dep(?:endent)?\s*1", col, re.IGNORECASE) for col in df.columns)
    if col_based:
        return "column_based", 0.9
    # Clue for row-based: presence of a relationship column (with spouse/child entries)
    rel_col, _, _ = detect_relationship(df)
    if rel_col:
        return "row_based", 0.9
    # Default (if uncertain, assume row-based as it is more common, but low confidence)
    return "row_based", 0.5

def detect_plan_per_row(df: pd.DataFrame, id_col: str = None):
    """Detect if multiple benefit plans are listed in separate rows per person or in columns."""
    # If we have an identifier (like SSN or EmpID), check for multiple entries per person
    if id_col and id_col in df.columns:
        counts = df[id_col].value_counts()
        multi_ids = counts[counts > 1].index
        if len(multi_ids) > 0:
            # We have at least one person with multiple rows (suggests plan-per-row layout)
            # Guess the plan type column: likely one with a small set of repeating values (e.g., Benefit Type)
            candidate_cols = []
            for col in df.columns:
                if col == id_col: 
                    continue
                # Check if this column varies for the same person
                for person in multi_ids:
                    vals = set(df[df[id_col] == person][col].astype(str))
                    if len(vals) > 1 and len(vals) < 10:
                        candidate_cols.append(col)
                        break
            plan_type_col = None
            if candidate_cols:
                # Choose the column with the fewest unique values as the likely plan type
                plan_type_col = min(candidate_cols, key=lambda c: df[c].nunique())
            # Plan attributes: other columns that vary with plan (exclude the ID and plan_type_col)
            plan_attrs = [c for c in candidate_cols if c != plan_type_col]
            return "plan_per_row", plan_type_col, plan_attrs, 0.9 if plan_type_col else 0.7
    # If no multiple entries detected or no ID, assume one-row-per-person structure.
    # Check if multiple plan fields exist on one row (e.g., "Medical Plan", "Dental Plan")
    plan_cols = [col for col in df.columns if re.search(r"medical|dental|vision|plan", col, re.IGNORECASE)]
    if len(plan_cols) > 1:
        # Likely multiple plan columns -> plan-per-column layout
        return "plan_per_column", None, None, 0.9
    # Default to plan-per-column if nothing indicates otherwise
    return "plan_per_column", None, None, 0.5

# FastAPI Endpoints

@app.post("/profile")
async def profile(source_file: UploadFile = File(...), template_file: UploadFile = File(...)):
    """Profiling endpoint: analyze source & template and propose a transformation recipe."""
    # Read and normalize source data
    src_bytes = await source_file.read()
    df_raw = read_any_table(source_file.filename, src_bytes)
    df = normalize_headers(df_raw)
    # Read carrier template headers
    temp_bytes = await template_file.read()
    carrier_headers = read_template_headers(template_file, temp_bytes)
    # Infer key features
    emp_col, emp_conf = detect_employee_ssn(df)
    if emp_col:
        anchor = Anchor(type="employee_ssn", field=emp_col, fallback_composite=None, confidence=emp_conf)
    else:
        # No unique ID found – default to composite of First+Last+DOB
        anchor = Anchor(type="composite", field=None, fallback_composite=["FirstName", "LastName", "DOB"], confidence=0.6)
    rel_col, emp_tokens, rel_conf = detect_relationship(df)
    dep_struct, dep_conf = detect_dependents_structure(df)
    plan_mode, plan_type, plan_attrs, plan_conf = detect_plan_per_row(df, emp_col if emp_col else None)
    structure = StructureCfg(dependents=("row_based" if dep_struct=="row_based" else "column_based"),
                              plans=plan_mode, plan_type_field=plan_type, plan_attrs=(plan_attrs or []),
                              confidence=min(dep_conf, plan_conf) if plan_conf else dep_conf)
    relationship = RelationshipCfg(field=rel_col, employee_values=emp_tokens, confidence=rel_conf)
    # Prepare recipe draft
    recipe = Recipe(anchor=anchor,
                    relationship=relationship,
                    structure=structure,
                    mapping=MappingCfg(), 
                    export=ExportCfg(layout="Row-Based", include_dependents=True, columns_order=[]))
    # Formulate clarification questions if needed
    questions = []
    # If using composite because no strong ID, let user choose an anchor field
    if recipe.anchor.type == "composite" and recipe.anchor.confidence < 0.8:
        questions.append({
            "id": "choose_anchor",
            "text": "Unique ID not found confidently. Select the ID field or choose 'Use Composite: First+Last+DOB'.",
            "options": list(df.columns) + ["Use Composite: First+Last+DOB"]
        })
    # If relationship field uncertain
    if recipe.relationship.field is None or recipe.relationship.confidence < 0.8:
        questions.append({
            "id": "relationship_field",
            "text": "Which field contains the relationship (e.g., Employee/Spouse/Child)?",
            "options": list(df.columns)
        })
        questions.append({
            "id": "employee_value",
            "text": "Which value in that field denotes an Employee (as opposed to a dependent)?",
            "options": []  # to be filled after user picks relationship_field, if using interactive UI
        })
    # If plan structure is plan_per_row but we couldn't identify the plan type field
    if recipe.structure.plans == "plan_per_row" and recipe.structure.plan_type_field is None:
        questions.append({
            "id": "plan_type_field",
            "text": "Select the field that indicates the type of benefit/plan (e.g., Benefit Type).",
            "options": list(df.columns)
        })
        questions.append({
            "id": "plan_detail_fields",
            "text": "Select all fields that contain plan-specific details (you can choose multiple).",
            "options": list(df.columns)
        })
    return {
        "recipe_draft": recipe.dict(),
        "needs_questions": len(questions) > 0,
        "questions": questions,
        "carrier_headers": carrier_headers
    }

@app.post("/transform")
async def transform(recipe: Recipe = Body(...),
                   source_file: UploadFile = File(...),
                   template_file: UploadFile = File(...),
                   auto_approve: bool = Body(True)):
    """Transformation endpoint: apply the recipe to the source data and return a preview."""
    # Read source and template (we re-read files to ensure fresh data)
    df = normalize_headers(read_any_table(source_file.filename, await source_file.read()))
    carrier_headers = read_template_headers(template_file, await template_file.read())
    # Build unique key (anchor)
    if recipe.anchor.type == "employee_ssn" and recipe.anchor.field and recipe.anchor.field in df.columns:
        key = recipe.anchor.field
    else:
        # Construct composite key
        parts = recipe.anchor.fallback_composite or ["FirstName", "LastName", "DOB"]
        # Ensure all parts exist
        for p in parts:
            if p not in df.columns:
                raise ValueError(f"Missing composite part: {p}")
        key = "_composite_key"
        df[key] = df[parts].astype(str).agg("|".join, axis=1)
    # Handle dependents structure
    base_df = df.copy()
    if recipe.structure.dependents == "row_based" and recipe.relationship.field:
        # Filter to employee rows only (if relationship field is present)
        emp_vals_lower = {str(v).lower() for v in recipe.relationship.employee_values}
        base_df["_is_employee"] = df[recipe.relationship.field].astype(str).str.lower().isin(emp_vals_lower)
        base_df = base_df[base_df["_is_employee"]].copy()
    # Handle multiple plans structure
    out_df = base_df.copy()
    if recipe.structure.plans == "plan_per_row" and recipe.structure.plan_type_field:
        plan_col = recipe.structure.plan_type_field
        attrs = [a for a in (recipe.structure.plan_attrs or []) if a in df.columns]
        # Pivot plan rows to columns: one row per person (key)
        wide = base_df[[key]].drop_duplicates().set_index(key)
        plan_df = df[[key, plan_col] + attrs].copy()
        plan_df[plan_col] = plan_df[plan_col].astype(str)
        for plan in sorted(plan_df[plan_col].dropna().unique()):
            # For each plan type, take the first row (or aggregate) as representative
            plan_block = plan_df[plan_df[plan_col] == plan].groupby(key).first()
            # Rename columns to "<Plan> - <Attr>"
            plan_block = plan_block.rename(columns={a: f"{plan} - {a}" for a in attrs})
            # Join into wide table
            wide = wide.join(plan_block.drop(columns=[plan_col], errors="ignore"), how="left")
        out_df = wide.reset_index()
    # Align output columns to carrier template for consistency in preview
    carrier_df = map_to_carrier_headers(out_df, carrier_headers)
    preview_records = carrier_df.head(20).to_dict(orient="records")
    # Calculate some stats
    stats = {
        "rows_in": len(df),
        "rows_out": len(out_df),
        "unique_people": out_df[key].nunique() if key in out_df.columns else None
    }
    # Determine if we can auto-export (all confidences high and auto_approve requested)
    confidences = []
    try:
        confidences = [recipe.anchor.confidence, recipe.relationship.confidence, recipe.structure.confidence]
    except:
        pass
    export_ready = (all(c is not None and c >= 0.8 for c in confidences) and auto_approve)
    return {
        "preview": preview_records,
        "stats": stats,
        "export_ready": export_ready,
        "carrier_headers": carrier_headers  # (included for reference; not directly used by n8n in this step)
    }

@app.post("/export")
async def export(source_file: UploadFile = File(...),
                 template_file: UploadFile = File(...),
                 recipe_json: str = Form(...)):
    """Export endpoint: generate the final transformed file (Excel) given the confirmed recipe."""
    # Parse recipe (as JSON string because it's sent as form-data)
    recipe = Recipe.parse_raw(recipe_json)
    # Load source data and template headers
    df = normalize_headers(read_any_table(source_file.filename, await source_file.read()))
    carrier_headers = read_template_headers(template_file, await template_file.read())
    # Reapply transformation (same logic as in /transform, using final recipe)
    if recipe.anchor.type == "employee_ssn" and recipe.anchor.field and recipe.anchor.field in df.columns:
        key = recipe.anchor.field
    else:
        parts = recipe.anchor.fallback_composite or ["FirstName", "LastName", "DOB"]
        key = "_composite_key"
        df[key] = df[parts].astype(str).agg("|".join, axis=1)
    base_df = df.copy()
    if recipe.structure.dependents == "row_based" and recipe.relationship.field:
        emp_vals_lower = {str(v).lower() for v in recipe.relationship.employee_values}
        base_df["_is_employee"] = df[recipe.relationship.field].astype(str).str.lower().isin(emp_vals_lower)
        base_df = base_df[ base_df["_is_employee"] ].copy()
    out_df = base_df.copy()
    if recipe.structure.plans == "plan_per_row" and recipe.structure.plan_type_field:
        plan_col = recipe.structure.plan_type_field
        attrs = [a for a in (recipe.structure.plan_attrs or []) if a in df.columns]
        wide = base_df[[key]].drop_duplicates().set_index(key)
        plan_df = df[[key, plan_col] + attrs].copy()
        plan_df[plan_col] = plan_df[plan_col].astype(str)
        for plan in sorted(plan_df[plan_col].dropna().unique()):
            plan_block = plan_df[ plan_df[plan_col] == plan ].groupby(key).first()
            plan_block = plan_block.rename(columns={a: f"{plan} - {a}" for a in attrs})
            wide = wide.join(plan_block.drop(columns=[plan_col], errors="ignore"), how="left")
        out_df = wide.reset_index()
    carrier_df = map_to_carrier_headers(out_df, carrier_headers)
    # Stream the result as an Excel file (.xlsx)
    output_stream = io.BytesIO()
    with pd.ExcelWriter(output_stream, engine='openpyxl') as writer:
        carrier_df.to_excel(writer, index=False, sheet_name="Transformed")
    output_stream.seek(0)
    return StreamingResponse(output_stream, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                             headers={"Content-Disposition": "attachment; filename=transformed.xlsx"})
