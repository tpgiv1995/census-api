from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import io

app = FastAPI()

# Allow frontend to communicate
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/preview")
async def preview(source_file: UploadFile = File(...), template_file: UploadFile = File(...)):
    try:
        source_bytes = await source_file.read()
        template_bytes = await template_file.read()
        
        source_df = pd.read_excel(io.BytesIO(source_bytes))
        template_df = pd.read_excel(io.BytesIO(template_bytes))

        # Check for expected template format
        if 'Column Header' not in template_df.columns:
            return JSONResponse(status_code=400, content={"error": "Template file must have a 'Column Header' column."})

        expected_columns = template_df['Column Header'].dropna().tolist()

        filtered_df = pd.DataFrame()
        for col in expected_columns:
            filtered_df[col] = source_df[col] if col in source_df.columns else ""

        preview = filtered_df.head(10).to_dict(orient="records")
        return {"preview": preview, "columns": filtered_df.columns.tolist()}

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/transform")
async def transform(source_file: UploadFile = File(...), template_file: UploadFile = File(...)):
    try:
        source_bytes = await source_file.read()
        template_bytes = await template_file.read()
        
        source_df = pd.read_excel(io.BytesIO(source_bytes))
        template_df = pd.read_excel(io.BytesIO(template_bytes))

        if 'Column Header' not in template_df.columns:
            return JSONResponse(status_code=400, content={"error": "Template file must have a 'Column Header' column."})

        expected_columns = template_df['Column Header'].dropna().tolist()

        final_df = pd.DataFrame()
        for col in expected_columns:
            final_df[col] = source_df[col] if col in source_df.columns else ""

        # Save to file
        output = io.BytesIO()
        final_df.to_excel(output, index=False)
        output.seek(0)

        return JSONResponse(content={"message": "Transform complete. (Download feature coming next)."})

    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})
