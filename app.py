# app.py

from fastapi import FastAPI # type: ignore
from fastapi.responses import StreamingResponse # type: ignore
import io
from datetime import datetime
import os

from fx_generator import FXConfig, generate_fx_details

app = FastAPI(title="FX Details Generator API")

@app.post("/fx-details/xlsx", response_class=StreamingResponse)
def generate_fx_details_xlsx(config: FXConfig):
    df = generate_fx_details(config)

    # ---- 1. Generate timestamped filename ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"fx_details_{timestamp}.xlsx"

    # ---- 2. Save to disk in the current working directory ----
    df.to_excel(filename, index=False)

    # ---- 3. Also stream file back to user (Swagger) ----
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    buf.seek(0)

    return StreamingResponse(
        buf,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
