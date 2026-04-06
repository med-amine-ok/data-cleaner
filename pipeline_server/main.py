from __future__ import annotations

from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from loguru import logger

from pipeline.pipeline import StudentProfilePipeline

ALLOWED_EXTENSIONS: frozenset[str] = frozenset({".csv", ".json", ".xlsx", ".ods"})
ZIP_MEDIA_TYPE: str = "application/zip"

app = FastAPI(title="Stateless Student Cleaning Pipeline")
pipeline = StudentProfilePipeline()
frontend_path = Path(__file__).resolve().parent / "frontend" / "index.html"


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    """
    Serve the browser-based upload interface.

    Returns:
        An HTML page that lets the user upload a file and download the ZIP output.
    """
    if frontend_path.exists():
        return HTMLResponse(frontend_path.read_text(encoding="utf-8"))

    return HTMLResponse("<h1>Frontend not found</h1>", status_code=404)


def _validate_upload_filename(filename: str) -> str:
    """
    Validate the uploaded file extension.

    Args:
        filename: Original uploaded filename.

    Returns:
        The validated lowercase extension.

    Raises:
        HTTPException: If the filename is missing or the extension is unsupported.
    """
    if not filename:
        raise HTTPException(status_code=400, detail="Upload file must have a filename.")

    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed types are: {sorted(ALLOWED_EXTENSIONS)}",
        )

    return suffix


async def _save_upload_to_path(upload_file: UploadFile, destination: Path) -> Path:
    """
    Save an uploaded file to disk inside a temporary folder.

    Args:
        upload_file: The incoming FastAPI file object.
        destination: Path where the file should be written.

    Returns:
        The same destination path after writing the file.
    """
    content = await upload_file.read()
    destination.write_bytes(content)
    return destination


def _to_dataframe(value: Any) -> pd.DataFrame:
    """
    Convert pipeline output into a Pandas DataFrame.

    Args:
        value: Pipeline output value. Can already be a DataFrame, a list of dicts,
            a dict, or another table-like structure.

    Returns:
        A DataFrame representation of the value.
    """
    if isinstance(value, pd.DataFrame):
        return value.copy()

    if value is None:
        return pd.DataFrame()

    if isinstance(value, list):
        return pd.DataFrame(value)

    if isinstance(value, Mapping):
        return pd.DataFrame([dict(value)])

    return pd.DataFrame(value)


def _dataframe_to_xlsx_bytes(dataframe: pd.DataFrame) -> bytes:
    """
    Convert a DataFrame into Excel bytes.

    Args:
        dataframe: DataFrame to export.

    Returns:
        Binary XLSX content.
    """
    output = BytesIO()
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        dataframe.to_excel(writer, index=False)
    return output.getvalue()


def _build_zip_response(result: Mapping[str, Any]) -> StreamingResponse:
    """
    Build a ZIP streaming response from pipeline outputs.

    Args:
        result: Mapping returned by StudentProfilePipeline.run().

    Returns:
        A StreamingResponse that streams a ZIP archive to the client.
    """
    buffer = BytesIO()

    clean_records = _to_dataframe(result.get("clean_records"))
    quarantine_records = _to_dataframe(result.get("quarantine_records"))
    duplicates_records = _to_dataframe(result.get("duplicates_records"))
    summary_text = str(result.get("pipeline_summary", ""))

    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("clean_records.xlsx", _dataframe_to_xlsx_bytes(clean_records))
        archive.writestr("quarantine.xlsx", _dataframe_to_xlsx_bytes(quarantine_records))
        archive.writestr("duplicates.xlsx", _dataframe_to_xlsx_bytes(duplicates_records))
        archive.writestr("pipeline_summary.txt", summary_text)

    buffer.seek(0)
    headers = {
        "Content-Disposition": 'attachment; filename="pipeline_outputs.zip"'
    }
    return StreamingResponse(buffer, media_type=ZIP_MEDIA_TYPE, headers=headers)


@app.post("/upload", response_class=StreamingResponse)
async def upload(file: UploadFile = File(...)) -> StreamingResponse:
    """
    Accept a messy spreadsheet upload, run the pipeline, and return a ZIP file.

    Args:
        file: Uploaded spreadsheet file.

    Returns:
        StreamingResponse containing a ZIP archive with the cleaned outputs.

    Raises:
        HTTPException: If the upload is invalid or the pipeline returns an unexpected result.
    """
    suffix = _validate_upload_filename(file.filename or "")

    try:
        with TemporaryDirectory(prefix="student-import-") as temp_dir:
            input_path = Path(temp_dir) / f"input{suffix}"
            await _save_upload_to_path(file, input_path)

            logger.info("Processing upload {}", file.filename)

            result = pipeline.run(str(input_path))
            if not isinstance(result, Mapping):
                raise HTTPException(
                    status_code=500,
                    detail="Pipeline returned an invalid result structure.",
                )

            return _build_zip_response(result)
    finally:
        await file.close()