from fastapi import FastAPI

from pipeline.pipeline import StudentProfilePipeline

app = FastAPI(title="Student Profile Pipeline")
pipeline = StudentProfilePipeline()


@app.post("/import")
async def import_students():
    return {"message": "Import endpoint ready", "status": "ok"}
