from fastapi import FastAPI, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Union
import uuid
import asyncio
import logging
import os
import json
from pathlib import Path

# Import our modules
from job_manager import JobManager
from models import JobStatus, AttributionJobRequest, AttributionJobResponse, AttributionJobResultResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the FastAPI app
app = FastAPI(
    title="Inseq Feature Attribution API",
    description="API for running feature attribution on language models using Inseq",
    version="1.0.0"
)

# Initialize the job manager with a 30-minute timeout
job_manager = JobManager(timeout=1800)


@app.post("/attribution/jobs", response_model=AttributionJobResponse)
async def create_attribution_job(
        request: AttributionJobRequest,
        background_tasks: BackgroundTasks
):
    """
    Create a new attribution job.

    This endpoint accepts an attribution request and queues it for processing.
    It returns a job ID that can be used to check the status of the job and
    retrieve the results when completed.
    """
    # Create a new job
    job_id = job_manager.create_job(request)

    # Start the job in the background
    background_tasks.add_task(job_manager.run_job, job_id)

    return AttributionJobResponse(job_id=job_id, status=JobStatus.PENDING)


@app.get("/attribution/jobs/{job_id}", response_model=AttributionJobResultResponse)
async def get_attribution_job(job_id: str):
    """
    Get the status of an attribution job.

    This endpoint returns the status of the job and the output file path if
    the job has completed successfully.
    """
    try:
        job = job_manager.get_job(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    return AttributionJobResultResponse(
        job_id=job_id,
        status=job["status"],
        output_file=job["output_file"],
        error=job["error"]
    )


@app.get("/attribution/jobs", response_model=List[AttributionJobResponse])
async def list_attribution_jobs():
    """
    List all attribution jobs.

    This endpoint returns a list of all attribution jobs with their status.
    """
    jobs = job_manager.list_jobs()
    return [
        AttributionJobResponse(job_id=job["job_id"], status=job["status"])
        for job in jobs
    ]


@app.get("/attribution/jobs/{job_id}/results")
async def get_attribution_results(job_id: str):
    """
    Get the results of an attribution job.

    This endpoint returns the attribution results if the job has completed
    successfully. The results are returned as a JSON object.
    """
    try:
        job = job_manager.get_job(job_id)
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")

    if job["status"] != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job {job_id} is not completed. Current status: {job['status']}"
        )

    if not job["output_file"] or not os.path.exists(job["output_file"]):
        raise HTTPException(status_code=404, detail="Attribution results not found")

    # Read the attribution results JSON file
    try:
        with open(job["output_file"], "r") as f:
            results = json.load(f)

        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading attribution results: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)