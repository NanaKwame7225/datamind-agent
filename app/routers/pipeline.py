import uuid, time, logging
from fastapi import APIRouter
from app.models.schemas import PipelineRunRequest, PipelineRunResponse

router = APIRouter()
logger = logging.getLogger(__name__)
_runs: dict = {}

@router.post("/run", response_model=PipelineRunResponse)
async def run_pipeline(req: PipelineRunRequest):
    run_id = str(uuid.uuid4())[:12]
    t0 = time.perf_counter()
    outputs = {}
    for i, step in enumerate(req.steps):
        tool = step.get("tool", "unknown")
        outputs[f"step_{i+1}_{tool}"] = {"status": "completed"}
        logger.info(f"Step {i+1}: {tool}")
    run = {"run_id": run_id, "status": "completed",
           "steps_completed": len(req.steps), "steps_total": len(req.steps),
           "duration_ms": round((time.perf_counter()-t0)*1000, 1), "outputs": outputs}
    _runs[run_id] = run
    return PipelineRunResponse(**run)

@router.get("/status/{run_id}")
async def pipeline_status(run_id: str):
    return _runs.get(run_id, {"error": "not found"})
