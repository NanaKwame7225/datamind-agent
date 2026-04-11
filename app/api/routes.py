from fastapi import APIRouter, UploadFile
import pandas as pd
from app.services.orchestrator import DataIntelligenceAgent

router = APIRouter()
agent = DataIntelligenceAgent()


@router.post("/analyze")
async def analyze(file: UploadFile):

    df = pd.read_csv(file.file)

    result = agent.run(df)

    return result
