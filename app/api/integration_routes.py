from fastapi import APIRouter

router = APIRouter()

@router.get("/quickbooks")
def quickbooks():
    return {"status": "connected"}

@router.get("/sap")
def sap():
    return {"status": "connected"}
