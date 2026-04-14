from fastapi import APIRouter
from core.enterprise_connectors import (
    get_quickbooks_data,
    get_sap_data
)

router = APIRouter()

@router.get("/quickbooks")
def qb(token: str):
    return get_quickbooks_data(token)


@router.get("/sap")
def sap(base_url: str, token: str):
    return get_sap_data(base_url, token)
