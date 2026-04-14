from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

class TenantMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        tenant_id = request.headers.get("X-Tenant-ID")

        if not tenant_id:
            return {"error": "Tenant ID required"}

        request.state.tenant_id = tenant_id

        response = await call_next(request)
        return response
