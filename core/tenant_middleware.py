from starlette.middleware.base import BaseHTTPMiddleware

class TenantMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        tenant = request.headers.get("X-Tenant-ID", "default")
        request.state.tenant_id = tenant
        return await call_next(request)
