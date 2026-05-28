"""
DataMind Agent v2 — Rate Limiting Middleware
Per-user: respects plan tier limits.
Per-IP: protects unauthenticated endpoints.
Uses in-memory store (replace with Redis in production).
"""
from __future__ import annotations
import time, logging
from collections import defaultdict
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)

# In-memory rate limit store {key: [timestamps]}
_rate_store: dict[str, list[float]] = defaultdict(list)

# Rate limit rules per endpoint pattern
RATE_RULES = {
    "/api/v1/analysis/analyse": {"per_minute": 10,  "per_hour": 60},
    "/api/v1/finance/full":     {"per_minute": 5,   "per_hour": 30},
    "/api/v2/auth/login":       {"per_minute": 5,   "per_hour": 20},
    "/api/v2/auth/register":    {"per_minute": 3,   "per_hour": 10},
    "default":                  {"per_minute": 30,  "per_hour": 200},
}

PLAN_MULTIPLIERS = {
    "free":       1.0,
    "starter":    2.0,
    "pro":        5.0,
    "enterprise": 20.0,
}


class RateLimitMiddleware(BaseHTTPMiddleware):

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Skip health check and static
        if path in ["/health", "/", "/docs", "/redoc", "/openapi.json"]:
            return await call_next(request)

        # Get identity
        user_id = getattr(request.state, "user_id", None)
        plan    = getattr(request.state, "plan", "free")
        ip      = request.client.host if request.client else "unknown"
        key     = f"user:{user_id}" if user_id else f"ip:{ip}"

        # Get rule
        rule = RATE_RULES.get(path, RATE_RULES["default"])
        multiplier = PLAN_MULTIPLIERS.get(plan, 1.0)

        per_minute = int(rule["per_minute"] * multiplier)
        per_hour   = int(rule["per_hour"]   * multiplier)

        now = time.time()
        store_key_min  = f"{key}:{path}:min"
        store_key_hour = f"{key}:{path}:hour"

        # Clean old entries
        _rate_store[store_key_min]  = [t for t in _rate_store[store_key_min]  if now - t < 60]
        _rate_store[store_key_hour] = [t for t in _rate_store[store_key_hour] if now - t < 3600]

        # Check limits
        if len(_rate_store[store_key_min]) >= per_minute:
            logger.warning(f"Rate limit (per-minute) hit: {key} on {path}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Too many requests",
                    "detail": f"You have exceeded {per_minute} requests per minute on this endpoint. Please wait before trying again.",
                    "retry_after": 60,
                }
            )

        if len(_rate_store[store_key_hour]) >= per_hour:
            logger.warning(f"Rate limit (per-hour) hit: {key} on {path}")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "Hourly limit reached",
                    "detail": f"You have exceeded {per_hour} requests per hour. Upgrade your plan for higher limits.",
                    "retry_after": 3600,
                }
            )

        # Record this request
        _rate_store[store_key_min].append(now)
        _rate_store[store_key_hour].append(now)

        # Add rate limit headers
        response = await call_next(request)
        response.headers["X-RateLimit-Limit-Minute"] = str(per_minute)
        response.headers["X-RateLimit-Remaining-Minute"] = str(per_minute - len(_rate_store[store_key_min]))
        response.headers["X-RateLimit-Limit-Hour"] = str(per_hour)
        return response
