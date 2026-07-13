"""
DataMind Agent — Background Scheduler Loop

A single in-process asyncio task that wakes once a minute and asks the schedule
service to run anything due. No Celery, no Redis, no extra Railway service.

It's deliberately defensive: any error in one tick is logged and swallowed so
the loop never dies. If the database isn't configured, it simply idles.
"""
from __future__ import annotations
import asyncio, logging

logger = logging.getLogger(__name__)

_task = None
_running = False
TICK_SECONDS = 60


async def _loop():
    global _running
    _running = True
    logger.info("Scheduler loop started (ticks every %ss)", TICK_SECONDS)
    # Small initial delay so startup finishes first
    await asyncio.sleep(10)
    ticks = 0
    while _running:
        try:
            from app.services.schedule_service import schedule_service
            from app.database import is_configured
            if is_configured():
                res = await schedule_service.run_due()
                if res.get("ran"):
                    logger.info("Scheduler ran %s due report(s)", res["ran"])
            # Keep-warm: every ~4 minutes, self-ping the public URL so Railway
            # sees inbound traffic and doesn't spin the container down. Only
            # works if PUBLIC_URL / RAILWAY_PUBLIC_DOMAIN is set.
            ticks += 1
            if ticks % 4 == 0:
                await _self_ping()
        except Exception as e:
            logger.error("Scheduler tick failed: %s", e)
        await asyncio.sleep(TICK_SECONDS)


async def _self_ping():
    import os
    url = os.environ.get("PUBLIC_URL")
    if not url:
        dom = os.environ.get("RAILWAY_PUBLIC_DOMAIN")
        if dom:
            url = f"https://{dom}"
    if not url:
        return
    try:
        import urllib.request
        req = urllib.request.Request(url.rstrip("/") + "/health",
                                     headers={"User-Agent": "datamind-keepwarm"})
        # Run the blocking call in a thread so we don't stall the loop
        await asyncio.get_event_loop().run_in_executor(
            None, lambda: urllib.request.urlopen(req, timeout=15))
    except Exception:
        pass  # keep-warm is best-effort; never let it break the loop


def start():
    global _task
    if _task is not None:
        return
    try:
        loop = asyncio.get_event_loop()
        _task = loop.create_task(_loop())
        logger.info("Scheduler task scheduled")
    except Exception as e:
        logger.error("Could not start scheduler loop: %s", e)


async def stop():
    global _running, _task
    _running = False
    if _task:
        _task.cancel()
        _task = None
