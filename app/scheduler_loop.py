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
    while _running:
        try:
            from app.services.schedule_service import schedule_service
            from app.database import is_configured
            if is_configured():
                res = await schedule_service.run_due()
                if res.get("ran"):
                    logger.info("Scheduler ran %s due report(s)", res["ran"])
        except Exception as e:
            logger.error("Scheduler tick failed: %s", e)
        await asyncio.sleep(TICK_SECONDS)


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
