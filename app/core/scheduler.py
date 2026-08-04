"""Daily job scheduling via APScheduler (in-process).

The CCTV scrape runs every day at 02:00 WIB (Asia/Jakarta). The scrape
itself is blocking (requests), so it is executed in a worker thread via
asyncio.to_thread to keep the event loop responsive.
"""

import asyncio

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from app.core.config import SCHEDULER_ENABLED, SCRAPE_TIMEZONE
from app.services.scraping import run_scrape

scheduler = AsyncIOScheduler(timezone=SCRAPE_TIMEZONE)

SCRAPE_JOB_ID = "daily_cctv_scrape"


async def _daily_cctv_scrape() -> None:
    summary = await asyncio.to_thread(run_scrape)
    print(f"[scheduler] CCTV scrape done: {summary}")


def start_scheduler() -> None:
    """Register the daily job and start the scheduler (no-op if disabled or already running)."""
    if not SCHEDULER_ENABLED:
        print("[scheduler] disabled via SCHEDULER_ENABLED=false")
        return

    if scheduler.running:
        return

    scheduler.add_job(
        _daily_cctv_scrape,
        CronTrigger(hour=2, minute=0),
        id=SCRAPE_JOB_ID,
        replace_existing=True,
        max_instances=1,
    )
    scheduler.start()
    print(f"[scheduler] started; next run: {scheduler.get_job(SCRAPE_JOB_ID).next_run_time}")


def shutdown_scheduler() -> None:
    """Stop the scheduler if it is running."""
    if scheduler.running:
        scheduler.shutdown(wait=False)
        print("[scheduler] stopped")
