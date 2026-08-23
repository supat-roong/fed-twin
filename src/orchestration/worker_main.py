"""
Temporal worker process: registers the workflows and activities and polls.

Runs as a Deployment in the consumer namespace. Its ServiceAccount needs
create/delete on jobs and get/list/watch on pods plus pods/log, because
`launch_and_watch_pod` does exactly those things.
"""

from __future__ import annotations

import asyncio
import logging
import os
import signal
from datetime import timedelta

from temporalio.client import Client
from temporalio.worker import Worker

from src.orchestration.activities import cleanup_worker_job, launch_and_watch_pod
from src.orchestration.workflows import TASK_QUEUE, TrainRoundWorkflow, WorkerWorkflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# Kept under the Deployment's default 30s terminationGracePeriodSeconds so the
# process still has time to exit after shutdown() returns, rather than being
# SIGKILLed mid-cleanup.
GRACEFUL_SHUTDOWN_TIMEOUT = timedelta(seconds=20)


async def main() -> None:
    address = os.environ.get("TEMPORAL_ADDRESS", "temporal-frontend:7233")
    namespace = os.environ.get("TEMPORAL_NAMESPACE", "default")
    log.info(f"connecting to Temporal at {address} (namespace={namespace})")

    client = await Client.connect(address, namespace=namespace)
    worker = Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[TrainRoundWorkflow, WorkerWorkflow],
        activities=[launch_and_watch_pod, cleanup_worker_job],
        graceful_shutdown_timeout=GRACEFUL_SHUTDOWN_TIMEOUT,
    )

    # Kubernetes sends SIGTERM on pod termination (rollout, scale-down, node
    # drain). Without catching it, in-flight activity tasks are killed
    # abruptly instead of getting a chance to finish their current step.
    # `async with worker` starts polling on entry and calls `worker.shutdown()`
    # on exit, which waits up to GRACEFUL_SHUTDOWN_TIMEOUT for in-flight
    # activities before cancelling them.
    shutdown_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, shutdown_event.set)

    log.info(f"worker started on task queue '{TASK_QUEUE}'")
    async with worker:
        await shutdown_event.wait()
    log.info("worker shut down gracefully")


if __name__ == "__main__":
    asyncio.run(main())
