"""HTTP API (M6b): FastAPI over the M6a service/job/gate seams.

The API layer contains zero pipeline logic — DTO validation, path containment, and
typed-exception -> status mapping only. Heavy stages run as durable jobs on the M6a
single-flight runner; fast reads are synchronous endpoints. Deployment is exactly ONE
uvicorn worker process (`workers=1`): the GPU manager, the runner queue, and the
heavy-work gate are all process-local. Run: `uvicorn seiyuu.api.main:app`.
"""
