import uuid

import httpx
from loguru import logger

from app.agent.state import AgentState
from app.config import settings
from app.schemas.task import STaskSchema

TODOIST_API_BASE = "https://api.todoist.com/api/v1"


async def create_tasks(tasks: list[STaskSchema], state: AgentState) -> list[str]:
    """
    Создает задачи в Todoist

    Returns:
        list[str]: список созданных task_id
    """
    if not state["confirmed"]:
        raise PermissionError("create_tasks вызван без confirmed=True")

    task_ids = []
    headers = {
        "Authorization": f"Bearer {settings.TODOIST_API_TOKEN}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
        for task in tasks:
            description_parts = []
            if task.assignees:
                label = "Ответственные" if len(task.assignees) > 1 else "Ответственный"
                description_parts.append(f"{label}: {', '.join(task.assignees)}")
            if task.description:
                description_parts.append(task.description)

            payload: dict = {"content": task.title}
            if description_parts:
                payload["description"] = "\n".join(description_parts)
            if task.due_string:
                payload["due_string"] = task.due_string
            if settings.TODOIST_PROJECT_ID:
                payload["project_id"] = settings.TODOIST_PROJECT_ID

            request_id = str(uuid.uuid4())

            response = await client.post(
                f"{TODOIST_API_BASE}/tasks",
                json=payload,
                headers={"X-Request-Id": request_id},
            )

            if response.status_code in (401, 403):
                logger.error("Todoist auth error | status={}", response.status_code)
                raise httpx.HTTPStatusError(
                    "Todoist auth failed",
                    request=response.request,
                    response=response,
                )

            if response.status_code == 429:
                logger.warning("Todoist rate limit | waiting 60s")
                import asyncio

                await asyncio.sleep(60)
                response = await client.post(
                    f"{TODOIST_API_BASE}/tasks",
                    json=payload,
                    headers={"X-Request-Id": request_id},
                )

            if response.status_code >= 500:
                logger.warning("Todoist 5xx, retry | status={}", response.status_code)
                response = await client.post(
                    f"{TODOIST_API_BASE}/tasks",
                    json=payload,
                    headers={"X-Request-Id": request_id},
                )

            response.raise_for_status()
            task_id = response.json()["id"]
            logger.debug("Todoist task создан | task_id={}", task_id)
            task_ids.append(task_id)

    return task_ids
