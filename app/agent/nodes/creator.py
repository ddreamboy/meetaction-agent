from loguru import logger

from app.agent.state import AgentState
from app.schemas.common import EMeetingType
from app.schemas.task import STaskSchema
from app.tools.todoist import create_tasks


async def creator_node(state: AgentState) -> AgentState:
    if not state["confirmed"]:
        raise PermissionError("creator вызван без confirmed=True")

    session_id = state.get("session_id", "")
    logger.info(
        "Creator node | session_id={} meeting_type={}",
        session_id, state["meeting_type"],
    )

    if state["meeting_type"] != EMeetingType.WORK_MEETING.value:
        return {**state, "created_task_ids": []}

    tasks = [STaskSchema(**item) for item in state["proposed_output"]]

    try:
        task_ids = await create_tasks(tasks, state)
        logger.info(
            "Creator done | session_id={} created={}",
            session_id, len(task_ids),
        )
        return {**state, "created_task_ids": task_ids, "error_message": None}

    except Exception as exc:
        logger.error("Creator Todoist error | session_id={} error={}", session_id, str(exc))
        return {
            **state,
            "created_task_ids": [],
            "error_message": f"Не удалось создать задачи в Todoist: {exc}. Задачи показаны в UI.",
        }
