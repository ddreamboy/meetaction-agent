import json
import time

from loguru import logger

from app.agent.state import AgentState
from app.llm.client import llm_structured_call
from app.observability.token_counter import TokenCounter
from app.schemas.common import EMeetingType
from app.schemas.task import SInsightList, SRefineTaskList

WORK_SYSTEM_PROMPT = """\
You are a task refiner. The user has provided feedback on the proposed tasks.
Revise the tasks according to the feedback.

Rules:
- Keep the same language as the existing tasks — do not translate
- CRITICAL: Copy the 'assignees' array EXACTLY from matching tasks in Current output.
  Do NOT set assignees to [] unless the user explicitly asks to remove all assignees.
  If a refined task corresponds to an existing task, it must have the same assignees.
- If the user merges tasks, combine their assignees into one list (no duplicates)
- Use speaker names as they appear in the transcript — never use "speaker_0" style labels
- If the feedback is just an acknowledgment (e.g. "ok", "yes", "correct") with no specific changes requested,
  return the current tasks unchanged
"""

CONSULTATION_SYSTEM_PROMPT = """\
You are an insight refiner. The user has rejected the proposed insights and provided feedback.
Revise the insights according to the feedback.
Keep the same language as the existing insights — do not translate
"""


async def refine_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "")
    logger.info(
        "Refine node | session_id={} refinement_count={}",
        session_id,
        state.get("refinement_count", 0),
    )

    is_work = state["meeting_type"] == EMeetingType.WORK_MEETING.value
    system_prompt = WORK_SYSTEM_PROMPT if is_work else CONSULTATION_SYSTEM_PROMPT
    response_model = SRefineTaskList if is_work else SInsightList

    current_tasks = state["proposed_output"]
    current_json = json.dumps(current_tasks, ensure_ascii=False, indent=2)
    feedback = state.get("user_feedback") or ""

    assignees_summary = "\n".join(
        f'  - "{t.get("title", "")}" -> assignees: {t.get("assignees", [])}'
        for t in current_tasks
    )

    transcript = state.get("transcript_clean") or state.get("transcript_raw") or ""
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"<transcript>\n{transcript}\n</transcript>\n\n"
                f"Current tasks assignees (DO NOT CHANGE unless user explicitly asks):\n{assignees_summary}\n\n"
                f"Current tasks (full JSON — return ALL fields for each task):\n{current_json}\n\n"
                f"User feedback:\n{feedback}"
            ),
        },
    ]

    t0 = time.monotonic()
    try:
        result, input_tok, output_tok = await llm_structured_call(
            messages=messages,
            response_model=response_model,
            temperature=0.3,
            node="refine",
            session_id=session_id,
        )
        TokenCounter(session_id).add(input_tok, output_tok, "refine")

        proposed = [item.model_dump() for item in result.items]
        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "Refine done | session_id={} proposed={} latency_ms={}",
            session_id,
            len(proposed),
            latency_ms,
        )
        return {
            **state,
            "proposed_output": proposed,
            "refinement_count": state.get("refinement_count", 0) + 1,
            "user_feedback": None,
            "error_message": None,
        }

    except Exception as exc:
        logger.error("Refine failed | session_id={} error={}", session_id, str(exc))
        return {
            **state,
            "refinement_count": state.get("refinement_count", 0) + 1,
            "error_message": "Не удалось уточнить предложения",
        }
