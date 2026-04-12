import json
import time

from loguru import logger

from app.agent.state import AgentState
from app.llm.client import llm_structured_call
from app.observability.token_counter import TokenCounter
from app.schemas.common import EMeetingType
from app.schemas.task import SInsightList, STaskList

WORK_SYSTEM_PROMPT = """\
You are a task proposer. Convert the knowledge graph items into clear, actionable Todoist tasks.

Rules:
- SKIP items with confidence < 0.6
- MERGE duplicate or near-duplicate items into one task (same speaker + same intent)
- Write tasks in the SAME LANGUAGE as the transcript (Russian transcript -> Russian tasks)

TITLE rules (strict):
- Must be a self-contained action: verb + specific object + location/scope if relevant
- NEVER use pronouns (их, это, там, туда, он, она) — always replace with the actual noun from object field or context_quote
- NEVER use informal/slang words (фигню, штуку, хрень) — replace with the technical term
- If object field is vague or uses pronouns, EXPAND it using context_quote to name the exact thing
- The title must answer "what exactly to do, with what, and where"
- BAD: "Чистить их", "Закинуть фигню в Makefile", "Убрать мусор из корня"
- GOOD: "Удалять .env-файлы из сервера после сборки через Makefile", "Удалить посторонние файлы из корня репозитория"

DESCRIPTION rules (strict):
- Must add information that is NOT already in the title: the WHY, the trigger, the consequence, or the exact method
- NEVER just repeat or rephrase the title
- Use context_quote to extract the reason or detail
- If you cannot add new information beyond the title — set description to null
- BAD: "Убрать мусор из корня." (repeats title), "Чистить их, так как это необходимо." (no new info)
- GOOD: "Коля оставил pnpm-lock.yaml и project.yaml в корне — они не относятся к проекту и мешают навигации."

- assignees: list of speaker names responsible for this task; use the actual name from the "speaker" field in the knowledge graph item (NOT "speaker_0" labels — by this point they are already replaced with real names); if multiple items were merged and have different speakers, include all of them
- due_string: ONLY explicit deadlines ("к пятнице", "до конца недели"); null otherwise
"""

CONSULTATION_SYSTEM_PROMPT = """\
You are an insight proposer. Convert the knowledge graph items into structured insights.

For each topic create an insight with:
- topic: the subject
- key_points: list of key findings
- recommendations: list of actionable recommendations
"""


async def proposer_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "")
    logger.info(
        "Proposer node | session_id={} type={}", session_id, state["meeting_type"]
    )

    is_work = state["meeting_type"] == EMeetingType.WORK_MEETING.value
    system_prompt = WORK_SYSTEM_PROMPT if is_work else CONSULTATION_SYSTEM_PROMPT
    response_model = STaskList if is_work else SInsightList

    kg_json = json.dumps(state["knowledge_graph"], ensure_ascii=False)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"<knowledge_graph>\n{kg_json}\n</knowledge_graph>",
        },
    ]

    t0 = time.monotonic()
    try:
        result, input_tok, output_tok = await llm_structured_call(
            messages=messages,
            response_model=response_model,
            temperature=0.3,
            node="proposer",
            session_id=session_id,
        )
        TokenCounter(session_id).add(input_tok, output_tok, "proposer")

        proposed = [item.model_dump() for item in result.items]
        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "Proposer done | session_id={} proposed={} latency_ms={}",
            session_id,
            len(proposed),
            latency_ms,
        )
        return {**state, "proposed_output": proposed, "error_message": None}

    except Exception as exc:
        logger.error("Proposer failed | session_id={} error={}", session_id, str(exc))
        return {
            **state,
            "proposed_output": [],
            "error_message": "Не удалось сформировать предложения",
        }
