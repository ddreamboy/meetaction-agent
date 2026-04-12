import time

from loguru import logger

from app.agent.state import AgentState
from app.llm.client import llm_structured_call
from app.observability.token_counter import TokenCounter
from app.schemas.extraction import SWorkGraph

SYSTEM_PROMPT = """\
You are a task extraction assistant. Extract ONLY explicit FUTURE work commitments from the meeting transcript.

IMPORTANT: Write all text fields in the SAME LANGUAGE as the transcript. Do not translate.

КРИТИЧЕСКИ ВАЖНО — различай прошедшее и будущее время:
- "я написал введение" -> НЕ ЗАДАЧА (уже сделано, прошедшее время)
- "я указал четыре источника" -> НЕ ЗАДАЧА (уже сделано)
- "мы посмотрели на код" -> НЕ ЗАДАЧА (прошедшее, отчёт о сделанном)
- "нужно поискать статьи на semantic scholar" -> ЗАДАЧА (будущее намерение)
- "закину официальные статьи" -> ЗАДАЧА (будущее действие)

Примеры по тональности:
- "я себе кофе сделаю" -> НЕ ЗАДАЧА — бытовое, не рабочее (confidence 0.0)
- "может когда-нибудь" -> НЕ ЗАДАЧА — нет коммитмента (confidence 0.1)
- "нужно сделать мини-документ по серверу" -> ЗАДАЧА (confidence 0.9)
- "уберу мусор из корня репы" -> ЗАДАЧА (confidence 0.85)

Для каждого коммитмента:
- action: конкретный глагол/фраза (в будущем времени)
- object: ПОЛНОЕ описание без местоимений и без слэнга — раскрывай "это", "там", "туда", "фигню", "штуку" через контекст; пиши конкретный технический термин (например: ".env-файлы на сервере", "pnpm-lock.yaml в корне репозитория"); если раскрыть нельзя — поле опускай
- deadline: ТОЛЬКО явные сроки ("к пятнице", "до конца недели"); никогда "скорее всего", "в будущем", "после краша"
- confidence: 0.0–1.0 (бытовые = 0.0; туманные = 0.1–0.4; реальные но нечёткие = 0.5–0.7; явные = 0.8–1.0)
- context_quote: дословная цитата из транскрипта (1–2 предложения), которая ДОКАЗЫВАЕТ что это будущий коммитмент, а не отчёт о прошлом; цитата должна содержать достаточно контекста чтобы понять ЧТО именно делается — не обрывай на местоимении

Нет цитаты — нет задачи. Be strict.
"""


async def work_extractor_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "")
    logger.info("Work extractor node | session_id={}", session_id)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"<transcript>\n{state['transcript_clean']}\n</transcript>",
        },
    ]

    t0 = time.monotonic()
    try:
        result, input_tok, output_tok = await llm_structured_call(
            messages=messages,
            response_model=SWorkGraph,
            temperature=0.0,
            node="work_extractor",
            session_id=session_id,
        )
        TokenCounter(session_id).add(input_tok, output_tok, "work_extractor")

        items = [item.model_dump() for item in result.items]
        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "Work extractor done | session_id={} items={} latency_ms={}",
            session_id,
            len(items),
            latency_ms,
        )
        return {**state, "knowledge_graph": items, "error_message": None}

    except Exception as exc:
        logger.error(
            "Work extractor failed | session_id={} error={}", session_id, str(exc)
        )
        return {**state, "error_message": "Не удалось извлечь задачи из транскрипта"}
