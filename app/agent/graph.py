from collections.abc import Awaitable, Callable

from langgraph.checkpoint.redis import AsyncRedisSaver
from langgraph.graph import END, StateGraph

from app.agent.nodes.classifier import classifier_node
from app.agent.nodes.creator import creator_node
from app.agent.nodes.extractor_consultation import consultation_extractor_node
from app.agent.nodes.extractor_work import work_extractor_node
from app.agent.nodes.hitl_gate import hitl_gate_node
from app.agent.nodes.ingest import ingest_node
from app.agent.nodes.injection_check import injection_check_node
from app.agent.nodes.pii_filter import pii_filter_node
from app.agent.nodes.proposer import proposer_node
from app.agent.nodes.rag_indexer import rag_indexer_node
from app.agent.nodes.refine import refine_node
from app.agent.nodes.transcribe import transcribe_node
from app.agent.state import AgentState
from app.config import settings
from app.schemas.common import EMeetingType

NodeCallable = Callable[[AgentState], Awaitable[AgentState]]


def _with_progress(step_label: str, node: NodeCallable) -> NodeCallable:
    async def wrapped(state: AgentState) -> AgentState:
        progress_steps = list(state.get("progress_steps", []))
        progress_steps.append(step_label)
        next_state = {
            **state,
            "current_step": step_label,
            "progress_steps": progress_steps,
        }
        return await node(next_state)

    wrapped.__name__ = f"{node.__name__}_with_progress"
    return wrapped


def _route_ingest(state: AgentState) -> str:
    if state.get("error_message"):
        return END
    if state.get("transcript_raw"):
        return "pii_filter"
    return "transcribe"


def _route_transcribe(state: AgentState) -> str:
    if state.get("error_message"):
        return END
    return "pii_filter"


def _route_injection_check(state: AgentState) -> str:
    if state.get("error_message"):
        return END
    return "classifier"


def _route_classifier(state: AgentState) -> str:
    if state.get("meeting_type") == EMeetingType.CONSULTATION.value:
        return "consultation_extractor"
    return "work_extractor"


def _route_extractor(state: AgentState) -> str:
    if state.get("error_message"):
        return END
    return "proposer"


def _route_proposer(state: AgentState) -> str:
    if not state.get("proposed_output"):
        return END
    return "hitl_gate"


def _route_hitl_gate(state: AgentState) -> str:
    if state.get("confirmed"):
        return "creator"

    feedback = (state.get("user_feedback") or "").strip()
    refinement_count = state.get("refinement_count", 0)

    if not feedback:
        return END

    if refinement_count >= settings.MAX_REFINEMENT_COUNT:
        return END

    return "refine"


def _route_creator(state: AgentState) -> str:
    return "rag_indexer"


def build_graph(checkpointer: AsyncRedisSaver | None = None):
    graph = StateGraph(AgentState)

    graph.add_node("ingest", _with_progress("Проверяем файл", ingest_node))
    graph.add_node(
        "transcribe", _with_progress("Транскрибируем запись", transcribe_node)
    )
    graph.add_node(
        "pii_filter", _with_progress("Маскируем персональные данные", pii_filter_node)
    )
    graph.add_node(
        "injection_check",
        _with_progress("Проверяем transcript на инъекции", injection_check_node),
    )
    graph.add_node(
        "classifier", _with_progress("Определяем тип встречи", classifier_node)
    )
    graph.add_node(
        "work_extractor",
        _with_progress("Извлекаем рабочие задачи", work_extractor_node),
    )
    graph.add_node(
        "consultation_extractor",
        _with_progress(
            "Извлекаем консультационные инсайты", consultation_extractor_node
        ),
    )
    graph.add_node("proposer", _with_progress("Формируем предложения", proposer_node))
    graph.add_node(
        "hitl_gate",
        _with_progress("Ожидаем подтверждение пользователя", hitl_gate_node),
    )
    graph.add_node(
        "refine", _with_progress("Уточняем предложения по фидбеку", refine_node)
    )
    graph.add_node("creator", _with_progress("Создаем задачи в Todoist", creator_node))
    graph.add_node(
        "rag_indexer",
        _with_progress("Индексируем встречу в базу знаний", rag_indexer_node),
    )

    graph.set_entry_point("ingest")

    graph.add_conditional_edges(
        "ingest",
        _route_ingest,
        {"transcribe": "transcribe", "pii_filter": "pii_filter", END: END},
    )
    graph.add_conditional_edges(
        "transcribe", _route_transcribe, {"pii_filter": "pii_filter", END: END}
    )
    graph.add_edge("pii_filter", "injection_check")
    graph.add_conditional_edges(
        "injection_check",
        _route_injection_check,
        {"classifier": "classifier", END: END},
    )
    graph.add_conditional_edges(
        "classifier",
        _route_classifier,
        {
            "work_extractor": "work_extractor",
            "consultation_extractor": "consultation_extractor",
        },
    )
    graph.add_conditional_edges(
        "work_extractor", _route_extractor, {"proposer": "proposer", END: END}
    )
    graph.add_conditional_edges(
        "consultation_extractor", _route_extractor, {"proposer": "proposer", END: END}
    )
    graph.add_conditional_edges(
        "proposer", _route_proposer, {"hitl_gate": "hitl_gate", END: END}
    )
    graph.add_conditional_edges(
        "hitl_gate",
        _route_hitl_gate,
        {"creator": "creator", "refine": "refine", END: END},
    )
    graph.add_edge("refine", "hitl_gate")
    graph.add_conditional_edges(
        "creator", _route_creator, {"rag_indexer": "rag_indexer"}
    )
    graph.add_edge("rag_indexer", END)

    if checkpointer is None:
        # langgraph Redis saver expects a Redis URL, not a redis client instance.
        checkpointer = AsyncRedisSaver(redis_url=settings.REDIS_URL)

    return graph.compile(
        checkpointer=checkpointer,
        # Pause before HITL gate; resume is driven by /confirm and /refine endpoints.
        interrupt_before=["hitl_gate"],
    )
