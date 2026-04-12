from loguru import logger

from app.agent.state import AgentState


async def hitl_gate_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "")
    logger.info(
        "HITL gate | session_id={} confirmed={} refinement_count={}",
        session_id,
        state.get("confirmed"),
        state.get("refinement_count", 0),
    )
    return state
