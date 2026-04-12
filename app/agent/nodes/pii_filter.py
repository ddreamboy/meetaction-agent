import re

from loguru import logger

from app.agent.state import AgentState

_PHONE_RE = re.compile(r"\+?[\d\s\-\(\)]{10,15}")
_EMAIL_RE = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
_PASSPORT_RE = re.compile(r"\b\d{4}\s?\d{6}\b")
_INN_RE = re.compile(r"\b\d{10,12}\b")


def _mask_pii(text: str) -> str:
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _PHONE_RE.sub("[PHONE]", text)
    text = _PASSPORT_RE.sub("[PASSPORT]", text)
    text = _INN_RE.sub("[INN]", text)
    return text


async def pii_filter_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "")
    logger.info("PII filter node | session_id={}", session_id)

    transcript_clean = _mask_pii(state["transcript_raw"])

    replaced = (
        transcript_clean.count("[EMAIL]")
        + transcript_clean.count("[PHONE]")
        + transcript_clean.count("[PASSPORT]")
        + transcript_clean.count("[INN]")
    )
    logger.info("PII filter done | session_id={} replaced={}", session_id, replaced)

    return {**state, "transcript_clean": transcript_clean}
