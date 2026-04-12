import os
import time
from datetime import date

from loguru import logger

from app.agent.state import AgentState
from app.llm.client import llm_call_with_retry
from app.observability.token_counter import TokenCounter
from app.retriever.rag import index_meeting

SUMMARY_SYSTEM_PROMPT = """\
You are a meeting summarizer. Create a concise summary of the meeting (max 500 tokens).
The summary must NOT contain any PII (names, phone numbers, emails, passport data).
Replace all personal names with roles like [SPEAKER], [MANAGER], [CLIENT].

Include: main topics discussed, decisions made, key outcomes.
Do NOT include: verbatim quotes, task lists (they are stored separately)
"""


async def rag_indexer_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "")
    logger.info("RAG indexer node | session_id={}", session_id)

    t0 = time.monotonic()

    try:
        speakers = set()
        for item in state.get("knowledge_graph", []):
            if "speaker" in item:
                speakers.add(item["speaker"])
        participants_count = max(len(speakers), 1)

        messages = [
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"<transcript>\n{state['transcript_clean']}\n</transcript>",
            },
        ]

        content, input_tok, output_tok = await llm_call_with_retry(
            messages=messages,
            temperature=0.3,
            node="rag_indexer",
            session_id=session_id,
        )
        TokenCounter(session_id).add(input_tok, output_tok, "rag_indexer")

        await index_meeting(
            session_id=session_id,
            date=date.today().isoformat(),
            meeting_type=state["meeting_type"],
            participants_count=participants_count,
            summary=content,
        )

        file_path = state.get("file_path", "")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.info("File deleted | session_id={} path={}", session_id, file_path)

        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "RAG indexer done | session_id={} latency_ms={}", session_id, latency_ms
        )

        return {
            **state,
            "transcript_raw": "",
            "transcript_clean": "",
            "file_path": "",
        }

    except Exception as exc:
        logger.error(
            "RAG indexer failed | session_id={} error={}", session_id, str(exc)
        )
        return {**state, "transcript_raw": "", "transcript_clean": "", "file_path": ""}
