import shutil
import tempfile
import uuid
from contextlib import asynccontextmanager

import redis.asyncio as aioredis
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import RedirectResponse
from langgraph.checkpoint.redis import AsyncRedisSaver
from loguru import logger

from app.agent.graph import build_graph
from app.config import settings
from app.observability.logger import setup_logging
from app.retriever.rag import generate_rag_answer, search_meetings
from app.schemas.api import (
    SClarifyRequest,
    SClarifyResponse,
    SConfirmRequest,
    SProcessResponse,
    SQueryRequest,
    SQueryResponse,
    SRefineRequest,
    SRenameSpeakersRequest,
    SRenameSpeakersResponse,
)

setup_logging()

redis_client: aioredis.Redis | None = None
graph = None
checkpointer: AsyncRedisSaver | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global redis_client, graph, checkpointer
    redis_client = aioredis.from_url(settings.REDIS_URL, decode_responses=False)
    await redis_client.ping()
    logger.info("Redis connected | url={}", settings.REDIS_URL)

    checkpointer = AsyncRedisSaver(redis_url=settings.REDIS_URL)
    await checkpointer.asetup()
    logger.info("Redis checkpointer setup complete")

    graph = build_graph(checkpointer=checkpointer)
    logger.info("Graph compiled")
    yield
    await redis_client.aclose()


app = FastAPI(title="MeetAction Agent", lifespan=lifespan)


def _compute_task_grouping(proposed: list[dict]) -> dict:
    by_assignee: dict[str, list[str]] = {}
    multiple: list[dict] = []
    no_assignee: list[str] = []

    for task in proposed:
        assignees = task.get("assignees", [])
        title = task.get("title", "")
        if not assignees:
            no_assignee.append(title)
        elif len(assignees) == 1:
            a = assignees[0]
            by_assignee.setdefault(a, []).append(title)
        else:
            multiple.append({"title": title, "assignees": assignees})

    return {
        "by_assignee": by_assignee,
        "multiple_assignees": multiple,
        "no_assignee": no_assignee,
    }


def _get_graph_config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}}


async def _run_graph_session(session_id: str, initial_state: dict) -> None:
    await graph.ainvoke(initial_state, config=_get_graph_config(session_id))


@app.post("/process", response_model=SProcessResponse)
async def process_meeting(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    logger.info("New session | session_id={} filename={}", session_id, file.filename)

    suffix = "." + (file.filename.rsplit(".", 1)[-1] if "." in file.filename else "bin")
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        shutil.copyfileobj(file.file, tmp)
        tmp.close()
    except Exception as exc:
        logger.error(
            "File upload failed | session_id={} error={}", session_id, str(exc)
        )
        raise HTTPException(status_code=500, detail="Ошибка загрузки файла")

    initial_state = {
        "session_id": session_id,
        "file_path": tmp.name,
        "transcript_raw": "",
        "transcript_clean": "",
        "meeting_type": "",
        "knowledge_graph": [],
        "proposed_output": [],
        "user_feedback": None,
        "refinement_count": 0,
        "confirmed": False,
        "created_task_ids": [],
        "error_message": None,
        "current_step": "Инициализация",
        "progress_steps": [],
    }

    try:
        await _run_graph_session(session_id, initial_state)
    except Exception as exc:
        logger.error(
            "Graph execution error | session_id={} error={}", session_id, str(exc)
        )
        raise HTTPException(status_code=500, detail=str(exc))

    current_state = await graph.aget_state(_get_graph_config(session_id))
    values = current_state.values if current_state and current_state.values else {}

    proposed_output = values.get("proposed_output", [])
    error_message = values.get("error_message")
    current_step = values.get("current_step", "")
    progress_steps = values.get("progress_steps", [])
    transcript_text = (
        values.get("transcript_clean") or values.get("transcript_raw") or ""
    )

    if proposed_output and not error_message:
        status = "awaiting_confirmation"
    else:
        status = "completed"

    return SProcessResponse(
        session_id=session_id,
        status=status,
        proposed_output=proposed_output,
        error_message=error_message,
        current_step=current_step,
        progress_steps=progress_steps,
        transcript_text=transcript_text,
    )


@app.post("/confirm")
async def confirm_session(body: SConfirmRequest):
    session_id = body.session_id
    logger.info("Confirm | session_id={}", session_id)

    config = _get_graph_config(session_id)
    state = await graph.aget_state(config)
    if not state or not state.values:
        raise HTTPException(status_code=404, detail="Сессия не найдена")

    update = {**state.values, "confirmed": True}
    await graph.aupdate_state(config, update)

    try:
        await graph.ainvoke(None, config=config)
    except Exception as exc:
        logger.error(
            "Confirm graph error | session_id={} error={}", session_id, str(exc)
        )
        raise HTTPException(status_code=500, detail=str(exc))

    final = await graph.aget_state(config)
    task_ids = final.values.get("created_task_ids", [])
    error = final.values.get("error_message")
    current_step = final.values.get("current_step", "")
    progress_steps = final.values.get("progress_steps", [])
    proposed = final.values.get("proposed_output", [])

    return {
        "session_id": session_id,
        "status": "completed",
        "created_task_ids": task_ids,
        "error_message": error,
        "current_step": current_step,
        "progress_steps": progress_steps,
        "task_grouping": _compute_task_grouping(proposed),
    }


@app.post("/refine")
async def refine_session(body: SRefineRequest):
    session_id = body.session_id
    logger.info("Refine | session_id={}", session_id)

    config = _get_graph_config(session_id)
    state = await graph.aget_state(config)
    if not state or not state.values:
        raise HTTPException(status_code=404, detail="Сессия не найдена")

    update = {**state.values, "user_feedback": body.feedback, "confirmed": False}
    await graph.aupdate_state(config, update)

    try:
        await graph.ainvoke(None, config=config)
    except Exception as exc:
        logger.error(
            "Refine graph error | session_id={} error={}", session_id, str(exc)
        )
        raise HTTPException(status_code=500, detail=str(exc))

    final = await graph.aget_state(config)
    proposed = final.values.get("proposed_output", [])
    current_step = final.values.get("current_step", "")
    progress_steps = final.values.get("progress_steps", [])

    return {
        "session_id": session_id,
        "status": "awaiting_confirmation",
        "proposed_output": proposed,
        "current_step": current_step,
        "progress_steps": progress_steps,
    }


@app.post("/rename_speakers", response_model=SRenameSpeakersResponse)
async def rename_speakers(body: SRenameSpeakersRequest):
    config = _get_graph_config(body.session_id)
    state = await graph.aget_state(config)
    if not state or not state.values:
        raise HTTPException(status_code=404, detail="Сессия не найдена")

    speaker_map = {k: v.strip() for k, v in body.speaker_map.items() if v.strip()}

    def _replace(text: str) -> str:
        for old, new in sorted(
            speaker_map.items(), key=lambda x: len(x[0]), reverse=True
        ):
            if old != new:
                text = text.replace(f"{old}:", f"{new}:")
        return text

    values = dict(state.values)
    values["transcript_raw"] = _replace(values.get("transcript_raw", ""))
    values["transcript_clean"] = _replace(values.get("transcript_clean", ""))

    kg = []
    for item in values.get("knowledge_graph", []):
        item = dict(item)
        if item.get("speaker") in speaker_map:
            item["speaker"] = speaker_map[item["speaker"]]
        kg.append(item)
    values["knowledge_graph"] = kg

    po = []
    for item in values.get("proposed_output", []):
        item = dict(item)
        item["assignees"] = [speaker_map.get(a, a) for a in item.get("assignees", [])]
        po.append(item)
    values["proposed_output"] = po

    await graph.aupdate_state(config, values)
    logger.info(
        "Speakers renamed | session_id={} map={}",
        body.session_id,
        speaker_map,
    )

    transcript_text = values.get("transcript_clean") or values.get("transcript_raw", "")
    return SRenameSpeakersResponse(
        transcript_text=transcript_text,
        proposed_output=po,
    )


@app.post("/clarify", response_model=SClarifyResponse)
async def clarify_feedback(body: SClarifyRequest):
    import json

    from app.llm.client import llm_call_with_retry

    config = _get_graph_config(body.session_id)
    state = await graph.aget_state(config)
    if not state or not state.values:
        raise HTTPException(status_code=404, detail="Сессия не найдена")

    proposed = state.values.get("proposed_output", [])
    tasks_str = json.dumps(proposed, ensure_ascii=False, indent=2)
    transcript = (
        state.values.get("transcript_clean") or state.values.get("transcript_raw") or ""
    )

    system = (
        "Ты ассистент, помогающий уточнить правки к списку задач из рабочей встречи.\n\n"
        f"Транскрипция встречи (с актуальными именами спикеров):\n<transcript>\n{transcript}\n</transcript>\n\n"
        f"Текущий список задач:\n{tasks_str}\n\n"
        "Твоя задача: понять что именно хочет изменить пользователь, "
        "переформулировать это чётко и конкретно своими словами, "
        "и спросить правильно ли ты понял. "
        "Отвечай на языке пользователя. Будь краток 2-4 предложения."
    )

    messages: list[dict] = [{"role": "system", "content": system}]
    for msg in body.chat_history:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": body.message})

    try:
        content, _, _ = await llm_call_with_retry(
            messages=messages,
            temperature=0.3,
            node="clarify",
            session_id=body.session_id,
        )
    except Exception as exc:
        logger.error(
            "Clarify failed | session_id={} error={}", body.session_id, str(exc)
        )
        raise HTTPException(status_code=500, detail=str(exc))

    return SClarifyResponse(bot_message=content)


@app.post("/query", response_model=SQueryResponse)
async def query_meetings(body: SQueryRequest):
    logger.info("RAG query | query_len={}", len(body.query))

    try:
        contexts = await search_meetings(body.query, body.meeting_type)
        answer = await generate_rag_answer(body.query, contexts)
        return SQueryResponse(answer=answer, sources_count=len(contexts))
    except Exception as exc:
        logger.error("RAG query failed | error={}", str(exc))
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ui", include_in_schema=False)
@app.get("/ui/", include_in_schema=False)
async def ui_redirect():
    return RedirectResponse(url="http://localhost:7860")
