import os

from loguru import logger

from app.agent.state import AgentState
from app.config import settings

SUPPORTED_EXTENSIONS = {
    ".mp3",
    ".mp4",
    ".wav",
    ".m4a",
    ".ogg",
    ".flac",
    ".webm",
    ".mkv",
    ".avi",
}


async def ingest_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "")
    file_path = state["file_path"]
    logger.info("Ingest node | session_id={} file={}", session_id, file_path)

    if not os.path.exists(file_path):
        logger.error("File not found | session_id={} file={}", session_id, file_path)
        return {**state, "error_message": f"Файл не найден: {file_path}"}

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in SUPPORTED_EXTENSIONS:
        logger.error("Unsupported format | session_id={} ext={}", session_id, ext)
        return {**state, "error_message": f"Неподдерживаемый формат: {ext}"}

    file_size = os.path.getsize(file_path)
    max_size_bytes = settings.MAX_AUDIO_DURATION_SECONDS * 30_000
    if file_size > max_size_bytes:
        logger.warning(
            "File may exceed duration limit | session_id={} size_mb={:.1f}",
            session_id,
            file_size / 1_048_576,
        )
        return {
            **state,
            "error_message": "Файл превышает максимальную длительность 90 минут",
        }

    logger.info("Ingest ok | session_id={} size_bytes={}", session_id, file_size)
    return {**state, "error_message": None}
