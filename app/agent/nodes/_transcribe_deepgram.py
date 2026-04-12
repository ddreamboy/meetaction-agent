import time

import httpx
from loguru import logger

from app.agent.state import AgentState
from app.config import settings

DEEPGRAM_TIMEOUT_SECONDS = 600


async def transcribe_deepgram(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "")
    logger.info("Transcribe via DeepGram API | session_id={}", session_id)

    if not settings.DEEPGRAM_API_KEY:
        return {**state, "error_message": "DEEPGRAM_API_KEY не задан"}

    t0 = time.monotonic()
    try:
        with open(state["file_path"], "rb") as f:
            audio_bytes = f.read()

        params = {
            "model": settings.DEEPGRAM_MODEL,
            "language": settings.DEEPGRAM_LANGUAGE,
            "diarize": "true" if settings.DEEPGRAM_DIARIZE else "false",
            "punctuate": "true",
            "paragraphs": "true",
            "smart_format": "true",
        }

        headers = {
            "Authorization": f"Token {settings.DEEPGRAM_API_KEY}",
            "Content-Type": "audio/*",
        }

        async with httpx.AsyncClient(timeout=DEEPGRAM_TIMEOUT_SECONDS) as client:
            response = await client.post(
                settings.DEEPGRAM_API_URL,
                content=audio_bytes,
                headers=headers,
                params=params,
            )
            response.raise_for_status()

        data = response.json()
        alternative = data["results"]["channels"][0]["alternatives"][0]

        duration_seconds = data.get("metadata", {}).get("duration", 0)
        if duration_seconds > settings.MAX_AUDIO_DURATION_SECONDS:
            return {**state, "error_message": "Аудио превышает 90 минут"}

        words = alternative.get("words", [])
        lines: list[str] = []
        if settings.DEEPGRAM_DIARIZE and words and "speaker" in words[0]:
            current_speaker: int | None = None
            buffer: list[str] = []
            seg_start: float = 0.0
            for word in words:
                spk = word.get("speaker", 0)
                if spk != current_speaker:
                    if buffer:
                        start_h = int(seg_start) // 3600
                        start_min = (int(seg_start) % 3600) // 60
                        start_sec = int(seg_start) % 60
                        lines.append(
                            f"[{start_h:02d}:{start_min:02d}:{start_sec:02d}] speaker_{current_speaker}: {' '.join(buffer)}"
                        )
                    current_speaker = spk
                    seg_start = word.get("start", 0.0)
                    buffer = [word["word"]]
                else:
                    buffer.append(word["word"])
            if buffer:
                start_h = int(seg_start) // 3600
                start_min = (int(seg_start) % 3600) // 60
                start_sec = int(seg_start) % 60
                lines.append(
                    f"[{start_h:02d}:{start_min:02d}:{start_sec:02d}] speaker_{current_speaker}: {' '.join(buffer)}"
                )
            transcript_raw = "\n".join(lines)
        else:
            transcript_raw = alternative.get("transcript", "")

        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "DeepGram transcribe complete | session_id={} duration_s={:.1f} latency_ms={}",
            session_id,
            duration_seconds,
            latency_ms,
        )
        return {**state, "transcript_raw": transcript_raw, "error_message": None}

    except httpx.TimeoutException as exc:
        logger.error(
            "DeepGram timeout | session_id={} exc_type={} timeout={}s",
            session_id,
            type(exc).__name__,
            DEEPGRAM_TIMEOUT_SECONDS,
        )
        return {
            **state,
            "error_message": f"DeepGram таймаут ({DEEPGRAM_TIMEOUT_SECONDS}s) — файл слишком большой или сеть медленная",
        }
    except httpx.HTTPStatusError as exc:
        logger.error(
            "DeepGram HTTP error | session_id={} status={} body={}",
            session_id,
            exc.response.status_code,
            exc.response.text[:200],
        )
        return {
            **state,
            "error_message": f"DeepGram API ошибка {exc.response.status_code}",
        }
    except Exception as exc:
        logger.error(
            "DeepGram transcribe failed | session_id={} exc_type={} exc_repr={}",
            session_id,
            type(exc).__name__,
            repr(exc),
        )
        return {
            **state,
            "error_message": f"Ошибка транскрибации (DeepGram): {type(exc).__name__}",
        }
