import time
from inspect import signature

from loguru import logger

from app.agent.state import AgentState
from app.config import settings


async def transcribe_node(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "")
    logger.info(
        "Transcribe node | session_id={} backend={}",
        session_id,
        settings.TRANSCRIBE_BACKEND,
    )

    if settings.TRANSCRIBE_BACKEND == "deepgram":
        from app.agent.nodes._transcribe_deepgram import transcribe_deepgram

        return await transcribe_deepgram(state)

    return await _transcribe_local(state)


async def _transcribe_local(state: AgentState) -> AgentState:
    session_id = state.get("session_id", "")

    t0 = time.monotonic()
    try:
        import whisperx
        from whisperx.diarize import DiarizationPipeline

        model = whisperx.load_model(
            settings.WHISPER_MODEL_SIZE,
            device=settings.WHISPER_DEVICE,
            compute_type=settings.WHISPER_COMPUTE_TYPE,
        )
        audio = whisperx.load_audio(state["file_path"])

        sample_rate = 16000
        duration_seconds = len(audio) / sample_rate
        if duration_seconds > settings.MAX_AUDIO_DURATION_SECONDS:
            return {**state, "error_message": "Аудио превышает 90 минут"}

        result = model.transcribe(audio, batch_size=16)

        try:
            diarize_kwargs = {"device": settings.WHISPER_DEVICE}
            if settings.HF_TOKEN:
                diarize_sig = signature(DiarizationPipeline.__init__)
                if "token" in diarize_sig.parameters:
                    diarize_kwargs["token"] = settings.HF_TOKEN
                elif "use_auth_token" in diarize_sig.parameters:
                    diarize_kwargs["use_auth_token"] = settings.HF_TOKEN
                else:
                    logger.warning(
                        "HF token provided but diarization pipeline does not accept token args"
                    )
            diarize_model = DiarizationPipeline(**diarize_kwargs)
            diarize_segments = diarize_model(audio)
            result = whisperx.assign_word_speakers(diarize_segments, result)
        except Exception as diarize_exc:
            logger.warning(
                "Diarization skipped | session_id={} error={}",
                session_id,
                str(diarize_exc),
            )

        lines = []
        for segment in result.get("segments", []):
            speaker = segment.get("speaker", "speaker_0")
            text = segment.get("text", "").strip()
            if text:
                start = segment.get("start", 0)
                hours = int(start) // 3600
                minutes = (int(start) % 3600) // 60
                seconds = int(start) % 60
                lines.append(
                    f"[{hours:02d}:{minutes:02d}:{seconds:02d}] {speaker}: {text}"
                )

        transcript_raw = "\n".join(lines)

        latency_ms = int((time.monotonic() - t0) * 1000)
        logger.info(
            "Transcribe complete | session_id={} segments={} latency_ms={}",
            session_id,
            len(result.get("segments", [])),
            latency_ms,
        )
        return {**state, "transcript_raw": transcript_raw, "error_message": None}

    except Exception as exc:
        logger.error("Transcribe failed | session_id={} error={}", session_id, str(exc))
        return {**state, "error_message": f"Ошибка транскрибации: {exc}"}
