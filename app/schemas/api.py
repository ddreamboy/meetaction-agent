from pydantic import BaseModel, Field

from app.schemas.common import EMeetingType


class SProcessResponse(BaseModel):
    session_id: str
    status: str
    proposed_output: list[dict] = Field(default_factory=list)
    transcript_text: str = ""
    error_message: str | None = None
    current_step: str = ""
    progress_steps: list[str] = Field(default_factory=list)


class SConfirmRequest(BaseModel):
    session_id: str


class SRefineRequest(BaseModel):
    session_id: str
    feedback: str


class SClarifyRequest(BaseModel):
    session_id: str
    message: str
    chat_history: list[dict] = Field(
        default_factory=list
    )  # [{"role": "user"|"assistant", "content": str}]


class SClarifyResponse(BaseModel):
    bot_message: str


class SRenameSpeakersRequest(BaseModel):
    session_id: str
    speaker_map: dict[str, str]  # {"speaker_0": "Иван", "speaker_1": "Катя"}


class SRenameSpeakersResponse(BaseModel):
    transcript_text: str
    proposed_output: list[dict]


class SProcessTextRequest(BaseModel):
    transcript: str
    meeting_type: str | None = None  # "work_meeting" | "consultation" | None (auto-classify)


class SQueryRequest(BaseModel):
    query: str
    meeting_type: EMeetingType | None = None


class SQueryResponse(BaseModel):
    answer: str
    sources_count: int
    sources: list[dict] = Field(default_factory=list)
