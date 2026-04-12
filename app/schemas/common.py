from enum import Enum

from pydantic import BaseModel


class EMeetingType(str, Enum):
    WORK_MEETING = "work_meeting"
    CONSULTATION = "consultation"


class ESessionOutcome(str, Enum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    ERROR = "error"


class SInjectionCheckResult(BaseModel):
    injection_detected: bool
    reason: str | None = None


class SClassifierResult(BaseModel):
    meeting_type: EMeetingType
    confidence: float
