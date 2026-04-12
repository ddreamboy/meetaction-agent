from pydantic import BaseModel


class SWorkGraphItem(BaseModel):
    speaker: str
    action: str
    object: str | None = None
    deadline: str | None = None
    confidence: float  # 0.0-1.0: насколько это реальный коммитмент, а не упоминание
    context_quote: str  # дословная цитата из транскрипта, обосновывающая коммитмент


class SWorkGraph(BaseModel):
    items: list[SWorkGraphItem]


class SConsultationGraphItem(BaseModel):
    topic: str
    insight: str
    recommendation: str | None = None
    context_quote: str  # дословная цитата из транскрипта, откуда взят инсайт


class SConsultationGraph(BaseModel):
    items: list[SConsultationGraphItem]
