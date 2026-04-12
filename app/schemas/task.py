from pydantic import BaseModel, field_validator


class STaskSchema(BaseModel):
    title: str
    description: str | None = None
    assignees: list[str] = []
    due_string: str | None = None

    @field_validator("title")
    @classmethod
    def title_max_length(cls, v: str) -> str:
        if len(v) > 500:
            return v[:500]
        return v


class SInsightSchema(BaseModel):
    topic: str
    key_points: list[str]
    recommendations: list[str]


class STaskList(BaseModel):
    items: list[STaskSchema]


class SInsightList(BaseModel):
    items: list[SInsightSchema]
