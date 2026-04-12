from typing import TypedDict


class AgentState(TypedDict):
    session_id: str
    file_path: str
    transcript_raw: str
    transcript_clean: str
    meeting_type: str  # EMeetingType.value
    knowledge_graph: list[dict]  # work: [{speaker, action, object, deadline}]
    # consult: [{topic, insight, recommendation}]
    proposed_output: list[dict]  # STaskSchema / SInsightSchema
    user_feedback: str | None
    refinement_count: int
    confirmed: bool
    created_task_ids: list[str]
    error_message: str | None
    current_step: str
    progress_steps: list[str]
