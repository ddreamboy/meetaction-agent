# Spec: Memory / Context

## Session State

Стейт сессии хранится в Redis через LangGraph `RedisSaver`. Один ключ на сессию: `session:{session_id}`.

### Схема

```python
class AgentState(TypedDict):
    file_path: str                  # путь к загруженному файлу (tmp)
    transcript_raw: str             # сырой транскрипт с таймкодами и speaker labels
    transcript_clean: str           # транскрипт после PII-фильтра
    meeting_type: str               # "work_meeting" | "consultation"
    knowledge_graph: list[dict]     # work:   [{speaker, action, object, deadline}]
                                    # consult: [{topic, insight, recommendation}]
    proposed_output: list[dict]     # task / insight JSON для показа в UI
    user_feedback: str | None       # фидбек при reject
    refinement_count: int           # счетчик итераций refine (max=2)
    confirmed: bool                 # HITL флаг - разблокирует creator node
    created_task_ids: list[str]     # ID задач в Todoist (только work_meeting)
```

### Lifecycle

```
Загрузка файла
    -> создание session_id (uuid4)
    -> инициализация AgentState в Redis

Завершение сессии (confirmed=True + creator отработал)
    -> transcript_raw и transcript_clean удаляются из стейта
    -> file_path удаляется с диска
    -> остаток стейта (task_ids, meeting_type) хранится еще 1 час, затем TTL истекает

Отказ пользователя / таймаут > 30 мин бездействия
    -> весь стейт удаляется из Redis (TTL)
```

### TTL политика

| Ключ | TTL |
| :--- | :--- |
| Активная сессия | 30 мин с момента последнего обновления |
| Завершенная сессия (без транскрипта) | 1 час |

---

## Context Budget (LLM)

Gemini 2.5 Flash Lite поддерживает 1M токенов контекста - транскрипт любой длины укладывается в один вызов. Map-reduce не нужен.

Тем не менее формируем промпт аккуратно:

| Часть промпта | Лимит |
| :--- | :--- |
| System prompt | ~500 токенов |
| Транскрипт (`<transcript>...</transcript>`) | без лимита, вся длина |
| RAG-контекст (`<context>...</context>`) | ≤ 4000 токенов (top-3 chunks) |
| История refine (фидбек пользователя) | ≤ 500 токенов |
| Output schema (Pydantic JSON schema) | ~300 токенов |

### Изоляция данных в промпте

Транскрипт всегда подается в отдельном теге, изолированно от системных инструкций:

```
[system prompt]
You are a meeting analysis assistant. Analyze the transcript below.
Extract tasks according to the schema.

<transcript>
speaker_0: Иван, сдашь отчет к пятнице?
speaker_1: Да, сделаю.
</transcript>
```

Системные инструкции и данные пользователя **никогда не конкатенируются** в одну строку.

---

## Long-term Memory (RAG)

Реализована через Qdrant. Персистентна между сессиями.

Что сохраняется после завершения сессии:

```python
{
    "session_id": "abc-123",
    "date": "2026-03-31",
    "meeting_type": "work_meeting",
    "participants_count": 3,
    "summary": "Обсудили релиз v2.1. [NAME] возьмет ревью до пятницы..."
    # PII уже замаскированы в summary
}
```

Что **не сохраняется**: сырой транскрипт, промпты LLM
