# Spec: Agent / Orchestrator

## Фреймворк

LangGraph `StateGraph` с `RedisSaver` checkpointer.

---

## Узлы и переходы


> START
>   -> ingest
>   -> transcribe
>   -> pii_filter
>   -> injection_check
>   -> classifier
>   -> extractor           (work_extractor | consultation_extractor)
>   -> proposer            (task_proposer | insight_proposer)
>   -> [INTERRUPT] hitl_gate
>   -> refine              (если reject, возврат к hitl_gate)
>   -> creator             (todoist_tool | save_insights)
>   -> rag_indexer
> END

---

## Правила переходов

| От | Условие | К |
| :--- | :--- | :--- |
| `ingest` | файл валиден | `transcribe` |
| `ingest` | файл > 90 мин или неверный формат | `END` (ошибка) |
| `transcribe` | успех | `pii_filter` |
| `transcribe` | exception | `END` (ошибка) |
| `injection_check` | ok | `classifier` |
| `injection_check` | injection detected | `END` (abort) |
| `classifier` | `work_meeting` | `work_extractor` |
| `classifier` | `consultation` | `consultation_extractor` |
| `classifier` | невалидный ответ после retry x2 | `work_extractor` (default) |
| `extractor` | valid JSON | `proposer` |
| `extractor` | невалидный JSON после retry x2 | `END` (fallback: саммари) |
| `proposer` | непустой список | `hitl_gate` |
| `proposer` | пустой список | `END` (fallback: текстовый саммари) |
| `hitl_gate` | `confirmed=True` | `creator` |
| `hitl_gate` | reject + feedback, `refinement_count < 2` | `refine` |
| `hitl_gate` | reject + feedback, `refinement_count >= 2` | `END` (показать для ручного редактирования) |
| `refine` | всегда | `hitl_gate` |
| `creator` | успех | `rag_indexer` |
| `creator` | Todoist 5xx / timeout | `rag_indexer` + fallback в UI |
| `rag_indexer` | всегда | `END` (success) |

---

## Stop Conditions

- `END` достигнут по любой ветке (успех или ошибка)
- `refinement_count >= 2` - принудительный выход из цикла refine
- Таймаут сессии в Redis (30 мин бездействия) - стейт удаляется

---

## Retry / Fallback политика

| Узел | Retry | Fallback при исчерпании |
| :--- | :--- | :--- |
| `injection_check` | нет | abort сессии |
| `classifier` | x2, затем default `work_meeting` | продолжить |
| `work_extractor` | x2 | показать текстовый саммари, `END` |
| `consultation_extractor` | x2 | показать текстовый саммари, `END` |
| `proposer` | x2 | показать текстовый саммари, `END` |
| `creator` (Todoist) | x1 при 5xx | показать задачи в UI, продолжить к `rag_indexer` |
| LLM вызовы (все) | exponential backoff 2s -> 4s | см. выше по узлу |

---

## Human-in-the-Loop

Реализован через `interrupt_before("creator")` в LangGraph:

```python
graph = StateGraph(AgentState)
# ... добавление узлов ...
graph.compile(
    checkpointer=RedisSaver(redis_client),
    interrupt_before=["creator"]
)
```

После `interrupt_before` граф останавливается и ждет явного `resume` от пользователя через API эндпоинт `/confirm`. Дополнительная проверка на уровне кода:

```python
def creator_node(state: AgentState) -> AgentState:
    if not state["confirmed"]:
        raise PermissionError("creator called without confirmed=True")
    ...
```

---

## Конфигурация

| Параметр | Значение | Источник |
| :--- | :--- | :--- |
| `MAX_REFINEMENT_COUNT` | 2 | `config.py` |
| `SESSION_TTL_SECONDS` | 1800 (30 мин) | `config.py` |
| `CLASSIFIER_DEFAULT` | `"work_meeting"` | `config.py` |
| `LLM_TIMEOUT` | 60 сек | `config.py` |
| `LLM_RETRY_COUNT` | 2 | `config.py` |
