# Spec: Tools / APIs

## Todoist Tool

### Контракт

```python
def create_tasks(tasks: list[TaskSchema], state: AgentState) -> list[str]:
    """
    Создает задачи в Todoist. Вызов физически заблокирован без confirmed=True.

    Args:
        tasks: список задач из AgentState.proposed_output
        state: текущий стейт агента

    Returns:
        list[str]: список созданных task_id

    Raises:
        PermissionError: если state["confirmed"] != True
        TodoistAPIError: при 4xx/5xx от Todoist API
    """
    if not state["confirmed"]:
        raise PermissionError("create_tasks called without user confirmation")
    pass
```

### Схема задачи

```python
class TaskSchema(BaseModel):
    title: str                  # обязательно, max 500 символов
    description: str | None     # опционально
    assignee: str | None        # имя из транскрипта, не ID Todoist
    due_string: str | None      # "tomorrow", "next friday", etc.
```

> Поле `assignee` пишется в description как текст

### Эндпоинт

| Параметр | Значение |
| :--- | :--- |
| **URL** | `POST https://api.todoist.com/api/v1/tasks` |
| **Auth** | `Authorization: Bearer $TODOIST_API_TOKEN` |
| **Content-Type** | `application/json` |
| **Timeout** | 10 сек |
| **Retry** | 1 retry при 5xx, exponential backoff 1s -> 2s |
| **Retry при 4xx** | Нет - клиентская ошибка, retry не поможет |

### Коды ошибок

| Код | Причина | Действие |
| :--- | :--- | :--- |
| 401 | Неверный токен | Показать пользователю: "Проверьте TODOIST_API_TOKEN" |
| 403 | Нет прав | Показать: "Токен не имеет прав на создание задач" |
| 429 | Rate limit | Подождать `Retry-After` секунд, retry |
| 5xx | Сервер Todoist | 1 retry -> fallback: показать задачи в UI |
| timeout | Сеть / сервер | 1 retry -> fallback: показать задачи в UI |

### Side Effects

Создание задачи - **необратимое действие** Удаление в рамках PoC не реализуется
Пользователь должен явно нажать "Подтвердить" в UI перед вызовом

---

## LLM API

### LLM (gemini-2.5-flash-lite-preview-09-2025)

Используется в узлах: Injection Check, Meeting Classifier, Work Extractor, Consultation Extractor, Proposer, Refine, RAG Generation

| Параметр | Значение |
| :--- | :--- |
| **Endpoint** | `https://routerai.ru/api/v1` |
| **Auth** | `x-goog-api-key: $GOOGLE_API_KEY` |
| **Контекст** | 1M токенов |
| **Timeout** | 30 секунд |
| **Retry** | Exponential backoff: 2s -> 4s, до 2 попыток |
| **Temperature** | 0.0 для extraction/classification, 0.3 для proposer/refine |

### Embeddings (gemini-embedding-001)

Используется в RAG Indexer и Retriever.

| Параметр | Значение |
| :--- | :--- |
| **Размерность** | 768 |
| **Макс. контекст** | 20K токенов |
| **Стоимость** | 15 ₽/1M токенов вход, выход бесплатный |

### Защита от превышения бюджета

```python
# Перед каждым LLM-вызовом
if session_token_counter.input_tokens > 500_000:
    logger.warning("Token budget warning: >500k input tokens in session")
```

При превышении 500k токенов - предупреждение в логах, вызов не блокируется

### Коды ошибок

| Код | Причина | Действие |
| :--- | :--- | :--- |
| 400 | Невалидный запрос / превышен контекст | Логировать, показать ошибку пользователю |
| 401 / 403 | Неверный API ключ | Показать: "Проверьте LLM_API_KEY" |
| 429 | Rate limit | Retry с exponential backoff |
| 5xx | Сервер LLM | Retry x2 -> показать ошибку пользователю |
