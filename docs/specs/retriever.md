# Spec: Retriever

## Назначение

Отвечает на вопросы пользователя по прошлым встречам - "кто обещал сдать отчет?", "что решили на созвоне с клиентом?". Используется только в RAG-режиме (чат в Gradio UI)

---

## Источники данных

| Источник | Формат | Когда добавляется |
| :--- | :--- | :--- |
| Summary рабочей встречи | Текст ≤ 500 токенов | После `confirmed=True` + создания задач в Todoist |
| Summary консультации | Текст ≤ 500 токенов | После `confirmed=True` + сохранения инсайтов |

Транскрипт и исходный аудио/видео файл в индекс **не попадают**

---

## Индекс

| Параметр | Значение |
| :--- | :--- |
| **Хранилище** | Qdrant |
| **Коллекция** | `meeting_summaries` |
| **Векторная размерность** | 768 (gemini-embedding-001) |
| **Метрика** | Cosine similarity |
| **Payload (metadata)** | `date: str`, `meeting_type: work_meeting\|consultation`, `participants_count: int`, `session_id: str` |

---

## Chunk Strategy

Один документ = одно summary встречи. Разбивки на части нет - summary генерируется LLM и укладывается в ≤ 500 токенов

При превышении 500 токенов LLM-промпт instructed truncate до лимита

---

## Поиск

```python
results = qdrant_client.search(
    collection_name="meeting_summaries",
    query_vector=embedding,
    limit=3,
    score_threshold=0.75,
    query_filter=Filter(           # опционально
        must=[FieldCondition(
            key="meeting_type",
            match=MatchValue(value="work_meeting")
        )]
    )
)
```

- top-3 по cosine similarity
- threshold 0.75 - ниже считается нерелевантным
- фильтрация по `meeting_type` передается агентом если пользователь явно спрашивает про определенный тип встречи

---

## Reranking

Нет - out-of-scope для PoC

---

## Ограничения

- RAG недоступен до первой успешно завершенной сессии (коллекция пуста)
- Similarity < 0.75 -> ответ: "Нет релевантных данных по прошлым встречам"
- Qdrant недоступен -> RAG отключается для сессии, пользователь получает предупреждение, остальные функции работают
- PII в индексе отсутствуют - summary генерируется из уже замаскированного транскрипта
