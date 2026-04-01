# Diagrams: MeetAction Agent (PoC)

---

## C4 Level 1 - Context

Показывает систему целиком, ее пользователей и внешние зависимости.

```mermaid
C4Context
    title MeetAction Agent - System Context

    Person(user, "Пользователь", "PM / тимлид / аналитик, участник рабочих созвонов")

    System(meetaction, "MeetAction Agent", "Принимает запись встречи, извлекает задачи / инсайты, создает карточки в таск-трекере")

    System_Ext(todoist, "Todoist", "Таск-трекер. REST API v2")
    System_Ext(gemini, "Google AI API", "LLM: gemini-2.5-flash-lite Embeddings: gemini-embedding-001")

    Rel(user, meetaction, "Загружает запись встречи, подтверждает задачи / инсайты", "Gradio UI / HTTPS")
    Rel(meetaction, todoist, "Создает задачи", "REST API, Bearer token")
    Rel(meetaction, gemini, "Классификация, извлечение, эмбеддинги", "HTTPS / gRPC")

    UpdateLayoutConfig($c4ShapeInRow="3", $c4BoundaryInRow="1")
```

---

## C4 Level 2 - Container

Показывает внутренние контейнеры системы, их роли и взаимодействия.

```mermaid
C4Container
    title MeetAction Agent - Containers

    Person(user, "Пользователь", "PM / тимлид / аналытик")

    System_Boundary(meetaction, "MeetAction Agent") {
        Container(ui, "Gradio UI", "Python / Gradio", "Загрузка файла, просмотр результатов, HITL кнопки, RAG-чат")
        Container(api, "FastAPI Backend", "Python / FastAPI", "REST-эндпоинты: /process, /confirm, /refine, /query")
        Container(orchestrator, "Agent Orchestrator", "LangGraph", "Граф состояний: ingest -> classify -> extract -> propose -> confirm -> create")
        Container(whisperx_svc, "Transcription Service", "Python / whisperx", "STT + диаризация. Вывод: текст с метками speaker_0/1")
        Container(pii, "PII Filter", "Python / spaCy + regex", "Маскировка PII перед отправкой в LLM")
        Container(redis, "Redis", "Redis 7 / Docker", "LangGraph checkpointer. Session state между шагами графа")
        Container(qdrant, "Qdrant", "Qdrant / Docker", "Векторное хранилище для RAG. Коллекция meeting_summaries")
    }

    System_Ext(todoist, "Todoist", "Таск-трекер")
    System_Ext(gemini, "Google AI API", "LLM + Embeddings")

    Rel(user, ui, "Загружает файл, взаимодействует с агентом", "HTTPS")
    Rel(ui, api, "API-вызовы", "HTTP / JSON")
    Rel(api, orchestrator, "Запуск и управление графом агента", "Python call")
    Rel(orchestrator, whisperx_svc, "Запуск транскрибации", "Python call")
    Rel(orchestrator, pii, "Фильтрация транскрипта", "Python call")
    Rel(orchestrator, redis, "Чтение / запись стейта", "Redis protocol")
    Rel(orchestrator, qdrant, "Индексирование саммари, поиск по прошлым встречам", "REST / gRPC")
    Rel(orchestrator, gemini, "Классификация, извлечение графа, генерация задач", "HTTPS")
    Rel(orchestrator, todoist, "POST /tasks (только после confirmed=True)", "REST API")

    UpdateLayoutConfig($c4ShapeInRow="4", $c4BoundaryInRow="1")
```

---

## C4 Level 3 - Component (Agent Orchestrator)

Внутреннее устройство ядра системы граф узлов LangGraph

```mermaid
C4Component
    title Agent Orchestrator - Components (LangGraph nodes)

    Container_Boundary(orchestrator, "Agent Orchestrator") {
        Component(ingest, "Ingestion Node", "Python", "Валидация файла, ffmpeg video->audio, проверка длины ≤ 90 мин")
        Component(transcribe, "Transcription Node", "whisperx", "STT + диаризация -> speaker_N: текст")
        Component(pii_node, "PII Filter Node", "spaCy + regex", "Маскировка PII, замена на [PHONE], [NAME] и т.д.")
        Component(inject_check, "Injection Check Node", "LLM-as-judge", "Проверка транскрипта на prompt injection перед передачей в основные узлы")
        Component(classifier, "Meeting Classifier Node", "LLM", "work_meeting | consultation При ошибке -> default: work_meeting")
        Component(extractor_work, "Work Extractor Node", "LLM + Pydantic", "Граф: speaker - action - object - deadline Retry x2 при невалидном JSON")
        Component(extractor_cons, "Consultation Extractor Node", "LLM + Pydantic", "Граф: topic - insight - recommendation Retry x2 при невалидном JSON")
        Component(proposer, "Proposer Node", "LLM + Pydantic", "Task / Insight Proposer. Формирует финальный JSON для UI")
        Component(hitl, "HITL Gate Node", "LangGraph interrupt", "interrupt_before(create). Агент ждет confirmed=True от пользователя")
        Component(refine, "Refine Node", "LLM", "Переработка задач / инсайтов с учетом фидбека. Max 2 итерации")
        Component(creator, "Creator Node", "Todoist Tool / save", "work: POST /tasks в Todoist consultation: сохранить инсайты в Qdrant")
        Component(indexer, "RAG Indexer Node", "Qdrant client", "Embedding summary -> Qdrant с payload: date, meeting_type")
    }

    Rel(ingest, transcribe, "audio path")
    Rel(transcribe, pii_node, "raw transcript + speaker labels")
    Rel(pii_node, inject_check, "clean transcript")
    Rel(inject_check, classifier, "ok")
    Rel(classifier, extractor_work, "work_meeting")
    Rel(classifier, extractor_cons, "consultation")
    Rel(extractor_work, proposer, "knowledge graph")
    Rel(extractor_cons, proposer, "knowledge graph")
    Rel(proposer, hitl, "proposed JSON")
    Rel(hitl, refine, "reject + feedback")
    Rel(hitl, creator, "confirmed=True")
    Rel(refine, hitl, "refined JSON")
    Rel(creator, indexer, "summary")
```

---

## Workflow / Graph Diagram

Пошаговое выполнение запроса со всеми ветками ошибок *См. `system-design.md`, раздел 3 - Основной workflow*

---

## Data Flow Diagram

Как данные проходят через систему, что хранится и что логируется.

```mermaid
flowchart LR
    subgraph INPUT["Вход"]
        F([audio / video файл])
    end

    subgraph PROC["Обработка (локально, до LLM)"]
        W[whisperx STT + диаризация]
        P[PII Filter маскировка]
    end

    subgraph LLM_LAYER["LLM (Google AI API)"]
        IC[Injection Check]
        CL[Classifier]
        EX[Extractor]
        PR[Proposer]
        RF[Refine]
        RG[RAG Generation]
    end

    subgraph STORAGE["Хранилище"]
        RD[(Redis Session State)]
        QD[(Qdrant RAG Index)]
    end

    subgraph OUTPUT["Выход"]
        UI[Gradio UI пользователь]
        TD[Todoist задачи]
    end

    subgraph LOGS["Observability"]
        LG[Loguru локальные логи]
        TC[Token Counter бюджет]
    end

    F --> W
    W -->|"raw transcript (не уходит в LLM)"| P
    P -->|"clean transcript [PII замаскированы]"| IC
    IC -->|ok| CL
    CL --> EX
    EX --> PR
    PR -->|proposed JSON| UI
    UI -->|feedback| RF
    RF --> UI
    UI -->|confirmed| TD

    EX -->|граф сущностей| RD
    PR -->|proposed output| RD
    TD -->|task_ids| RD

    PR -->|summary + meeting_type| QD
    QD -->|top-3 chunks| RG
    RG -->|ответ| UI

    RD -.->|"удаление transcript после завершения"| RD

    IC -->|токены, latency| LG
    CL -->|токены, latency| LG
    EX -->|токены, latency| LG
    PR -->|токены, latency| LG
    LG --> TC

    style INPUT fill:#e3f2fd
    style PROC fill:#f3e5f5
    style LLM_LAYER fill:#fff3e0
    style STORAGE fill:#e8f5e9
    style OUTPUT fill:#fce4ec
    style LOGS fill:#f5f5f5
```

**Что хранится:**
- **Redis** - session state (временно, удаляется после завершения сессии)
- **Qdrant** - summary встреч без PII, meeting_type, дата (постоянно)
- **Логи** - latency, токены, коды ошибок (локально, без текстов промптов)

**Что НЕ хранится:**
- Сырой транскрипт после завершения сессии
- Оригинальный аудио/видео файл после транскрибации
