# Spec: Serving / Config

## Запуск

```bash
docker-compose up --build
```

### Сервисы в docker-compose

| Сервис | Image | Порт | Назначение |
| :--- | :--- | :--- | :--- |
| `app` | `./Dockerfile` (Python 3.11) | 7007 | FastAPI + Gradio UI |
| `redis` | `redis:7-alpine` | 6379 | LangGraph checkpointer |
| `qdrant` | `qdrant/qdrant:latest` | 6333 / 6334 | Векторное хранилище RAG |

### Минимальные требования

| Ресурс | Минимум | Рекомендуется |
| :--- | :--- | :--- |
| CPU | 4 ядра | 8 ядер |
| RAM | 8 GB | 16 GB |
| Диск | 15 GB | 25 GB |
| GPU | не требуется | не требуется |

> whisperx запускается на CPU. Для `whisper-base` - ~3-5 мин на 90-минутное аудио

---

## Конфигурация

Все настройки через `.env` файл (`.env.example` в репозитории):

```bash
# API Keys
LLM_BASE_URL=...
LLM_API_KEY=...
TODOIST_API_TOKEN=...

# Redis
REDIS_URL=redis://redis:6379

# Qdrant
QDRANT_HOST=qdrant
QDRANT_PORT=6333

# LLM
LLM_MODEL=google/gemini-2.5-flash-lite-preview-09-2025
EMBEDDING_MODEL=google/gemini-embedding-001
LLM_TIMEOUT=60
LLM_RETRY_COUNT=2

# Agent
MAX_REFINEMENT_COUNT=2
SESSION_TTL_SECONDS=1800
CLASSIFIER_DEFAULT=work_meeting

# Whisperx
WHISPER_MODEL_SIZE=base
WHISPER_DEVICE=cpu
WHISPER_COMPUTE_TYPE=int8

# Limits
MAX_AUDIO_DURATION_SECONDS=5400  # 90 min
```

---

## Версии моделей

| Компонент | Версия |
| :--- | :--- |
| LLM | `google/gemini-2.5-flash-lite-preview-09-2025` |
| Embeddings | `google/gemini-embedding-001` |
| Whisperx | `faster-whisper` backend, модель `base` |
| Диаризация | `pyannote/speaker-diarization-3.1` |

> Модели зафиксированы строками в `.env`

---

## Локальный запуск без Docker (разработка)

```bash
# Зависимости
pip install -r requirements.txt

# Redis и Qdrant все равно через Docker
docker-compose up redis qdrant -d

# Приложение
uvicorn app.main:app --reload --port 8000 &
python -m gradio app/ui.py
```
