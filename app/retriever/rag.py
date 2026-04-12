from loguru import logger
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    VectorParams,
)

from app.config import settings
from app.llm.client import get_embedding
from app.schemas.common import EMeetingType

COLLECTION_NAME = "meeting_summaries"
SCORE_THRESHOLD = 0.75
TOP_K = 3
EMBEDDING_DIM = 3072


def _get_client() -> AsyncQdrantClient:
    return AsyncQdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


async def ensure_collection() -> None:
    client = _get_client()
    collections = await client.get_collections()
    names = [c.name for c in collections.collections]

    if COLLECTION_NAME in names:
        info = await client.get_collection(COLLECTION_NAME)
        actual_dim = info.config.params.vectors.size
        if actual_dim != EMBEDDING_DIM:
            logger.warning(
                "Qdrant collection dim mismatch, recreating | expected={} actual={}",
                EMBEDDING_DIM,
                actual_dim,
            )
            await client.delete_collection(COLLECTION_NAME)
            names = []

    if COLLECTION_NAME not in names:
        await client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE),
        )
        logger.info(
            "Qdrant collection created | name={} dim={}", COLLECTION_NAME, EMBEDDING_DIM
        )


async def index_meeting(
    session_id: str,
    date: str,
    meeting_type: str,
    participants_count: int,
    summary: str,
) -> None:
    client = _get_client()
    await ensure_collection()

    embedding = await get_embedding(summary)

    import uuid

    point_id = str(uuid.uuid4())

    from qdrant_client.models import PointStruct

    await client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            PointStruct(
                id=point_id,
                vector=embedding,
                payload={
                    "session_id": session_id,
                    "date": date,
                    "meeting_type": meeting_type,
                    "participants_count": participants_count,
                    "summary": summary,
                },
            )
        ],
    )
    logger.info("RAG indexed | session_id={} meeting_type={}", session_id, meeting_type)


async def search_meetings(
    query: str,
    meeting_type: EMeetingType | None = None,
) -> list[dict]:
    client = _get_client()
    embedding = await get_embedding(query)

    query_filter = None
    if meeting_type:
        query_filter = Filter(
            must=[
                FieldCondition(
                    key="meeting_type", match=MatchValue(value=meeting_type.value)
                )
            ]
        )

    response = await client.query_points(
        collection_name=COLLECTION_NAME,
        query=embedding,
        limit=TOP_K,
        score_threshold=SCORE_THRESHOLD,
        query_filter=query_filter,
    )

    logger.debug(
        "RAG search | query_len={} results={}", len(query), len(response.points)
    )
    return [r.payload for r in response.points]


async def generate_rag_answer(query: str, contexts: list[dict]) -> str:
    from app.llm.client import llm_call_with_retry

    if not contexts:
        return "Нет релевантных данных по прошлым встречам"

    context_text = "\n\n".join(
        f"[{c.get('date', '?')} | {c.get('meeting_type', '?')}]\n{c.get('summary', '')}"
        for c in contexts
    )

    messages = [
        {
            "role": "system",
            "content": "You are a meeting assistant. Answer the user's question based on past meeting summaries provided in the context.",
        },
        {
            "role": "user",
            "content": f"<context>\n{context_text}\n</context>\n\nQuestion: {query}",
        },
    ]

    content, _, _ = await llm_call_with_retry(
        messages=messages,
        temperature=0.3,
        node="rag_generation",
        session_id="rag",
    )
    return content
