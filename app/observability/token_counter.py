from loguru import logger

from app.config import settings


class TokenCounter:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.input_tokens = 0
        self.output_tokens = 0

    def add(self, input_tokens: int, output_tokens: int, node: str) -> None:
        self.input_tokens += input_tokens
        self.output_tokens += output_tokens

        logger.info(
            "LLM call | session_id={} node={} tokens_in={} tokens_out={}",
            self.session_id,
            node,
            input_tokens,
            output_tokens,
        )

        if self.input_tokens > settings.TOKEN_BUDGET_WARNING:
            logger.warning(
                "Token budget warning | session_id={} total_input={}",
                self.session_id,
                self.input_tokens,
            )
