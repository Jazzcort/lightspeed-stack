from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

import httpx
from pydantic_ai import AgentRunResult, RunContext
from pydantic_ai._agent_graph import GraphAgentState
from pydantic_ai.capabilities import WrapRunHandler
from pydantic_ai.direct import model_request
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    UserContent,
)
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider

from client import AsyncOgxClientHolder
from log import get_logger
from models.common.moderation import (
    ShieldModerationBlocked,
    ShieldModerationResult,
)
from models.config import GraniteGuardianConfig
from pydantic_ai_lightspeed.capabilities.base import AbstractSafetyCapability
from pydantic_ai_lightspeed.capabilities.granite_guardian.utils import (
    build_guardian_block,
    pass_confidence_threhold,
)
from utils.conversations import append_turn_to_conversation

MODEL_NAME = "ibm-granite/granite-guardian-4.1-8b"

logger = get_logger(__name__)


def _extract_message_str_from_user_content(user_content: Sequence[UserContent]) -> str:
    """Extract and combine all text content into a string from a UserContent sequence.

    Parameters:
        user_content: A sequence of user content items to extract text from.

    Returns:
        A single string with all text content joined by newlines.
    """
    str_arr: list[str] = []
    for c in user_content:
        match c:
            case str() as s:
                str_arr.append(s)
            case TextContent(content=c):
                str_arr.append(c)

    return "\n".join(str_arr)


def _message_to_str(message: Optional[str | Sequence[UserContent]]) -> str:
    """Convert a user message (string, content sequence, or None) to plain text.

    Parameters:
        message: The user input as a string, sequence of user content, or None.

    Returns:
        A plain-text representation of the message, or an empty string for None.
    """
    match message:
        case str() as s:
            return s
        case Sequence() as seq:
            return _extract_message_str_from_user_content(seq)
        case None:
            return ""


def _extract_conversation_id(model: Model) -> Optional[str]:
    """Extract the Llama Stack conversation ID from the agent's model settings.

    The main agent's model is built with ``conversation`` in its
    ``extra_body`` model settings (see ``OgxResponsesModel.from_ogx_client``).
    This pulls it back out so the capability can persist the rejected turn
    to the same conversation.

    Parameters:
        model: The model bound to the current agent run (``ctx.model``).

    Returns:
        The conversation ID, or None if the model has no such setting
        (e.g. when used outside a Llama Stack-backed agent).
    """
    extra_body = (model.settings or {}).get("extra_body")
    if not isinstance(extra_body, dict):
        return None

    conversation_id = extra_body.get("conversation")
    return conversation_id if isinstance(conversation_id, str) else None


@dataclass
class GraniteGuardian(AbstractSafetyCapability):

    config: GraniteGuardianConfig
    _model: Model = field(init=False)

    def __post_init__(self) -> None:
        http_client = httpx.AsyncClient(verify=self.config.verify_ssl)

        provider = OpenAIProvider(
            base_url=self.config.url,
            api_key=(
                self.config.api_key.get_secret_value()
                if self.config.api_key is not None
                else None
            ),
            http_client=http_client,
        )

        self._model = OpenAIChatModel(MODEL_NAME, provider=provider)

    async def wrap_run(
        self, ctx: RunContext, *, handler: WrapRunHandler
    ) -> AgentRunResult:
        """Run the question validity check before delegating to the main agent.

        Sends the user prompt to the validity model for classification.
        If the question is allowed, the handler proceeds normally.
        Otherwise, a rejection response is returned and the main agent
        is bypassed.

        Parameters:
            ctx: The run context containing the user prompt and usage tracker.
            handler: The handler that invokes the main agent run.

        Returns:
            The agent run result, either from the main agent or a rejection.
        """
        user_prompt = _message_to_str(ctx.prompt)

        guardian_blocks = [
            (
                risk.name,
                build_guardian_block(risk.description, think=risk.enable_thinking),
                risk.threshold,
            )
            for risk in self.config.risks
            if risk.enabled
        ]

        for guardian_block in guardian_blocks:
            risk_name, block, threhold = guardian_block

            result = await model_request(
                model=self._model,
                messages=[
                    ModelRequest.user_text_prompt(user_prompt, instructions=block)
                ],
                model_settings=OpenAIChatModelSettings(
                    openai_logprobs=True, openai_top_logprobs=5
                ),
            )
            ctx.usage.incr(result.usage)

            assert (
                result.provider_details
            ), "No provider_details provided from granite guardian's response"
            assert result.provider_details[
                "logprobs"
            ], "No logprobs field in provider_details"

            if not pass_confidence_threhold(
                threhold, result.provider_details["logprobs"]
            ):
                block_message = f"Blocked by Granite Guardian guardrail because the user input didn't pass {risk_name} content check"

                state = GraphAgentState(
                    usage=ctx.usage,
                    message_history=[
                        ModelRequest.user_text_prompt(user_prompt),
                        ModelResponse(
                            [TextPart(block_message)],
                            finish_reason="stop",
                        ),
                    ],
                )
                conversation_id = _extract_conversation_id(ctx.model)
                if conversation_id is not None:
                    await append_turn_to_conversation(
                        AsyncOgxClientHolder().get_client(),
                        conversation_id,
                        user_prompt,
                        block_message,
                    )
                else:
                    logger.warning(
                        "Unable to determine conversation ID from model settings; "
                        "skipping v1/conversation persistence for rejected question."
                    )

                return AgentRunResult(output=block_message, _state=state)

        return await handler()  # proceed with the real run

    async def run(self, input_text: str) -> ShieldModerationResult:
        return ShieldModerationBlocked(
            message="hahahaah",
            moderation_id=f"modr-{uuid4()}",
        )
