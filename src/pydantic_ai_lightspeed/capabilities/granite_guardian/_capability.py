"""Granite Guardian safety capability for input/output guardrail moderation."""

from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

import httpx
from pydantic_ai import AgentRunResult, RunContext
from pydantic_ai._agent_graph import GraphAgentState
from pydantic_ai.capabilities import WrapRunHandler
from pydantic_ai.direct import model_request
from pydantic_ai.exceptions import UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
)
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RequestUsage

from client.ogx import AsyncOgxClientHolder
from log import get_logger
from models.common.moderation import (
    ShieldModerationBlocked,
    ShieldModerationPassed,
    ShieldModerationResult,
)
from models.config import GraniteGuardianConfig, GuardrailPoint, RiskDefinition
from pydantic_ai_lightspeed.capabilities.base import AbstractSafetyCapability
from pydantic_ai_lightspeed.capabilities.granite_guardian.utils import (
    build_guardian_block,
    is_safe,
)
from pydantic_ai_lightspeed.capabilities.utils import (
    extract_conversation_id,
    message_to_str,
)
from utils.conversations import append_turn_to_conversation

type Guardrail = tuple[str, str, float, str]

MODEL_NAME = "ibm-granite/granite-guardian-4.1-8b"

logger = get_logger(__name__)


async def _run_risk_check(
    prompt: str,
    model: Model,
    guardrails: list[Guardrail],
) -> tuple[Optional[str], RequestUsage]:
    """Evaluate the prompt against each guardrail sequentially.

    Returns on the first violation found, or None if all checks pass.

    Parameters:
        prompt: The text to evaluate.
        model: The Granite Guardian model to use for evaluation.
        guardrails: Ordered list of guardrail tuples to check.

    Returns:
        A tuple of (violation_message, token_usage). violation_message is
        None when all checks pass.

    Raises:
        UnexpectedModelBehavior: When the model response is missing
            provider_details or logprobs.
    """
    token_usage = RequestUsage()
    for guardrail in guardrails:
        _risk_name, block, threshold, violation_message = guardrail

        result = await model_request(
            model=model,
            messages=[ModelRequest.user_text_prompt(prompt, instructions=block)],
            model_settings=OpenAIChatModelSettings(
                openai_logprobs=True, openai_top_logprobs=5
            ),
        )

        token_usage.incr(result.usage)

        if not result.provider_details:
            raise UnexpectedModelBehavior(
                "No provider_details provided from granite guardian's response"
            )
        if not result.provider_details["logprobs"]:
            raise UnexpectedModelBehavior("No logprobs field in provider_details")

        if not is_safe(threshold, result.provider_details["logprobs"]):

            return violation_message, token_usage

    return None, token_usage


def _filter_guardrails(
    risks: list[RiskDefinition], point: GuardrailPoint
) -> list[Guardrail]:
    """Filter risk definitions to guardrail tuples for a given guardrail point.

    Parameters:
        risks: All configured risk definitions.
        point: The guardrail point to filter by (INPUT, OUTPUT, or TOOL).

    Returns:
        A list of guardrail tuples for enabled risks matching the point.
    """
    return [
        (
            risk.name,
            build_guardian_block(risk.description, think=risk.enable_thinking),
            risk.threshold,
            risk.violation_message,
        )
        for risk in risks
        if risk.enabled and point in risk.points
    ]


@dataclass
class GraniteGuardian(AbstractSafetyCapability):
    """Safety capability using Granite Guardian for risk-based moderation.

    Uses Granite Guardian's logprob-based scoring to evaluate user input
    against configured risk categories. When used as a pydantic-ai capability,
    ``wrap_run`` applies input guardrails. The ``run`` method provides a
    standalone shield interface for use outside the agent lifecycle.

    Attributes:
        config: Granite Guardian configuration with risks and connection details.
        run_moderation_guardrail_point: The guardrail point used by the
            standalone ``run`` method.
    """

    config: GraniteGuardianConfig
    run_moderation_guardrail_point: GuardrailPoint = GuardrailPoint.INPUT
    _model: Model = field(init=False)

    def __post_init__(self) -> None:
        """Initialize the Granite Guardian model with the configured provider."""
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
        """Apply input guardrails before the agent run.

        Evaluates the user prompt against all INPUT-point risks. If any risk
        is violated, the run is short-circuited with a rejection message.
        Otherwise, the handler is called to proceed with the real run.

        Parameters:
            ctx: The run context containing the user prompt and usage tracker.
            handler: The handler to call if the input passes all guardrails.

        Returns:
            The agent run result, either a rejection or the handler's result.
        """
        user_prompt = message_to_str(ctx.prompt)

        input_guardrails = _filter_guardrails(self.config.risks, GuardrailPoint.INPUT)
        violation_message, token_usage = await _run_risk_check(
            user_prompt, self._model, input_guardrails
        )

        ctx.usage.incr(token_usage)

        if violation_message is not None:
            state = GraphAgentState(
                usage=ctx.usage,
                message_history=[
                    ModelRequest.user_text_prompt(user_prompt),
                    ModelResponse(
                        [TextPart(violation_message)],
                        finish_reason="stop",
                    ),
                ],
            )

            conversation_id = extract_conversation_id(ctx.model)
            if conversation_id is not None:
                await append_turn_to_conversation(
                    AsyncOgxClientHolder().get_client(),
                    conversation_id,
                    user_prompt,
                    violation_message,
                )
            else:
                logger.warning(
                    "Unable to determine conversation ID from model settings; "
                    "skipping v1/conversation persistence for rejected question."
                )

            return AgentRunResult(output=violation_message, _state=state)

        agent_result = await handler()  # proceed with the real run

        return agent_result

    async def run(self, input_text: str) -> ShieldModerationResult:
        """Run standalone shield moderation on the given text.

        Uses ``run_moderation_guardrail_point`` to filter which risks apply.

        Parameters:
            input_text: The text to evaluate.

        Returns:
            A blocked result with the violation message, or a passed result.
        """
        filtered_guardrails = _filter_guardrails(
            self.config.risks, self.run_moderation_guardrail_point
        )

        violation_message, token_usage = await _run_risk_check(
            input_text, self._model, filtered_guardrails
        )

        if violation_message is not None:
            return ShieldModerationBlocked(
                message=violation_message, moderation_id=f"modr-{uuid4()}"
            )

        return ShieldModerationPassed()
