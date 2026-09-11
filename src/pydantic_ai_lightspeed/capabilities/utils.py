"""Shared utility functions for safety capabilities."""

from collections.abc import Sequence
from typing import Optional

from openai.types.responses import ResponseOutputMessage
from pydantic_ai.messages import TextContent, UserContent
from pydantic_ai.models import Model

from log import get_logger

logger = get_logger(__name__)


def extract_message_str_from_user_content(
    user_content: Sequence[UserContent],
) -> str:
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


def message_to_str(message: Optional[str | Sequence[UserContent]]) -> str:
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
            return extract_message_str_from_user_content(seq)
        case None:
            return ""


def extract_conversation_id(model: Model) -> Optional[str]:
    """Extract the conversation ID from the agent's model settings.

    The main agent's model is built with ``conversation`` in its
    ``extra_body`` model settings (see ``OgxResponsesModel.from_ogx_client``).
    This pulls it back out so the capability can persist the rejected turn
    to the same conversation.

    Parameters:
        model: The model bound to the current agent run (``ctx.model``).

    Returns:
        The conversation ID, or None if the model has no such setting.
    """
    extra_body = (model.settings or {}).get("extra_body")
    if not isinstance(extra_body, dict):
        return None

    conversation_id = extra_body.get("conversation")
    return conversation_id if isinstance(conversation_id, str) else None


def extract_last_message_items(model: Model) -> Optional[list[ResponseOutputMessage]]:
    """Extract the output items captured from the last OGX response.

    Retrieves the ``last_output_items`` attribute from the model, which
    contains the structured output items (including their IDs) that OGX
    persisted during inference. Used by output guardrails to identify and
    replace conversation items when a violation is detected.

    Parameters:
        model: The pydantic-ai model, expected to be an ``OgxResponsesModel``
            that captures output items from OGX responses.

    Returns:
        The list of output message items, or ``None`` when the model did not
        capture any (e.g. a non-OGX model or a test double).
    """
    last_output_items = getattr(model, "last_output_items", None)

    if not isinstance(last_output_items, list):
        return None

    return last_output_items
