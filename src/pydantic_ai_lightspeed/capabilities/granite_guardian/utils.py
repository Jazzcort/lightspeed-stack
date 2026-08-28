from enum import Enum, StrEnum, auto
from math import exp

from openai.types.chat.chat_completion_token_logprob import ChatCompletionTokenLogprob

from log import get_logger


class ParsingStatus(Enum):
    BEFORE_THINK = auto()
    INSIDE_THINK = auto()
    POST_THINK = auto()
    INSIDE_SCORE = auto()
    POST_SCORE = auto()


class Token(StrEnum):
    SAFE = "no"
    RISKY = "yes"


THINK_START_TAG = "<think>"
THINK_END_TAG = "</think>"
SCORE_START_TAG = "<score>"
SCORE_END_TAG = "</score>"


_JUDGE_PREAMBLE_NOTHINK = (
    "<guardian><no-think>As a judge agent, assess whether the provided text "
    "meets the given judging criteria using all available information, "
    "including conversations, documents, and tools. Provide your score "
    "immediately without explanation. Output empty <think>\\n</think> tags "
    "followed by your score in <score></score> tags."
)

_JUDGE_PREAMBLE_THINK = (
    "<guardian><think>As a judge agent, carefully analyze whether the provided "
    "text meets the given judging criteria. Use all available information, "
    "including conversations, documents, and tools. Reason through the "
    "evidence step by step before providing your score. Wrap your reasoning "
    "in <think></think> tags, then provide your score in <score></score> tags."
)

_SCORING_SCHEMA_PROMPT_ONLY = (
    "If the last user's text meets the criteria, return 'yes'; otherwise, return 'no'."
)

logger = get_logger(__name__)


def build_guardian_block(criteria: str, *, think: bool = False) -> str:
    """Build the guardian evaluation block sent as a second user message."""
    preamble = _JUDGE_PREAMBLE_THINK if think else _JUDGE_PREAMBLE_NOTHINK
    return (
        f"{preamble}\n\n"
        f"### Criteria: {criteria}\n\n"
        f"### Scoring Schema: {_SCORING_SCHEMA_PROMPT_ONLY}"
    )


def _search_tag(tag: str, buffer: str):
    if not buffer.strip().startswith("<"):
        return "", False
    elif tag in buffer:
        return "", True
    else:
        return buffer, False


def _clean_up_candidates(candidates: list[ChatCompletionTokenLogprob]):
    buffer = ""
    while len(candidates) != 0:
        cur_candidate = candidates.pop()
        buffer = cur_candidate.token + buffer
        if SCORE_END_TAG in buffer:
            return


def _extract_tokens_inside_score_tag(
    logprobs: list[object],
) -> ChatCompletionTokenLogprob:
    cur_buffer = ""
    cur_status = ParsingStatus.BEFORE_THINK
    candidates: list[ChatCompletionTokenLogprob] = []

    for _logprob in logprobs:
        logprob = ChatCompletionTokenLogprob.model_validate(_logprob)
        cur_buffer += logprob.token

        match cur_status:
            case ParsingStatus.BEFORE_THINK:
                cur_buffer, found_tag = _search_tag(THINK_START_TAG, cur_buffer)
                if found_tag:
                    cur_status = ParsingStatus.INSIDE_THINK
            case ParsingStatus.INSIDE_THINK:
                cur_buffer, found_tag = _search_tag(THINK_END_TAG, cur_buffer)
                if found_tag:
                    cur_status = ParsingStatus.POST_THINK
            case ParsingStatus.POST_THINK:
                cur_buffer, found_tag = _search_tag(SCORE_START_TAG, cur_buffer)
                if found_tag:
                    cur_status = ParsingStatus.INSIDE_SCORE
            case ParsingStatus.INSIDE_SCORE:
                cur_buffer, found_tag = _search_tag(SCORE_END_TAG, cur_buffer)
                candidates.append(logprob)
                if found_tag:
                    cur_status = ParsingStatus.POST_SCORE
                    _clean_up_candidates(candidates)

            case ParsingStatus.POST_SCORE:
                candidates_num = len(candidates)
                if candidates_num == 0:
                    raise ValueError("No token found inside score tag")
                elif candidates_num > 1:
                    raise ValueError("More than one token found inside score tag")

                return candidates[0]

    raise ValueError("Model did not generate the required format")


def _get_risky_probabilities(token: ChatCompletionTokenLogprob):
    safe_prob, risky_prob = 0, 0
    for candidate_token in token.top_logprobs:
        match candidate_token.token.strip().lower():
            case Token.SAFE:
                safe_prob += exp(candidate_token.logprob)
            case Token.RISKY:
                risky_prob += exp(candidate_token.logprob)
    total_prob = safe_prob + risky_prob

    if total_prob == 0:
        raise ValueError("Logprob underflow")

    return risky_prob / total_prob


def pass_confidence_threhold(threhold: float, logprobs: list[object]):
    token_for_confidence_threhold = _extract_tokens_inside_score_tag(logprobs)
    p_risky = _get_risky_probabilities(token_for_confidence_threhold)

    return p_risky < threhold
