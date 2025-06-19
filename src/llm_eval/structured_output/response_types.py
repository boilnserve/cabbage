from pydantic import BaseModel
from typing import Literal, Union

#----------Response Classes------------
class ReasoningAnswer(BaseModel):
    reasoning: str
    answer: str
class ReasoningAnswerLetter(BaseModel):
    reasoning: str
    answer: Literal["A", "B", "C", "D", "E"]
class ReasoningLongAnswer(BaseModel):
    reasoning: str
    answer: str

ParsedResponse = Union[ReasoningAnswer, ReasoningAnswerLetter, ReasoningLongAnswer]

RESPONSE_MODEL_MAP = {
        "open_ended_short": ReasoningAnswer,
        "multiple_choice": ReasoningAnswerLetter,
        "open_ended_long": ReasoningLongAnswer,
    }