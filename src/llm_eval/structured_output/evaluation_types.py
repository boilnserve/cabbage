from enum import Enum
from pydantic import BaseModel

# class syntax
class Correctness(Enum):
    INCORRECT = '1'
    PARTIALLY = '2'
    MOSTLY = '3'
    LARGELY = '4'
    COMPLETE = '5'
    
class Specificity(Enum):
    GENERAL = '1'
    BASIC = '2'
    COMPARABLE = '3'
    IN_DEPTH = '4'
    SUPERIOR = '5'
    
class Conciseness(Enum):
    VERBOSE = '1'
    OVERLY_DETAILED = '2'
    BALANCED = '3'
    CONCISE = '4'
    EXCEPTIONALLY_CONCISE = '5'

class ProceduralAccuracy(Enum):
    LARGELY_INCORRECT = '1'
    PARTIALLY_CORRECT = '2'
    MOSTLY_CORRECT = '3'
    NEARLY_COMPLETE = '4'
    FULLY_ACCURATE = '5'

class ProceduralFlow(Enum):
    EXTREMELY_DISORGANIZED = '1'
    POORLY_ORGANIZED = '2'
    GENERALLY_LOGICAL = '3'
    WELL_ORGANIZED = '4'
    VERY_CLEAR = '5'
    
class EvaluationStep(BaseModel):
    step_analysis: str

class Scores(BaseModel):
    correctness_score: Correctness
    specificity_score: Specificity
    conciseness_score: Conciseness

class ScoresProcedural(BaseModel):
    procedural_accuracy_score: ProceduralAccuracy
    procedural_flow_score: ProceduralFlow
    conciseness_score: Conciseness
    
class EvaluationProcess(BaseModel):
    evaluation_steps: list[EvaluationStep]
    scores: Scores
    final_review: str
    
class EvaluationProcessProcedural(BaseModel):
    evaluation_steps: list[EvaluationStep]
    scores: ScoresProcedural
    final_review: str