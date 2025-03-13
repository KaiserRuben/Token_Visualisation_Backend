from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any, Literal
from enum import Enum

# Job status enum
class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

# Attribution methods from Inseq
class AttributionMethod(str, Enum):
    """Available attribution methods in Inseq"""
    INPUT_X_GRADIENT = "input_x_gradient"
    SALIENCY = "saliency"
    INTEGRATED_GRADIENTS = "integrated_gradients"
    DEEP_LIFT = "deep_lift"
    GRADIENT_SHAP = "gradient_shap"
    LAYER_INTEGRATED_GRADIENTS = "layer_integrated_gradients"
    LAYER_GRADIENT_X_ACTIVATION = "layer_gradient_x_activation"
    LAYER_DEEP_LIFT = "layer_deep_lift"
    ATTENTION = "attention"
    OCCLUSION = "occlusion"
    LIME = "lime"
    VALUE_ZEROING = "value_zeroing"
    REAGENT = "reagent"
    DISCRETIZED_INTEGRATED_GRADIENTS = "discretized_integrated_gradients"
    SEQUENTIAL_INTEGRATED_GRADIENTS = "sequential_integrated_gradients"

# Available step score functions
class StepScoreFunction(str, Enum):
    """Available step score functions in Inseq"""
    LOGIT = "logit"
    PROBABILITY = "probability"
    ENTROPY = "entropy"
    CROSS_ENTROPY = "crossentropy"
    PERPLEXITY = "perplexity"
    CONTRAST_LOGITS = "contrast_logits"
    CONTRAST_PROB = "contrast_prob"
    CONTRAST_LOGITS_DIFF = "contrast_logits_diff"
    CONTRAST_PROB_DIFF = "contrast_prob_diff"
    PCXMI = "pcxmi"
    KL_DIVERGENCE = "kl_divergence"
    IN_CONTEXT_PVI = "in_context_pvi"
    MC_DROPOUT_PROB_AVG = "mc_dropout_prob_avg"
    TOP_P_SIZE = "top_p_size"

# Request model for job submission
class AttributionJobRequest(BaseModel):
    """
    Model for submitting a new attribution job.
    """
    model: str = Field(
        default="meta-llama/Llama-3.2-3B",
        description="The model to use for attribution (HuggingFace model name or path)"
    )
    method: AttributionMethod = Field(
        default=AttributionMethod.INPUT_X_GRADIENT,
        description="The attribution method to use"
    )
    input_text: str = Field(
        ...,
        description="The input text to attribute"
    )
    n_steps: int = Field(
        default=20,
        description="Number of steps for methods that require it (integrated_gradients, deep_lift, gradient_shap)"
    )
    step_scores: List[StepScoreFunction] = Field(
        default=[StepScoreFunction.PROBABILITY],
        description="Step scores to compute alongside attribution"
    )
    force_cpu: bool = Field(
        default=False,
        description="Force CPU usage for all operations"
    )

# Response model for job submission
class AttributionJobResponse(BaseModel):
    """
    Response model for job submission.
    """
    job_id: str = Field(
        ...,
        description="The unique ID of the attribution job"
    )
    status: JobStatus = Field(
        ...,
        description="The current status of the job"
    )

# Job result response model
class AttributionJobResultResponse(BaseModel):
    """
    Response model for job status and results.
    """
    job_id: str = Field(
        ...,
        description="The unique ID of the attribution job"
    )
    status: JobStatus = Field(
        ...,
        description="The current status of the job"
    )
    output_file: Optional[str] = Field(
        default=None,
        description="Path to the output file containing the attribution results"
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the job failed"
    )

# Token model for visualization data
class Token(BaseModel):
    """
    Model for a token with ID.
    """
    token: str
    token_id: int

# These models match the Inseq data classes but are simplified for API responses
class SequenceAttribution(BaseModel):
    """
    Model for sequence attribution results.
    """
    source: List[Token]
    target: List[Token]
    source_attributions: Optional[List[List[float]]] = None
    target_attributions: Optional[List[List[float]]] = None
    step_scores: Optional[Dict[str, List[float]]] = None

class AttributionResult(BaseModel):
    """
    Model for complete attribution results.
    """
    sequence_attributions: List[SequenceAttribution]
    info: Dict[str, Any] = {}