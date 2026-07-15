from pydantic import BaseModel, ConfigDict, Field


class ProtocolSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_rounds: int = Field(default=3, ge=1, le=3)
    conflict_threshold: float = Field(default=0.15, gt=0.0, lt=1.0)
    conformal_alpha: float = Field(default=0.10, gt=0.0, lt=1.0)
    calibration_bins: int = Field(default=15, ge=2)
    minimum_evidence_relevance: float = Field(default=0.05, ge=0.0, le=1.0)
    support_adjustment: float = Field(default=0.04, ge=0.0, le=0.25)
    challenge_adjustment: float = Field(default=0.05, ge=0.0, le=0.25)
    equipoise_margin: float = Field(default=0.03, ge=0.0, le=0.25)


class EvaluationSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    bootstrap_iterations: int = Field(default=1000, ge=100)
    random_seed: int = 1729
    stochastic_runs: int = Field(default=5, ge=1)
    familywise_alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    pairwise_comparisons: int = Field(default=5, ge=1)

    @property
    def adjusted_alpha(self) -> float:
        return self.familywise_alpha / self.pairwise_comparisons
