"""Strict schemas for extracted intent and the final Unity-facing JSON."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class AssetMention(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_id: str
    evidence: str  # Exact words copied from the user's request.


class GameIntent(BaseModel):
    """Groq's output. All fields are required for Groq strict JSON Schema mode."""
    model_config = ConfigDict(extra="forbid")
    status: Literal["supported", "needs_clarification", "unsupported"]
    game_template: str | None
    environment: str | None
    assets: list[AssetMention]
    object_count: int | None
    duration_seconds: int | None
    difficulty: str | None
    max_hints: int | None
    score_per_object: int | None
    scoring_mode: Literal["fixed", "increasing"] | None
    reward_requested: bool | None
    reward_type: str | None
    unsupported_elements: list[str]
    clarification_questions: list[str]

    @field_validator("reward_requested", mode="before")
    @classmethod
    def normalize_reward_flag(cls, value):
        return False if value is None else value


class GameObject(BaseModel):
    model_config = ConfigDict(extra="forbid")
    asset_id: str
    score: int = Field(ge=10, le=100)


class Hints(BaseModel):
    model_config = ConfigDict(extra="forbid")
    enabled: bool
    max_hints: int = Field(ge=0, le=5)

    @model_validator(mode="after")
    def consistent(self):
        if self.enabled != (self.max_hints > 0):
            raise ValueError("hints.enabled must match max_hints > 0")
        return self


class WinCondition(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["find_all_objects"]


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["card"]
    reward_id: Literal["demo_card_01"]


class GameConfig(BaseModel):
    """The exact Unity-facing output structure in the assessment example."""
    model_config = ConfigDict(extra="forbid")
    game_template: Literal["hidden_objects", "treasure_hunt"]
    environment: Literal["bedroom", "v_shop", "park"]
    duration_seconds: int = Field(ge=30, le=180)
    difficulty: Literal["easy", "medium", "hard"]
    objects: list[GameObject] = Field(min_length=1, max_length=10)
    hints: Hints
    win_condition: WinCondition
    reward: Reward | None = None

    @model_validator(mode="after")
    def distinct_assets(self):
        ids = [obj.asset_id for obj in self.objects]
        if len(ids) != len(set(ids)):
            raise ValueError("object asset IDs must be unique in this POC")
        return self
