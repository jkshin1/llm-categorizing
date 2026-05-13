from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Stage1Result(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    major_job: str = Field(default="", alias="중직무")
    sub_job: str = Field(default="", alias="소직무")
    confidence: float = 0.0
    reason: str = ""
    needs_review: bool = True

    @field_validator("confidence", mode="before")
    @classmethod
    def parse_confidence(cls, value: object) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, parsed))


class FinalClassificationResult(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    major_job: str = Field(default="", alias="중직무")
    sub_job: str = Field(default="", alias="소직무")
    device: str = Field(default="", alias="Device")
    unit_job: str = Field(default="", alias="단위 직무")
    detail_job_1: str = Field(default="", alias="세부 직무1")
    detail_job_2: str = Field(default="", alias="세부 직무2")
    confidence: float = 0.0
    reason: str = ""
    needs_review: bool = True

    @field_validator("confidence", mode="before")
    @classmethod
    def parse_confidence(cls, value: object) -> float:
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return 0.0
        return max(0.0, min(1.0, parsed))
