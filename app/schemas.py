from pydantic import BaseModel, Field, EmailStr, field_validator, field_serializer, ConfigDict
from typing import Optional, List, Dict, Any
from enum import Enum


class CreateSNLI(BaseModel):
    premise: str = Field(..., description="The premise of the sentence")
    hypothesis: str = Field(..., description="The hypothesis of the sentence")


class SNLIClass(int, Enum):
    entailment = 0
    neutral = 1
    contradiction = 2


class SNLIResponse(BaseModel):
    premise: str
    hypothesis: str
    # label: SNLIClass  # predicted label
    label: str
    confidence: float


# class SNLIResponse(BaseModel):
#     premise: str
#     hypothesis: str
#     label: SNLIClass
#     confidence: float

#     class Config:
#         use_enum_values = False  # Important: don't serialize enum as value, but as name
#         json_encoders = {
#             SNLIClass: lambda v: v.name
#         }