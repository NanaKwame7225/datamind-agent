"""
DataMind Agent v2 — User & Auth Models
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime
from enum import Enum


class PlanTier(str, Enum):
    free       = "free"        # 10 analyses/month
    starter    = "starter"     # 100 analyses/month — $29/mo
    pro        = "pro"         # 500 analyses/month — $79/mo
    enterprise = "enterprise"  # unlimited — $199/mo


class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8)
    full_name: str = Field(min_length=2)
    company: Optional[str] = None
    industry: Optional[str] = None


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    id: str
    email: str
    full_name: str
    company: Optional[str]
    industry: Optional[str]
    plan: PlanTier
    analyses_used: int
    analyses_limit: int
    created_at: datetime
    is_active: bool


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserOut


class PasswordReset(BaseModel):
    email: EmailStr


class PasswordResetConfirm(BaseModel):
    token: str
    new_password: str = Field(min_length=8)


PLAN_LIMITS = {
    PlanTier.free:       {"analyses": 10,    "price": 0},
    PlanTier.starter:    {"analyses": 100,   "price": 29},
    PlanTier.pro:        {"analyses": 500,   "price": 79},
    PlanTier.enterprise: {"analyses": 99999, "price": 199},
}
