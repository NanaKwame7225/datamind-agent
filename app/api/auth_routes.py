from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from core.database import users_col
from passlib.context import CryptContext
from jose import jwt
import os, datetime

router  = APIRouter()
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET  = os.environ.get("JWT_SECRET", "datamind-secret-change-in-prod")
ALGO    = "HS256"

class RegisterRequest(BaseModel):
    email: str
    password: str
    company: str = "DEFAULT"

class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/register")
def register(req: RegisterRequest):
    if users_col.find_one({"email": req.email}):
        raise HTTPException(400, "Email already registered")
    users_col.insert_one({
        "email":      req.email,
        "password":   pwd_ctx.hash(req.password),
        "company_id": req.company,
        "created_at": datetime.datetime.utcnow().isoformat(),
    })
    return {"status": "registered", "email": req.email}

@router.post("/login")
def login(req: LoginRequest):
    user = users_col.find_one({"email": req.email})
    if not user or not pwd_ctx.verify(req.password, user["password"]):
        raise HTTPException(401, "Invalid credentials")
    token = jwt.encode(
        {
            "sub":     req.email,
            "company": user.get("company_id", "DEFAULT"),
            "exp":     datetime.datetime.utcnow() + datetime.timedelta(hours=24),
        },
        SECRET, algorithm=ALGO,
    )
    return {"access_token": token, "token_type": "bearer"}
