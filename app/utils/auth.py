from datetime import datetime, timedelta, timezone
from typing import Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from config.settings import settings

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 480
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token")

class User(BaseModel):
    username: str
    email: Optional[str] = None
    disabled: bool = False

_USERS_DB = {
    "admin": {
        "username": "admin",
        "hashed_password": pwd_context.hash("datamind2024"),
        "disabled": False,
    }
}

def create_access_token(data: dict):
    to_encode = data.copy()
    to_encode["exp"] = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
        username = payload.get("sub")
        user = _USERS_DB.get(username)
        if not user:
            raise HTTPException(status_code=401, detail="Invalid credentials")
        return User(**{k:v for k,v in user.items() if k != "hashed_password"})
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid credentials")
