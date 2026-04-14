from fastapi import APIRouter
from core.security import hash_password, verify_password, create_token

router = APIRouter()

USERS = {}

@router.post("/register")
def register(data: dict):
    email = data["email"]
    password = hash_password(data["password"])

    USERS[email] = {
        "email": email,
        "password": password,
        "company_id": data.get("company_id", "DEFAULT")
    }

    return {"message": "User created"}

@router.post("/login")
def login(data: dict):
    user = USERS.get(data["email"])

    if not user or not verify_password(data["password"], user["password"]):
        return {"error": "Invalid credentials"}

    token = create_token({
        "email": user["email"],
        "company_id": user["company_id"],
        "role": "admin"
    })

    return {"token": token}
