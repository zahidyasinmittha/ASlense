from fastapi import APIRouter

router = APIRouter()

@router.get("/ping")
async def practice_ping():
    return {"msg": "Practice endpoint works"}
