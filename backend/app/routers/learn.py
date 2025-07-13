from fastapi import APIRouter

router = APIRouter()

@router.get("/categories")
async def get_categories():
    return [{"id": "alphabet", "name": "Alphabet", "color": "blue", "count": 26}]
