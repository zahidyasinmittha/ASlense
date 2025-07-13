from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers import learn, translate, practice

app = FastAPI()

# Allow React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount routers
app.include_router(learn.router, prefix="/learn")
app.include_router(translate.router, prefix="/translate")
app.include_router(practice.router, prefix="/practice")
