# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.db import engine                     
from app.models import Base, Video
from app.routers import learn, translate, practice

from sqladmin import Admin
from app.db import engine
from sqladmin.models import ModelView



app = FastAPI()

class YourModelAdmin(ModelView, model=Video):
    column_list = [Video.id, Video.title, Video.category, Video.difficulty]

admin = Admin(app, engine)
admin.add_view(YourModelAdmin)


app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "Accept-Ranges", "Content-Length"],
)


# Create tables
Base.metadata.create_all(bind=engine)

# router mounting
app.include_router(learn.router,     prefix="/learn")
# app.include_router(translate.router, prefix="/translate")
# app.include_router(practice.router,  prefix="/practice")