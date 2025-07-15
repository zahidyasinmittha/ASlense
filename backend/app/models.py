# app/models.py
from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class Video(Base):
    __tablename__ = "videos"

    id          = Column(Integer, primary_key=True, index=True)
    word        = Column(String, index=True)
    title       = Column(String)
    description = Column(String)
    difficulty  = Column(String)
    duration    = Column(String)   # e.g. "42s"
    video_file  = Column(String)   # "A.mp4"
    thumbnail   = Column(String)   # "thumbnails/A.jpg"
    category    = Column(String, index=True)

# class Prediction(Base):
#     pass