from sqlalchemy import Column, Integer, String, Float, LargeBinary, DateTime
from sqlalchemy.sql import func
from app.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False)
    image_paths = Column(String, nullable=True)
    face_embedding = Column(LargeBinary, nullable=True)
    hog_feature = Column(LargeBinary, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
