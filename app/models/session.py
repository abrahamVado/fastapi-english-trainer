from sqlalchemy.orm import declarative_base, mapped_column
from sqlalchemy import Integer, String

Base = declarative_base()

class Session(Base):
    __tablename__ = "sessions"
    id = mapped_column(Integer, primary_key=True)
    user_id = mapped_column(Integer)
    status = mapped_column(String)
