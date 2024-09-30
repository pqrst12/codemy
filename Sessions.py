from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime
import uuid

Base = declarative_base()

# User model (for illustration; typically users would already be in the DB)
class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True)
    persona = Column(String, nullable=False)

# Session model for storing session data
class UserSession(Base):
    __tablename__ = 'user_sessions'
    session_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    persona = Column(String, nullable=False)
    expiry_time = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

# App Info model for persona details (genie_appinfo table)
class GenieAppInfo(Base):
    __tablename__ = 'genie_appinfo'
    user_id = Column(Integer, primary_key=True)
    persona = Column(String, nullable=False)





from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timedelta
import uuid

# Database configuration
DATABASE_URL = "postgresql://user:password@localhost/dbname"
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# FastAPI instance
app = FastAPI()

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Models for request and response
class SessionRequest(BaseModel):
    user_id: int

class SessionResponse(BaseModel):
    status: str
    data: dict

@app.post("/create_session", response_model=SessionResponse)
async def create_session(
    request: SessionRequest,
    X_Request_ID: str = Header(...),
    X_user_id: int = Header(...),
    db: Session = Depends(get_db)
):
    try:
        # Fetch the user's persona from genie_appinfo table
        app_info = db.query(GenieAppInfo).filter(GenieAppInfo.user_id == request.user_id).first()
        if not app_info:
            raise HTTPException(status_code=404, detail="User not found in genie_appinfo")

        user_persona = app_info.persona

        # Check if an active session exists
        active_session = db.query(UserSession).filter(
            UserSession.user_id == request.user_id,
            UserSession.expiry_time > datetime.utcnow()
        ).first()

        if active_session:
            # If active session exists, return it
            return SessionResponse(
                status="OK",
                data={"X-session-id": str(active_session.session_id)}
            )

        # Create a new session ID and set expiry time to 4 hours
        session_id = uuid.uuid4()
        expiry_time = datetime.utcnow() + timedelta(hours=4)

        # Create a new session entry in the DB
        new_session = UserSession(
            session_id=session_id,
            user_id=request.user_id,
            persona=user_persona,
            expiry_time=expiry_time
        )
        db.add(new_session)
        db.commit()

        return SessionResponse(
            status="OK",
            data={"X-session-id": str(session_id)}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")



{
    "user_id": 123
}

{
    "status": "OK",
    "data": {
        "X-session-id": "generated-session-id-here"
    }
}

