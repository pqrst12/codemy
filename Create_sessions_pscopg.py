import uuid
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel
import psycopg
from psycopg.rows import dict_row

# FastAPI instance
app = FastAPI()

# Database configuration
DATABASE_URL = "dbname=yourdbname user=youruser password=yourpassword host=localhost port=5432"

# Models for request and response
class SessionRequest(BaseModel):
    user_id: int

class SessionResponse(BaseModel):
    status: str
    data: dict

# Utility to get database connection
def get_db_connection():
    try:
        conn = psycopg.connect(DATABASE_URL, row_factory=dict_row)
        return conn
    except Exception as e:
        raise HTTPException(status_code=500, detail="Unable to connect to the database")


@app.post("/create_session", response_model=SessionResponse)
async def create_session(
    request: SessionRequest,
    X_Request_ID: str = Header(...),
    X_user_id: int = Header(...),
):
    try:
        # Step 1: Establish a database connection
        conn = get_db_connection()
        cur = conn.cursor()

        # Step 2: Fetch persona from the `genie_appinfo` table
        cur.execute("SELECT persona FROM genie_appinfo WHERE user_id = %s", (request.user_id,))
        app_info = cur.fetchone()
        if not app_info:
            raise HTTPException(status_code=404, detail="User not found in genie_appinfo")

        user_persona = app_info['persona']

        # Step 3: Check if an active session exists
        cur.execute("""
            SELECT session_id FROM user_sessions 
            WHERE user_id = %s AND expiry_time > %s
        """, (request.user_id, datetime.utcnow()))
        active_session = cur.fetchone()

        if active_session:
            # Return the existing active session
            return SessionResponse(
                status="OK",
                data={"X-session-id": str(active_session['session_id'])}
            )

        # Step 4: Create a new session if no active session exists
        session_id = uuid.uuid4()
        expiry_time = datetime.utcnow() + timedelta(hours=4)

        # Insert new session into the `user_sessions` table
        cur.execute("""
            INSERT INTO user_sessions (session_id, user_id, persona, expiry_time, created_at)
            VALUES (%s, %s, %s, %s, %s)
        """, (session_id, request.user_id, user_persona, expiry_time, datetime.utcnow()))

        # Commit the transaction
        conn.commit()

        # Step 5: Return the new session ID in the response
        return SessionResponse(
            status="OK",
            data={"X-session-id": str(session_id)}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

    finally:
        # Close the connection and cursor
        cur.close()
        conn.close()





CREATE TABLE user_sessions (
    session_id UUID PRIMARY KEY,
    user_id INTEGER NOT NULL,
    persona VARCHAR NOT NULL,
    expiry_time TIMESTAMP NOT NULL,
    created_at TIMESTAMP NOT NULL
);


CREATE TABLE genie_appinfo (
    user_id INTEGER PRIMARY KEY,
    persona VARCHAR NOT NULL
);



