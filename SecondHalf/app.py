import sqlite3
import hashlib
import secrets
from pathlib import Path
from contextlib import asynccontextmanager
import httpx
import json
from fastapi import FastAPI, Depends, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
BASE_DIR = Path(__file__).resolve().parent
class Settings(BaseSettings):
    openrouter_api_key: str = ""
    model_name: str = "openai/gpt-4o-mini"
    user1_api_key: str = ""
    user2_api_key: str = ""
    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        extra="ignore"
    )
settings = Settings()
class Database:
    def __init__(self, db_path):
        self.db_path = str(db_path)
    def initialize(self):
        with sqlite3.connect(self.db_path) as db:
            db.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    key_hash TEXT NOT NULL,
                    request_count INTEGER DEFAULT 0
                )
            """)
            db.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    messages TEXT DEFAULT '[]'
                )
            """)
    def add_user(self, username, api_key):
        key_hash = hashlib.sha256(
            api_key.encode()
        ).hexdigest()
        with sqlite3.connect(self.db_path) as db:
            db.execute("""
                INSERT OR REPLACE INTO users
                (username, key_hash, request_count)
                VALUES (?, ?, 0)
            """, (username, key_hash))
    def authenticate(self, api_key):
        key_hash = hashlib.sha256(
            api_key.encode()
        ).hexdigest()
        with sqlite3.connect(self.db_path) as db:
            users = db.execute("SELECT username, key_hash FROM users").fetchall()
        for username, stored_hash in users:
            if secrets.compare_digest(key_hash, stored_hash):
                return username
        return None
    def increment_requests(self, username):
        with sqlite3.connect(self.db_path) as db:
            db.execute("""
                UPDATE users
                SET request_count = request_count + 1
                WHERE username = ?
            """, (username,))
    def get_request_count(self, username):
        with sqlite3.connect(self.db_path) as db:
            result = db.execute(
                "SELECT request_count FROM users WHERE username = ?",
                (username,)
            ).fetchone()
        return result[0] if result else 0
    def add_session(self, session_id):
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
                (session_id,)
            )
    def session_exists(self, session_id):
        with sqlite3.connect(self.db_path) as db:
            result = db.execute(
                "SELECT session_id FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
        return result is not None
    def get_messages(self, session_id):
        with sqlite3.connect(self.db_path) as db:
            result = db.execute(
                "SELECT messages FROM sessions WHERE session_id = ?",
                (session_id,)
            ).fetchone()
        if result:
            return json.loads(result[0])
        return []
    def save_message(self, session_id, user_message, assistant_message):
        messages = self.get_messages(session_id)
        messages.append({
            "user": user_message,
            "assistant": assistant_message
        })
        with sqlite3.connect(self.db_path) as db:
            db.execute(
                "UPDATE sessions SET messages = ? WHERE session_id = ?",
                (json.dumps(messages), session_id)
            )
class LLMClient:
    def __init__(self, api_key):
        self.client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            timeout=30.0
        )
    async def generate(self, message, model):
        response = await self.client.post(
            "/chat/completions",
            json={
                "model": model,
                "messages": [
                    {"role": "user", "content": message}
                ]
            }
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    async def close(self):
        await self.client.aclose()
@asynccontextmanager
async def lifespan(app: FastAPI):
    db_path = BASE_DIR / "auth.db"
    app.state.db = Database(db_path)
    app.state.db.initialize()
    if settings.user1_api_key:
        app.state.db.add_user("user1", settings.user1_api_key)
    if settings.user2_api_key:
        app.state.db.add_user("user2", settings.user2_api_key)
    app.state.db.add_session("session1")
    app.state.llm = LLMClient(settings.openrouter_api_key)
    print("Application started")
    yield
    await app.state.llm.close()
app = FastAPI(lifespan=lifespan)
def get_db(request: Request):
    return request.app.state.db
def get_llm_client(request: Request):
    return request.app.state.llm
def verify_api_key(
    x_api_key: str | None = Header(default=None, alias="x-api-key"),
    db: Database = Depends(get_db)
):
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key is missing"
        )
    username = db.authenticate(x_api_key)
    if username is None:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    db.increment_requests(username)
    return username
class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
@app.exception_handler(HTTPException)
async def http_error_handler(request: Request, exc: HTTPException):
    error_names = {
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        502: "upstream_error"
    }
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": error_names.get(exc.status_code, "http_error"),
            "detail": exc.detail
        }
    )
@app.get("/health")
async def health():
    return {"status": "healthy"}
@app.post("/chat/{session_id}")
async def chat(
    session_id: str,
    body: ChatRequest,
    username: str = Depends(verify_api_key),
    db: Database = Depends(get_db),
    llm: LLMClient = Depends(get_llm_client)
):
    if not db.session_exists(session_id):
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    try:
        answer = await llm.generate(
            body.message,
            settings.model_name
        )
    except (httpx.HTTPError, KeyError, ValueError, IndexError, TypeError):
        raise HTTPException(
            status_code=502,
            detail="LLM service failed"
        )
    db.save_message(
        session_id,
        body.message,
        answer
    )
    return {
        "username": username,
        "session_id": session_id,
        "message": body.message,
        "response": answer,
        "request_count": db.get_request_count(username)
    }
@app.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    username: str = Depends(verify_api_key),
    db: Database = Depends(get_db)
):
    if not db.session_exists(session_id):
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    return {
        "username": username,
        "session_id": session_id,
        "messages": db.get_messages(session_id)
    }
@app.get("/my-requests")
async def my_requests(
    username: str = Depends(verify_api_key),
    db: Database = Depends(get_db)
):
    return {
        "username": username,
        "request_count": db.get_request_count(username)
    }
