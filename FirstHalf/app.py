import uuid
import json
import logging
from contextlib import asynccontextmanager
import httpx
from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    Request
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    openrouter_api_key: str = ""
    model_name: str = "openai/gpt-4o-mini"
    model_config = SettingsConfigDict(env_file=".env")
settings = Settings()
def get_settings():
    return settings
class Database:
    def __init__(self):
        self.sessions = {
            "session1": {
                "messages": []
            }
        }
class LLMClient:
    def __init__(self, api_key: str):
        self.client = httpx.AsyncClient(
            base_url="https://openrouter.ai/api/v1",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "Week 5 FastAPI Chatbot"
            },
            timeout=30.0
        )
    async def generate(self, message: str, model: str):
        response = await self.client.post(
            "/chat/completions",
            json={
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": message
                    }
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
    app.state.llm_client = LLMClient(settings.openrouter_api_key)
    app.state.db = Database()
    print("Shared LLM client created")
    yield
    await app.state.llm_client.close()
    print("LLM client closed")
app = FastAPI(lifespan=lifespan)
def get_llm_client(request: Request):
    return request.app.state.llm_client
def get_db(request: Request):
    return request.app.state.db
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger("fastapi_app")
@app.middleware("http")
async def request_logging(request: Request, call_next):
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    response = await call_next(request)
    log_data = {
        "request_id": request_id,
        "method": request.method,
        "path": request.url.path,
        "status_code": response.status_code
    }
    logger.info(json.dumps(log_data))
    response.headers["X-Request-ID"] = request_id
    return response
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    error_names = {
        404: "not_found",
        413: "payload_too_large",
        502: "upstream_error"
    }
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": error_names.get(
                exc.status_code,
                "http_error"
            ),
            "detail": exc.detail,
            "request_id": getattr(
                request.state,
                "request_id",
                "unknown"
            )
        }
    )
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unexpected application error")
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": "Something went wrong",
            "request_id": getattr(
                request.state,
                "request_id",
                "unknown"
            )
        }
    )
class ChatRequest(BaseModel):
    message: str
@app.post("/chat/{session_id}")
async def chat(
    session_id: str,
    body: ChatRequest,
    llm: LLMClient = Depends(get_llm_client),
    db: Database = Depends(get_db),
    config: Settings = Depends(get_settings)
):
    if session_id not in db.sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    if len(body.message) > 1000:
        raise HTTPException(
            status_code=413,
            detail="Message is too long"
        )
    try:
        answer = await llm.generate(
            body.message,
            config.model_name
        )
    except (
        httpx.HTTPError,
        KeyError,
        ValueError
    ):
        raise HTTPException(
            status_code=502,
            detail="LLM service failed"
        )
    db.sessions[session_id]["messages"].append({
        "user": body.message,
        "assistant": answer
    })
    return {
        "session_id": session_id,
        "message": body.message,
        "response": answer
    }
@app.get("/sessions/{session_id}")
async def get_session(
    session_id: str,
    db: Database = Depends(get_db)
):
    if session_id not in db.sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )
    return {
        "session_id": session_id,
        "messages": db.sessions[session_id]["messages"]
    }
