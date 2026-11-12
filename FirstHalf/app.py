
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


# ---------------------------------
# 1. SETTINGS
# ---------------------------------

class Settings(BaseSettings):

    openrouter_api_key: str = ""
    model_name: str = "openai/gpt-4o-mini"

    model_config = SettingsConfigDict(
        env_file=".env"
    )


settings = Settings()


def get_settings():
    return settings


# ---------------------------------
# 2. DATABASE
# ---------------------------------

class Database:

    def __init__(self):

        self.sessions = {
            "session1": {
                "messages": []
            }
        }


# ---------------------------------
# 3. SHARED LLM CLIENT
# ---------------------------------

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


# ---------------------------------
# 4. APPLICATION LIFESPAN
# ---------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    # Create LLM client only once
    app.state.llm_client = LLMClient(
        settings.openrouter_api_key
    )

    # Create database only once
    app.state.db = Database()

    print("Shared LLM client created")

    yield

    # Close client during shutdown
    await app.state.llm_client.close()

    print("LLM client closed")


app = FastAPI(lifespan=lifespan)


# ---------------------------------
# 5. DEPENDENCY INJECTION
# ---------------------------------

def get_llm_client(request: Request):

    return request.app.state.llm_client


def get_db(request: Request):

    return request.app.state.db


# ---------------------------------
# 6. CORS MIDDLEWARE
# ---------------------------------

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


# ---------------------------------
# 7. STRUCTURED JSON LOGGING
# ---------------------------------

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


# ---------------------------------
# 8. GLOBAL HTTP EXCEPTION HANDLER
# ---------------------------------

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


# ---------------------------------
# 9. GENERAL EXCEPTION HANDLER
# ---------------------------------

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


# ---------------------------------
# 10. REQUEST MODEL
# ---------------------------------

class ChatRequest(BaseModel):

    message: str


# ---------------------------------
# 11. ASYNC CHAT ROUTE
# ---------------------------------

@app.post("/chat/{session_id}")
async def chat(

    session_id: str,

    body: ChatRequest,

    llm: LLMClient = Depends(get_llm_client),

    db: Database = Depends(get_db),

    config: Settings = Depends(get_settings)

):

    # Check session
    if session_id not in db.sessions:

        raise HTTPException(
            status_code=404,
            detail="Session not found"
        )

    # Check message length
    if len(body.message) > 1000:

        raise HTTPException(
            status_code=413,
            detail="Message is too long"
        )

    try:

        # Call LLM asynchronously
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

    # Save conversation
    db.sessions[session_id]["messages"].append({

        "user": body.message,

        "assistant": answer

    })

    return {

        "session_id": session_id,

        "message": body.message,

        "response": answer

    }


# ---------------------------------
# 12. GET SESSION ROUTE
# ---------------------------------

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
