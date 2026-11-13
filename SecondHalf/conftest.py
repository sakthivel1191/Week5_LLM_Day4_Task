import pytest
from fastapi.testclient import TestClient
from app import app, Database, get_db, get_llm_client
TEST_API_KEY = "test_user_secret_123"
class FakeLLMClient:
    async def generate(self, message, model):
        return "Fake AI response"
@pytest.fixture
def test_db(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    db.add_user("testuser", TEST_API_KEY)
    db.add_session("session1")
    return db
@pytest.fixture
def client(test_db):
    app.dependency_overrides[get_db] = lambda: test_db
    app.dependency_overrides[get_llm_client] = lambda: FakeLLMClient()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
@pytest.fixture
def valid_headers():
    return {"x-api-key": TEST_API_KEY}

