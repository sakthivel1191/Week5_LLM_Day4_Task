from fastapi.testclient import TestClient
from FirstHalf.app import app,get_llm_client
class FakeLLMClient:
    async def generate(self, message, model):
        return "This is a fake AI response"
fake_llm = FakeLLMClient()
app.dependency_overrides[get_llm_client] =lambda: fake_llm
def test_chat_with_fake_llm():
    with TestClient(app) as client:
        response = client.post(
            "/chat/session1",
            json={
                "message": "Hello AI"
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert data["response"] == ("This is a fake AI response")
        assert data["session_id"] == "session1"
def test_unknown_session():
    with TestClient(app) as client:
        response = client.get("/sessions/unknown")
        assert response.status_code == 404
        data = response.json()
        assert data["error"] == "not_found"
        assert data["detail"] == "Session not found"
        assert "request_id" in data
def test_message_too_long():
    with TestClient(app) as client:
        response = client.post(
            "/chat/session1",
            json={
                "message": "A" * 1001
            }
        )
        assert response.status_code == 413
        assert response.json()["error"] == ("payload_too_large")
def test_dependency_override():
    with TestClient(app) as client:
        response = client.post(
            "/chat/session1",
            json={
                "message": "Testing dependency"
            }
        )
        assert response.status_code == 200
        assert response.json()["response"] == ("This is a fake AI response")
def teardown_module():
    app.dependency_overrides.clear()
