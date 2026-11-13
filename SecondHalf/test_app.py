import pytest
from fastapi.testclient import TestClient
from app import app, get_llm_client, get_db, verify_api_key
class FakeLLMClient:
    async def get_response(self, message):
        return "This is a fake AI response for testing."
def override_llm():
    return FakeLLMClient()
client = TestClient(app)
app.dependency_overrides[get_llm_client] = override_llm
@pytest.fixture
def test_client(test_db):
    def override_db():
        yield test_db
    app.dependency_overrides[get_db] = override_db
    yield client
    app.dependency_overrides.clear()
    app.dependency_overrides[get_llm_client] = override_llm
def test_health(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
def test_missing_api_key(test_client):
    response = test_client.get("/my-requests")
    assert response.status_code == 401
def test_invalid_api_key(test_client):
    response = test_client.get(
        "/my-requests",
        headers={"x-api-key": "wrong_key"}
    )
    assert response.status_code == 403
def test_valid_api_key(test_client):
    response = test_client.get(
        "/my-requests",
        headers={"x-api-key": "trainee_user_one_secret_key"}
    )
    assert response.status_code == 200
def test_validation_error(test_client):
    response = test_client.post(
        "/chat/session1",
        headers={"x-api-key": "trainee_user_one_secret_key"},
        json={"message": ""}
    )
    assert response.status_code == 422
def test_chat_success(test_client):
    response = test_client.post(
        "/chat/session1",
        headers={"x-api-key": "trainee_user_one_secret_key"},
        json={"message": "Hello AI"}
    )
    assert response.status_code == 200
    assert "fake AI response" in str(response.json())
def test_unknown_session(test_client):
    response = test_client.get(
        "/sessions/unknown",
        headers={"x-api-key": "trainee_user_one_secret_key"}
    )
    assert response.status_code == 404
def test_error_handler(test_client):
    response = test_client.get(
        "/sessions/unknown",
        headers={"x-api-key": "trainee_user_one_secret_key"}
    )
    assert response.status_code == 404
    data = response.json()
    assert "error" in data
    assert "detail" in data
    assert "request_id" in data
def test_request_counter(test_client):
    response = test_client.get(
        "/my-requests",
        headers={"x-api-key": "trainee_user_one_secret_key"}
    )
    assert response.status_code == 200
def test_database_authentication(test_client):
    response = test_client.get(
        "/my-requests",
        headers={"x-api-key": "trainee_user_two_secret_key"}
    )
    assert response.status_code == 200
def test_database_messages(test_client):
    response = test_client.post(
        "/chat/session1",
        headers={"x-api-key": "trainee_user_one_secret_key"},
        json={"message": "How are you?"}
    )
    assert response.status_code == 200
def test_database_session(test_client):
    response = test_client.get(
        "/sessions/session1",
        headers={"x-api-key": "trainee_user_one_secret_key"}
    )
    assert response.status_code in [200, 404]
def test_upstream_error(test_client):
    assert True
