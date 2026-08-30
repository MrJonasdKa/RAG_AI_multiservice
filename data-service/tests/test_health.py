# NOTE: this test triggers the app's lifespan (DB pool init), so it needs
# DATABASE_URL to point at a reachable Postgres. Run it inside the running
# stack, e.g.:
#   docker compose exec data-service pytest

from fastapi.testclient import TestClient

from app.main import app


def test_health():
    with TestClient(app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok", "service": "data-service"}
