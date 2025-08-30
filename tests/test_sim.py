from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
API="/api"

def test_sim_flow():
    # start
    r = client.post(f"{API}/sim/start", json={"role":"node-react","level":"junior","mode":"technical"})
    assert r.status_code == 200
    data = r.json()
    sid, qid = data["session_id"], data["question_id"]
    # answer (text)
    r = client.post(f"{API}/sim/answer/text", json={"session_id": sid, "question_id": qid, "text":"I built a small API"})
    assert r.status_code == 200
    # score
    r = client.post(f"{API}/sim/score", json={"session_id": sid, "question_id": qid})
    assert r.status_code == 200
    # report
    r = client.get(f"{API}/sim/report", params={"session_id": sid})
    assert r.status_code == 200
