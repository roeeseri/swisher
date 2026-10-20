from __future__ import annotations

import copy
import http.client
import json
import threading
from http.server import ThreadingHTTPServer

import av
import cv2
import numpy as np
import pytest

from swisher.studio.media import clip, frame
from swisher.studio.server import Handler, Studio
from swisher.studio.store import StudioStore, suggestion


@pytest.fixture
def workspace(tmp_path):
    store = StudioStore(tmp_path)
    job = store.add_job(
        name="test",
        source="private.mov",
        player="p01",
        view="left",
        consent=True,
        video={"frame_count": 100, "fps": 30},
    )
    candidate = {"shot_id": "shot_001", "shot_start": 10, "shot_end": 90, "release_frame": 50}
    store.data["shots"].append(
        {
            "id": "test-shot",
            "job_id": job["id"],
            "candidate": candidate,
            "suggestion": {"available": True, "text": "original suggestion"},
            "media_status": "ready",
            "reviews": [],
        }
    )
    store.save()
    return store


def review_payload(**overrides):
    return {
        "version": 0,
        "decision": "corrected",
        "coach": "coach_01",
        "note": "coach correction",
        "start_frame": 11,
        "end_frame": 89,
        "release_frame": 49,
        "outcome": "unknown",
        **overrides,
    }


def test_review_keeps_model_and_history_and_requeues_media(workspace):
    original = copy.deepcopy(workspace.shot("test-shot")["candidate"])
    workspace.review("test-shot", review_payload())
    workspace.review("test-shot", review_payload(version=1, decision="approved"))
    reloaded = StudioStore(workspace.root)
    shot = reloaded.shot("test-shot")
    assert shot["candidate"] == original
    assert shot["suggestion"]["text"] == "original suggestion"
    assert shot["reviews"][0]["note"] == "coach correction"
    assert shot["reviews"][1]["note"] == "original suggestion"
    assert shot["media_status"] == "queued"


@pytest.mark.parametrize(
    "fields",
    [
        {"version": 99},
        {"coach": ""},
        {"decision": "corrected", "note": ""},
        {"start_frame": 60, "end_frame": 40},
        {"release_frame": 99},
        {"end_frame": 100},
        {"outcome": "probably_made"},
    ],
)
def test_bad_review_does_not_mutate(workspace, fields):
    before = workspace.snapshot()
    with pytest.raises(ValueError):
        workspace.review("test-shot", review_payload(**fields))
    assert workspace.snapshot() == before


def test_abstention_cannot_be_approved(workspace):
    workspace.shot("test-shot")["suggestion"]["available"] = False
    with pytest.raises(ValueError):
        workspace.review("test-shot", review_payload(decision="approved"))


def test_suggestion_abstains_and_has_no_probability():
    measurement = {
        "valid_frame_coverage": 0.99,
        "metrics": {"min_knee_angle_deg": 100, "max_knee_angle_deg": 170},
    }
    assert not suggestion({"release_frame": None}, measurement)["available"]
    assert not suggestion({"release_frame": 10}, {**measurement, "tracking_review_failed": True})[
        "available"
    ]
    assert not suggestion({"release_frame": 10}, {**measurement, "valid_frame_coverage": 0.3})[
        "available"
    ]
    result = suggestion({"release_frame": 10}, measurement)
    assert result["available"] and "confidence" not in result
    assert "100" in result["evidence"][0]


def test_restart_recovers_interrupted_work(workspace):
    workspace.data["jobs"][0]["status"] = "processing"
    workspace.shot("test-shot")["media_status"] = "processing"
    workspace.save()
    restored = StudioStore(workspace.root)
    assert restored.data["jobs"][0]["status"] == "queued"
    assert restored.shot("test-shot")["media_status"] == "queued"


def test_media_clip_is_h264_frame_exact_and_source_retained(tmp_path):
    source = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for i in range(30):
        writer.write(np.full((240, 320, 3), i * 5, dtype=np.uint8))
    writer.release()
    original = source.read_bytes()
    target = tmp_path / "clip.mp4"
    clip(source, target, 5, 14, 30)
    with av.open(str(target)) as movie:
        assert movie.streams.video[0].codec_context.name == "h264"
        frames = list(movie.decode(video=0))
        assert len(frames) == 10
        assert abs(float(frames[-1].time) - 0.3) < 0.01
    frame(source, tmp_path / "release.jpg", 10)
    assert source.read_bytes() == original
    assert (tmp_path / "release.jpg").is_file()


@pytest.fixture
def api(tmp_path):
    studio = Studio(tmp_path)
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.studio = studio
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def request(method, url, payload=None, authorized=True, cookie=False):
        connection = http.client.HTTPConnection("127.0.0.1", server.server_port)
        headers = {"Content-Type": "application/json"}
        if authorized:
            headers["X-Swisher-Token"] = studio.token
        if cookie:
            headers["Cookie"] = "swisher=" + studio.token
        connection.request(
            method, url, body=json.dumps(payload) if payload is not None else None, headers=headers
        )
        response = connection.getresponse()
        data = response.read()
        connection.close()
        return response.status, data

    yield studio, request
    server.shutdown()
    server.server_close()


def test_api_private_state_requires_cookie_and_write_requires_token(api):
    studio, request = api
    assert request("GET", "/api/state", authorized=False)[0] == 403
    assert request("GET", "/api/state", cookie=True)[0] == 200
    assert request("POST", "/api/review", {}, authorized=False)[0] == 403
    assert request("POST", "/api/upload?name=x.mov", {})[0] == 400
    assert not studio.store.data["jobs"]
    assert not (studio.store.root / "uploads").exists()


def test_manual_shot_validation_and_export(api):
    studio, request = api
    job = studio.store.add_job(
        name="source",
        source="private.mov",
        player="p01",
        view="left",
        status="ready",
        video={"fps": 30, "frame_count": 100},
    )
    data = {"job_id": job["id"], "start_frame": 10, "end_frame": 60, "release_frame": 40}
    assert request("POST", "/api/manual-shot", {**data, "release_frame": 80})[0] == 400
    status, body = request("POST", "/api/manual-shot", data)
    assert status == 201
    shot = studio.store.shot(json.loads(body)["id"])
    assert shot["candidate"]["method"] == "manual_added"
    assert not shot["suggestion"]["available"]
    assert (
        request("POST", "/api/review", {"id": shot["id"], **review_payload(end_frame=60)})[0] == 200
    )
    status, body = request("GET", "/api/export", cookie=True)
    export = json.loads(body)
    assert status == 200
    assert export["shots"][0]["reviews"][0]["coach"] == "coach_01"
    assert b"private.mov" not in body


def test_phone_pairing_never_exposes_desktop_token(api):
    import time

    studio, request = api
    # Exercise phone authorization on the loopback test server, without LAN exposure.
    request("GET", "/")
    # The fixture's request closure holds the server through the bound helper.
    # A dedicated local phone handler makes bootstrap behavior explicit.
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.studio = studio
    server.phone = True
    threading.Thread(target=server.serve_forever, daemon=True).start()

    def call(method, url, data=None, cookie=None, token=None):
        conn = http.client.HTTPConnection("127.0.0.1", server.server_port)
        headers = {"Content-Type": "application/json"}
        if cookie:
            headers["Cookie"] = cookie
        if token:
            headers["X-Swisher-Token"] = token
        conn.request(method, url, json.dumps(data) if data is not None else None, headers)
        response = conn.getresponse()
        body = response.read()
        result = response.status, body, response.getheader("Set-Cookie")
        conn.close()
        return result

    try:
        status, body, cookie = call("GET", "/")
        assert status == 200 and studio.token.encode() not in body and cookie is None
        assert call("GET", "/api/state", cookie="swisher=" + studio.token)[0] == 403
        assert call("POST", "/api/pair", {"code": "bad"})[0] == 400
        status, body, cookie = call("POST", "/api/pair", {"code": studio.pair_code})
        assert status == 200 and "Secure" in cookie and "HttpOnly" in cookie
        phone_token = cookie.split(";")[0].split("=")[1]
        assert phone_token != studio.token
        assert call("GET", "/api/state", cookie=cookie.split(";")[0])[0] == 200
        assert call("POST", "/api/pair", {"code": studio.pair_code})[0] == 400
        studio.phone_sessions[phone_token] = time.monotonic() - 1
        assert call("GET", "/api/state", cookie=cookie.split(";")[0])[0] == 403
    finally:
        server.shutdown()
        server.server_close()


def test_tracking_drawing_does_not_invent_hidden_joints_or_balls():
    from swisher.studio.media import draw_tracking

    image = np.zeros((200, 200, 3), dtype=np.uint8)
    pose = {
        "valid": True,
        "xy": {"ls": [30, 30], "le": [90, 90]},
        "visibility_by_name": {"ls": 0.1, "le": 0.1},
    }
    draw_tracking(image, pose, [])
    assert not image.any()
    pose["visibility_by_name"] = {"ls": 0.9, "le": 0.9}
    draw_tracking(image, pose, [{"center": [140, 140], "confidence": 0.8}])
    assert image[60, 60, 0] > 100  # visible skeletal edge
    assert image[140, 154, 1] > 150  # observed ball ring
