"""Chess tool implementations, decoupled from the agent that registers them.

Every function here takes the HTTP client explicitly instead of reading it off
an agent, so the same code can run in the agent process or inside the sandbox
beside the server it talks to.
"""

from __future__ import annotations

import base64
import json
from typing import Any

import httpx

CHESS_PORT = 8000


def _request_state(
    client: httpx.Client, method: str, endpoint: str, **kwargs: Any
) -> dict[str, Any]:
    """Make one chess API request and validate its JSON response."""

    response = client.request(method, endpoint, **kwargs)
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(
            f"Chess server returned non-JSON ({response.status_code})."
        ) from exc
    if response.status_code >= 400:
        detail = (
            payload.get("detail", payload) if isinstance(payload, dict) else payload
        )
        raise ValueError(str(detail))
    if not isinstance(payload, dict):
        raise RuntimeError("Chess server response must be a JSON object.")
    return payload


def _simulate_move(client: httpx.Client, arguments: str) -> str:
    """New tool: inspect FEN or simulate one ply without changing the game.

    Takes the raw JSON arguments of one tool call and returns the observation
    to send back, so a bad argument or a server error reaches the model as a
    recoverable ``<chess_error>`` instead of ending the run.
    """
    try:
        parsed = json.loads(arguments)
        if not isinstance(parsed, dict):
            raise ValueError("Arguments must be a JSON object.")
        if not isinstance(parsed.get("fen"), str):
            raise ValueError("fen is required and must be a string.")
        payload = {"fen": parsed["fen"]}
        if "move" in parsed:
            if not isinstance(parsed["move"], str):
                raise ValueError("move must be a string.")
            payload["move"] = parsed["move"]
        return json.dumps(_request_state(client, "POST", "/api/simulate", json=payload))
    except Exception as exc:
        return f"<chess_error>{exc}</chess_error>"


def _play_move(client: httpx.Client, arguments: str) -> str:
    """Existing tool: play one move as White and return the resulting state.

    Takes the raw JSON arguments of one tool call. Returns the new state, or a
    `<chess_error>` observation if the move could not be played.
    """
    try:
        parsed = json.loads(arguments)
        if not isinstance(parsed, dict):
            raise ValueError("Arguments must be a JSON object.")
        if not isinstance(parsed.get("move"), str):
            raise ValueError("move is required and must be a string.")
        return json.dumps(
            _request_state(client, "POST", "/api/move", json={"move": parsed["move"]})
        )
    except Exception as exc:
        return f"<chess_error>{exc}</chess_error>"


def _run_python(env: Any, port: int, arguments: str) -> str:
    """New tool: run Python with access to the existing registered tools.

    The snippet runs inside the sandbox, which already has the tool
    implementations and the chess server, so code the model wrote never
    executes in the agent process.
    """
    try:
        parsed = json.loads(arguments)
        if not isinstance(parsed, dict):
            raise ValueError("Arguments must be a JSON object.")
        if not isinstance(parsed.get("code"), str):
            raise ValueError("code is required and must be a string.")
        encoded = base64.b64encode(parsed["code"].encode()).decode("ascii")
        result = env.execute(
            ["python", "/opt/assignment/sandbox_python.py", str(port), encoded],
            shell=False,
        )
        if result["returncode"] != 0:
            raise RuntimeError(
                result.get("exception_info")
                or result.get("stderr")
                or "Sandbox execution failed."
            )
        return result["stdout"]
    except Exception as exc:
        return f"<chess_error>{exc}</chess_error>"


def _invoke_skill(skills: dict[str, dict[str, str]], arguments: str) -> str:
    """Existing tool: load one skill's instructions into the conversation."""
    try:
        parsed = json.loads(arguments)
        if not isinstance(parsed, dict):
            raise ValueError("Arguments must be a JSON object.")
        name = parsed.get("name")
        if not isinstance(name, str):
            raise ValueError("name is required and must be a string.")
        if name not in skills:
            raise ValueError(f"Unknown skill {name!r}")
        return skills[name]["content"]
    except Exception as exc:
        return f"<chess_error>{exc}</chess_error>"


def _game_state(client: httpx.Client, reset: bool = False) -> dict:
    """Read the live game, or start a new one and read the opening position."""

    method, endpoint = ("POST", "/api/reset") if reset else ("GET", "/api/state")
    return _request_state(client, method, endpoint)
