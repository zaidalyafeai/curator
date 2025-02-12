import json
import logging
import os
import typing as t
import uuid

import httpx
import requests

from bespokelabs.curator.constants import BASE_CLIENT_URL, PUBLIC_CURATOR_VIEWER_URL

logger = logging.getLogger(__name__)


class _SessionStatus:
    """A class to represent the status of a session."""

    STARTED = "STARTED"
    INPROGRESS = "INPROGRESS"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class Client:
    """A class to represent the client for the Curator Viewer."""

    def __init__(self) -> None:
        """Initialize the client."""
        self._session = None
        self._state = None
        self._hosted = os.environ.get("HOSTED_CURATOR_VIEWER") in ["True", "true", "1", "t"]

    @property
    def session(self):
        """Get the session ID."""
        return self._session

    @property
    def hosted(self):
        """Check if the client is hosted."""
        return self._hosted

    def create_session(self, metadata: t.Dict):
        """Sends a POST request to the server to create a session."""
        if not self.hosted:
            return str(uuid.uuid4().hex)

        if self.session:
            return self.session
        metadata.update({"status": _SessionStatus.STARTED})

        response = requests.post(f"{BASE_CLIENT_URL}/session", json=metadata)

        if response.status_code == 200:
            self._session = response.json().get("session_id")
            logger.info(f"Created session: {self.session}")
            logger.info("Viewer available at: " + f"{PUBLIC_CURATOR_VIEWER_URL}/{self.session}")
            self._state = _SessionStatus.STARTED
            return self.session
        else:
            logger.warning(f"Failed to create session: {response.status_code}, {response.text}")
            return str(uuid.uuid4().hex)

    async def _update_state(self):
        async with httpx.AsyncClient() as client:
            response = await client.put(
                f"{BASE_CLIENT_URL}/session/{self.session}",
                json={"status":self._state}
            )
            if response.status_code != 200:
                logger.debug(f"Failed to update session status: {response.status_code}, {response.text}")

    async def session_completed(self):
        """Updates the session status to completed."""
        self._state = _SessionStatus.COMPLETED
        await self._update_state()

    async def stream_response(self, response_data: str, idx: int):
        """Streams the response data to the server."""
        if not self._hosted or not self.session:
            return
        if self._state == _SessionStatus.STARTED:
            self._state = _SessionStatus.INPROGRESS
            await self._update_state()

        response_data = json.dumps({"response_data": response_data})
        async with httpx.AsyncClient() as client:
            response = await client.post(f"{BASE_CLIENT_URL}/{self.session}/stream_response/{idx}", data=response_data)

        if response.status_code != 200:
            logger.debug(f"Failed to stream response to curator Viewer: {response.status_code}, {response.text}")
