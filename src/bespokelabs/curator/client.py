import logging
import uuid
import os
import json
import typing as t

import requests

from bespokelabs.curator.constants import BASE_CLIENT_URL, PUBLIC_CURATOR_VIEWER_URL
logger = logging.getLogger(__name__)


class Client:
    """A class to represent the client for the Curator Viewer."""
    def __init__(self) -> None:
        """Initialize the client."""
        self._session = None
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
            return

        response = requests.post(f"{BASE_CLIENT_URL}/create_session", json={"metadata": metadata})

        if response.status_code == 200:
            self._session = response.json().get("session_id")
            logger.info(f"Created session: {self.session}")
            logger.info("Viewer available at: " + f"{PUBLIC_CURATOR_VIEWER_URL}/{self.session}")
            return self.session
        else:
            raise Exception(f"Failed to create session: {response.status_code}, {response.text}")

    def stream_response(self, response_data: str, idx: int):
        """Streams the response data to the server."""
        if not self._hosted:
            return
        response_data = json.dumps({"response_data": response_data})
        response = requests.post(f"{BASE_CLIENT_URL}/{self.session}/stream_response/{idx}", data=response_data)

        if response.status_code != 200:
            raise Exception(f"Failed to stream response: {response.status_code}, {response.text}")
