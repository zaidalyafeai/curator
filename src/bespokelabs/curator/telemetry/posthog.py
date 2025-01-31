import logging
import os
from dataclasses import dataclass
from typing import Optional

import posthog
from bespokelabs.curator.telemetry.events import TelemetryEvent

logger = logging.getLogger(__name__)

@dataclass
class TelemetryEvent:
    """Base class for all telemetry events."""
    event_type: str
    event_id: str
    metadata: dict


@dataclass
class PosthogConfig:
    """Configuration settings for PostHog client."""
    api_key: str
    enabled: bool = True
    debug: bool = False
    disable_geoip: bool = False
    host: Optional[str] = None


class PosthogClient:
    """
    Client for sending telemetry events to PostHog analytics.
    
    This uses a write-only project API key that can only create new events.
    It cannot read events or access other PostHog data, making it safe for public apps.
    """

    def __init__(self, config: PosthogConfig):
        """Initialize PostHog client with the given configuration."""
        self.config = config
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Set up the PostHog client with configuration settings"""
        
        posthog.project_api_key = self.config.api_key
        posthog.debug = self.config.debug
        posthog.disable_geoip = self.config.disable_geoip
        
        if self.config.host:
            posthog.host = self.config.host

    def capture(self, event: TelemetryEvent) -> None:
        """
        Capture and send a telemetry event to PostHog
        
        Args:
            event: The telemetry event to capture
        """
        if not self.config.enabled:
            return
            
        try:
            posthog.capture(
                distinct_id=event.event_id,
                event=event.event_type,
                properties=event.properties
            )
        except Exception as e:
            logger.error(f"Failed to capture telemetry event: {e}")


# Initialize the telemetry client with environment-based configuration
config = PosthogConfig(
    api_key=os.getenv("POSTHOG_API_KEY"),
    enabled=os.getenv("TELEMETRY_ENABLED", "true").lower() in ("true", "1", "t"),
    debug=os.getenv("DEBUG_MODE", "false").lower() in ("true", "1", "t"),
    disable_geoip=os.getenv("TELEMETRY_ENABLED", "false").lower() in ("true", "1", "t"),
    host=os.getenv("POSTHOG_HOST")
)

telemetry_client = PosthogClient(config=config)
