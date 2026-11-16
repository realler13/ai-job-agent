import enum
import logging
import threading
import uuid
from typing import Optional, Callable

logger = logging.getLogger(__name__)


class SessionPhase(enum.Enum):
    IDLE = "IDLE"
    SCRAPING = "SCRAPING"
    FILTERING = "FILTERING"
    GENERATING = "GENERATING"
    POSTPROCESSING = "POSTPROCESSING"
    DONE = "DONE"
    ERROR = "ERROR"


class SessionManager:
    """Typed session lifecycle manager replacing raw control_flags dicts."""

    def __init__(self, session_id: Optional[str] = None):
        self.id = session_id or str(uuid.uuid4())
        self._phase = SessionPhase.IDLE
        self._progress = 0.0
        self._message = ""
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._pause_event.set()  # Not paused by default
        self._lock = threading.Lock()

    @property
    def phase(self) -> SessionPhase:
        with self._lock:
            return self._phase

    @property
    def progress(self) -> float:
        with self._lock:
            return self._progress

    @property
    def message(self) -> str:
        with self._lock:
            return self._message

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    @property
    def is_paused(self) -> bool:
        return not self._pause_event.is_set()

    def update_progress(self, phase: str, pct: float, msg: str = ""):
        """Update session progress. phase can be a string or SessionPhase name."""
        with self._lock:
            try:
                self._phase = SessionPhase(phase)
            except ValueError:
                pass  # Keep current phase if invalid
            self._progress = max(0.0, min(1.0, float(pct)))
            self._message = msg

    def check_stop(self) -> bool:
        """Returns True if stop has been requested."""
        return self._stop_event.is_set()

    def wait_if_paused(self):
        """Blocks until unpaused. Returns immediately if not paused."""
        self._pause_event.wait()

    def request_stop(self):
        """Signal the session to stop."""
        self._stop_event.set()
        self._pause_event.set()  # Unpause so the thread can exit

    def toggle_pause(self):
        """Toggle pause state."""
        if self._pause_event.is_set():
            self._pause_event.clear()  # Pause
        else:
            self._pause_event.set()  # Unpause

    def as_control_flags(self) -> dict:
        """Legacy compatibility: return a dict-like view for old code paths."""
        return {
            "stop": self._stop_event.is_set(),
            "pause": not self._pause_event.is_set(),
        }

    def as_progress_callback(self) -> Callable:
        """Return a progress_callback compatible with the existing pipeline."""
        def callback(text: str, percent: float):
            # Infer phase from text
            text_lower = text.lower()
            if "scrap" in text_lower:
                phase = SessionPhase.SCRAPING.value
            elif "filter" in text_lower or "ai" in text_lower:
                phase = SessionPhase.FILTERING.value
            elif "generat" in text_lower or "saving" in text_lower or "tailor" in text_lower:
                phase = SessionPhase.GENERATING.value
            elif "email" in text_lower or "sheet" in text_lower:
                phase = SessionPhase.POSTPROCESSING.value
            elif "complete" in text_lower or "done" in text_lower:
                phase = SessionPhase.DONE.value
            elif "error" in text_lower or "fail" in text_lower:
                phase = SessionPhase.ERROR.value
            else:
                phase = self._phase.value
            self.update_progress(phase, percent, text)
        return callback
