"""
Short-Term Session Memory Manager for ContextIQ.

Maintains conversational state across turns within an active session,
specifically tracking pending single-shot disambiguation/HITL options
so that follow-up selections (e.g., 'A', '1', or 'password') can be resolved
deterministically without entering conversational loops.
"""
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List


@dataclass
class SessionState:
    """State stored for an active conversation session."""
    session_id: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    pending_disambiguation: Optional[Dict[str, Any]] = None
    turn_count: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)

    def touch(self) -> None:
        """Updates last accessed timestamp and increments turn count."""
        self.last_accessed = time.time()
        self.turn_count += 1


class SessionManager:
    """Thread-safe, in-memory session manager with TTL expiration."""

    def __init__(self, default_ttl_seconds: int = 1800):
        self.default_ttl = default_ttl_seconds
        self._sessions: Dict[str, SessionState] = {}
        self._lock = threading.Lock()

    def get_or_create_session(self, session_id: Optional[str] = None) -> SessionState:
        """
        Retrieves an existing session by ID or creates a new one.
        If session_id is None or empty, generates a new unique ID.
        """
        with self._lock:
            # Periodic cleanup of expired sessions
            self._cleanup_expired_locked()

            if session_id and session_id in self._sessions:
                session = self._sessions[session_id]
                session.touch()
                return session

            new_id = session_id if session_id else uuid.uuid4().hex[:12]
            session = SessionState(session_id=new_id)
            self._sessions[new_id] = session
            return session

    def get_session(self, session_id: str) -> Optional[SessionState]:
        """Retrieves session if it exists and is not expired."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                if (time.time() - session.last_accessed) > self.default_ttl:
                    del self._sessions[session_id]
                    return None
                session.touch()
            return session

    def set_pending_disambiguation(
        self,
        session_id: str,
        original_query: str,
        options: List[Dict[str, Any]],
    ) -> None:
        """Stores pending disambiguation options for single-shot follow-up resolution."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.pending_disambiguation = {
                    "original_query": original_query,
                    "options": options,
                    "timestamp": time.time(),
                }
                session.touch()

    def clear_pending(self, session_id: str) -> None:
        """Clears pending disambiguation state from a session."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.pending_disambiguation = None

    def _cleanup_expired_locked(self) -> int:
        """Removes sessions that have exceeded the TTL (must be called with _lock held)."""
        now = time.time()
        expired = [
            sid for sid, s in self._sessions.items()
            if (now - s.last_accessed) > self.default_ttl
        ]
        for sid in expired:
            del self._sessions[sid]
        return len(expired)

    def active_session_count(self) -> int:
        """Returns the number of active sessions."""
        with self._lock:
            return len(self._sessions)
