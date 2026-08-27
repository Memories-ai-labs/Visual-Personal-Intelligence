"""Typed DataLake errors.

The API is consistent: a failure is `{"error": {"code", "message", "request_id",
"retry_after"}}`. We map the codes we act on differently and keep `request_id` on
every error, because it is the only thing support can look up.
"""

from __future__ import annotations


class DataLakeError(RuntimeError):
    """Base class. Carries the API's own code, message and request id."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "",
        status: int = 0,
        request_id: str = "",
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.request_id = request_id
        self.retry_after = retry_after

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.code:
            parts.append(f"code={self.code}")
        if self.request_id:
            parts.append(f"request_id={self.request_id}")
        return " ".join(parts)


class AuthError(DataLakeError):
    """401 — missing or invalid key."""


class PermissionDenied(DataLakeError):
    """403 — a scoped key used outside its scope."""


class NotFound(DataLakeError):
    """404."""


class InvalidArgument(DataLakeError):
    """400 — malformed request."""


class EndpointDeprecated(DataLakeError):
    """A 400 whose message says the endpoint is retired.

    Kept separate so a retired endpoint reads as "this capability is gone",
    not "you sent a bad request" — the two demand opposite responses.
    """


class RateLimited(DataLakeError):
    """429 — respect `retry_after`."""


class VideoNotReady(DataLakeError):
    """409 video_not_ready — indexing has not finished. `retry_after` is set."""


class ServerError(DataLakeError):
    """5xx — retryable."""


def from_response(status: int, payload: dict, fallback: str) -> DataLakeError:
    """Build the right exception from an error response body."""
    err = payload.get("error") or {}
    code = str(err.get("code") or "")
    message = str(err.get("message") or fallback)
    request_id = str(err.get("request_id") or "")
    retry_after = err.get("retry_after")
    retry_after = float(retry_after) if isinstance(retry_after, (int, float)) else None
    kwargs = {
        "code": code,
        "status": status,
        "request_id": request_id,
        "retry_after": retry_after,
    }

    if status == 401:
        return AuthError(message, **kwargs)
    if status == 403:
        return PermissionDenied(message, **kwargs)
    if status == 404:
        return NotFound(message, **kwargs)
    if status == 429:
        return RateLimited(message, **kwargs)
    if status == 409:
        return VideoNotReady(message, **kwargs)
    if status >= 500:
        return ServerError(message, **kwargs)
    if status == 400 and "deprecated" in message.lower():
        return EndpointDeprecated(message, **kwargs)
    if status == 400:
        return InvalidArgument(message, **kwargs)
    return DataLakeError(message, **kwargs)
