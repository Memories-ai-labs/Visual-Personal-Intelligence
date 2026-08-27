"""DataLake response models.

`extra="allow"` everywhere on purpose: the API adds fields as it grows and the
docs say callers must tolerate that. We never assert a field we did not read.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class _Model(BaseModel):
    model_config = ConfigDict(extra="allow")


class Collection(_Model):
    id: str
    name: str = ""
    video_count: int = 0
    face_recognition_enabled: bool = False
    enabled_detectors: list[str] = Field(default_factory=list)
    created_at: datetime | None = None


class Video(_Model):
    id: str
    collection_id: str = ""
    status: str = ""
    duration: float | None = None
    captured_at: datetime | None = None
    metadata: dict = Field(default_factory=dict)

    @property
    def title(self) -> str:
        return str(self.metadata.get("title") or self.id)

    @property
    def is_ready(self) -> bool:
        return self.status == "ready"


class Operation(_Model):
    operation: str
    kind: str = ""
    done: bool = False
    cancelled: bool = False
    resource: str = ""
    progress: dict = Field(default_factory=dict)
    error: dict | str | None = None

    @property
    def failed(self) -> bool:
        """A non-null error means failure — including partial failure."""
        return self.error is not None


class SearchHit(_Model):
    """One moment hit.

    `score` is only comparable against other hits from the *same* request: the
    API computes it as cosine / ts_rank / RRF / sigmoid depending on the path.
    `request_id` is carried so downstream code can enforce that.
    """

    ref: str
    video_id: str = ""
    target: str = ""
    score: float = 0.0
    start: float = 0.0
    end: float = 0.0
    snippet: str = ""
    thumbnail_url: str = ""
    request_id: str = ""


class SearchPage(_Model):
    results: list[SearchHit] = Field(default_factory=list)
    next_cursor: str | None = None
    hint: str = ""
    request_id: str = ""


class Segment(_Model):
    """A caption or transcription span."""

    start: float = 0.0
    end: float = 0.0
    text: str = ""
    speaker_id: str | None = None


class Frame(_Model):
    t: float = 0.0
    url: str = ""


class Speaker(_Model):
    label: str = ""
    segments: int = 0
    speaker_id: str | None = None


class Moment(_Model):
    """A time slice of one video, with whatever was expanded."""

    ref: str
    video_id: str = ""
    start: float = 0.0
    end: float = 0.0
    caption: list[Segment] = Field(default_factory=list)
    transcription: list[Segment] = Field(default_factory=list)
    frames: list[Frame] = Field(default_factory=list)
    clip_url: str = ""
    speakers: list[Speaker] = Field(default_factory=list)
