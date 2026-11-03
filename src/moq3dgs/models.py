"""Pydantic data contracts for all client ↔ server messaging.

Every message crossing the MoQ transport boundary is defined here so that
serialisation / validation is centralised and enforced.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ImportanceTier(IntEnum):
    """Categorisation of Gaussians based on their spatial contribution.
    
    Subsets are ordered by their contribution to the scene's primary 
    structure (skeleton).
    """
    LARGE = 0   # Top 5% WFPS coverage
    MEDIUM = 1  # Next 15% WFPS coverage
    SMALL = 2   # Remaining 80% coverage


# ---------------------------------------------------------------------------
# Client → Server messages
# ---------------------------------------------------------------------------


class ViewportUpdate(BaseModel):
    """Client message requesting new spatial data based on movement trace.

    Sent each time the client camera moves so the server can adjust
    which tracks / groups are published with which priority.
    """

    client_id: str
    timestamp_ms: int
    camera_position: List[float] = Field(
        ..., min_length=3, max_length=3, description="[x, y, z]"
    )
    view_matrix: List[List[float]] = Field(
        ..., description="4×4 view matrix (row-major)"
    )
    fov: float = Field(..., gt=0, lt=180, description="Vertical FOV in degrees")
    fov_angles: tuple[float, float, float, float] | None = Field(
        None, description="Signed radians: left, right, down, up; preserves off-axis projection"
    )

    @field_validator("fov_angles")
    @classmethod
    def validate_fov_angles(
        cls, value: tuple[float, float, float, float] | None,
    ) -> tuple[float, float, float, float] | None:
        if value is not None:
            from moq3dgs.viewport.frustum import fov_tangents
            fov_tangents(value)
        return value


class MoQSubscription(BaseModel):
    """Subscription request from client to server for a specific track/group.

    max_object_id defines the requested quality level: 0 = base only,
    1 = base + enhancement.
    """

    track_id: str
    group_id: str
    subgroup_id: int = Field(
        ..., ge=0, le=2, description="Splat subset (0=Large, 1=Medium, 2=Small)"
    )
    quality_level: int = Field(
        0, ge=0, le=1, description="Quality level (0=Base SH0, 1=Full SH0-3)"
    )
    priority: int = Field(
        ..., ge=0, le=255, description="0=highest, 255=lowest"
    )


# ---------------------------------------------------------------------------
# Scene manifest (Server → Client)
# ---------------------------------------------------------------------------


class GroupInfo(BaseModel):
    """Metadata for a single MoQ Group (a chunk of Gaussians within a track).

    Stores the axis-aligned bounding box and the number of Gaussians so
    that the client can do local frustum culling without downloading data.
    """

    group_id: str
    num_gaussians: int
    bbox_min: List[float] = Field(..., min_length=3, max_length=3)
    bbox_max: List[float] = Field(..., min_length=3, max_length=3)
    available_subgroups: List[int] = Field(
        default=[0, 1, 2, 3, 4], description="Which Importance Tiers are available"
    )


class TrackInfo(BaseModel):
    """Metadata for a single MoQ Track (a spatial volume / octree node).

    A track contains one or more groups, each covering a sub-region of the
    volume.
    """

    track_id: str
    bbox_min: List[float] = Field(..., min_length=3, max_length=3)
    bbox_max: List[float] = Field(..., min_length=3, max_length=3)
    groups: List[GroupInfo]


class SceneManifest(BaseModel):
    """Top-level manifest for the entire 3DGS broadcast.

    Distributed once at session start; the client uses it to know which
    track/group IDs exist and where they sit in world space.
    """

    broadcast_name: str
    total_gaussians: int
    tracks: List[TrackInfo]


# ---------------------------------------------------------------------------
# Gaussian data payload
# ---------------------------------------------------------------------------


class GaussianCluster(BaseModel):
    """Serialisable payload for a batch of Gaussians sent over the wire.

    Tensor data is flattened to lists for JSON transport; the receiver
    reconstructs torch tensors from these.  For binary transport the
    ``raw_bytes`` field carries the pre-packed buffer instead.
    """

    track_id: str
    group_id: str
    subgroup_id: int = Field(
        ..., ge=0, le=4, description="Importance Tier (0-4)"
    )
    object_id: int = Field(
        ..., ge=0, description="Payload fragment sequence within the subgroup"
    )
    num_gaussians: int

    # Flattened attribute arrays (JSON path)
    means: Optional[List[float]] = None          # [N*3]
    opacities: Optional[List[float]] = None      # [N]
    sh_coeffs: Optional[List[float]] = None      # [N*C]  (C depends on LoD)
    scales: Optional[List[float]] = None         # [N*3]
    rotations: Optional[List[float]] = None      # [N*4]

    # Binary path — mutually exclusive with the lists above
    raw_bytes: Optional[bytes] = None


# ---------------------------------------------------------------------------
# Server → Client control
# ---------------------------------------------------------------------------


class SubscriptionAck(BaseModel):
    """Server acknowledgement of a subscription request."""

    track_id: str
    group_id: str
    accepted: bool
    reason: Optional[str] = None


class ServerStatus(BaseModel):
    """Periodic heartbeat from the server carrying session-level stats."""

    active_subscriptions: int
    total_tracks: int
    total_groups: int
    bytes_sent: int
    uptime_ms: int
