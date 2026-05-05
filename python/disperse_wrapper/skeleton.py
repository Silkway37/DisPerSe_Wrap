"""Skeleton data model for DisPerSE output."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class CriticalPoint:
    """A critical point (node) in the Morse-Smale complex.

    Attributes
    ----------
    id : int
        Zero-based index.
    position : np.ndarray, shape (ndims,)
        Spatial coordinates.
    cp_type : int
        Critical point type (0 = minimum, 1 = 1-saddle, 2 = 2-saddle, 3 = maximum in 3-D).
    value : float
        Scalar field value at this point.
    pair_id : int
        Id of the paired critical point (persistence pairing); -1 if unpaired.
    is_boundary : bool
        Whether the point lies on the domain boundary.
    arc_ids : list[int]
        Ids of arcs incident to this critical point.
    """

    id: int
    position: np.ndarray
    cp_type: int = -1
    value: float = 0.0
    pair_id: int = -1
    is_boundary: bool = False
    arc_ids: list[int] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def type_name(self) -> str:
        """Human-readable type name."""
        names = {0: "minimum", 1: "1-saddle", 2: "2-saddle", 3: "maximum"}
        return names.get(self.cp_type, f"type-{self.cp_type}")


@dataclass
class Arc:
    """A filament (arc) connecting two critical points.

    Attributes
    ----------
    id : int
        Zero-based index.
    node_ids : tuple[int, int]
        Ids of the two endpoint critical points ``(from, to)``.
    points : np.ndarray, shape (n_samples, ndims)
        Polyline sample positions along the arc.
    persistence : float | None
        Persistence significance of this arc (if available).
    length : float | None
        Arc length computed as the sum of segment lengths.
    meta : dict
        Additional key/value metadata (e.g. extra scalar statistics).
    """

    id: int
    node_ids: tuple[int, int]
    points: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))
    persistence: float | None = None
    length: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.length is None and len(self.points) > 1:
            diffs = np.diff(self.points, axis=0)
            self.length = float(np.sum(np.linalg.norm(diffs, axis=1)))


@dataclass
class Skeleton:
    """The full DisPerSE skeleton output.

    Parameters
    ----------
    ndims : int
        Spatial dimensionality (typically 3).
    critical_points : list[CriticalPoint]
        All critical points (nodes).
    arcs : list[Arc]
        All filaments (arcs).
    bbox_min, bbox_max : np.ndarray or None
        Domain bounding box.
    comment : str
        Creation info / comment string embedded by DisPerSE.
    """

    ndims: int = 3
    critical_points: list[CriticalPoint] = field(default_factory=list)
    arcs: list[Arc] = field(default_factory=list)
    bbox_min: np.ndarray | None = None
    bbox_max: np.ndarray | None = None
    comment: str = ""

    # ------------------------------------------------------------------
    # Basic statistics
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        """Total number of critical points."""
        return len(self.critical_points)

    @property
    def n_arcs(self) -> int:
        """Total number of arcs."""
        return len(self.arcs)

    @property
    def total_length(self) -> float:
        """Sum of all arc lengths."""
        return sum(a.length for a in self.arcs if a.length is not None)

    # ------------------------------------------------------------------
    # Node-type helpers
    # ------------------------------------------------------------------

    def nodes_of_type(self, cp_type: int) -> list[CriticalPoint]:
        """Return critical points of a given type."""
        return [cp for cp in self.critical_points if cp.cp_type == cp_type]

    @property
    def minima(self) -> list[CriticalPoint]:
        return self.nodes_of_type(0)

    @property
    def maxima(self) -> list[CriticalPoint]:
        return self.nodes_of_type(3)

    @property
    def saddles_1(self) -> list[CriticalPoint]:
        return self.nodes_of_type(1)

    @property
    def saddles_2(self) -> list[CriticalPoint]:
        return self.nodes_of_type(2)

    # ------------------------------------------------------------------
    # NumPy array helpers
    # ------------------------------------------------------------------

    def to_numpy(
        self,
    ) -> dict[str, np.ndarray]:
        """Return a dict of arrays representing the skeleton.

        Returns
        -------
        dict with keys:
            ``cp_positions``, ``cp_types``, ``cp_values``,
            ``arc_node_ids``, ``arc_lengths``
        """
        cp_pos = (
            np.array([cp.position for cp in self.critical_points])
            if self.critical_points
            else np.empty((0, self.ndims))
        )
        cp_types = np.array([cp.cp_type for cp in self.critical_points], dtype=np.int32)
        cp_vals = np.array([cp.value for cp in self.critical_points], dtype=np.float64)
        arc_nids = (
            np.array([arc.node_ids for arc in self.arcs], dtype=np.int32)
            if self.arcs
            else np.empty((0, 2), dtype=np.int32)
        )
        arc_lens = np.array(
            [arc.length if arc.length is not None else np.nan for arc in self.arcs],
            dtype=np.float64,
        )
        return {
            "cp_positions": cp_pos,
            "cp_types": cp_types,
            "cp_values": cp_vals,
            "arc_node_ids": arc_nids,
            "arc_lengths": arc_lens,
        }

    # ------------------------------------------------------------------
    # NetworkX export (optional dependency)
    # ------------------------------------------------------------------

    def to_networkx(self):  # type: ignore[return]
        """Return a ``networkx.Graph`` of the skeleton.

        Requires ``networkx`` (``pip install disperse-wrapper[graph]``).

        Nodes carry the ``CriticalPoint`` object as the ``"cp"`` attribute.
        Edges carry the ``Arc`` object as the ``"arc"`` attribute.
        """
        try:
            import networkx as nx
        except ImportError as exc:
            raise ImportError(
                "networkx is required for to_networkx().  "
                "Install it with: pip install disperse-wrapper[graph]"
            ) from exc

        G = nx.Graph()
        for cp in self.critical_points:
            G.add_node(cp.id, cp=cp, pos=cp.position.tolist(), cp_type=cp.cp_type)
        for arc in self.arcs:
            G.add_edge(*arc.node_ids, arc=arc, length=arc.length)
        return G

    # ------------------------------------------------------------------
    # String repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"Skeleton(ndims={self.ndims}, "
            f"n_nodes={self.n_nodes}, "
            f"n_arcs={self.n_arcs}, "
            f"total_length={self.total_length:.3g})"
        )

    def summary(self) -> str:
        """Multi-line human-readable summary."""
        lines = [
            "DisPerSE Skeleton",
            f"  Dimensions : {self.ndims}",
            f"  Nodes      : {self.n_nodes}",
            f"  Arcs       : {self.n_arcs}",
            f"  Total len  : {self.total_length:.4g}",
        ]
        if self.bbox_min is not None:
            lines.append(
                f"  BBox       : {self.bbox_min.tolist()} → {self.bbox_max.tolist()}"
            )
        type_counts = {}
        for cp in self.critical_points:
            type_counts[cp.cp_type] = type_counts.get(cp.cp_type, 0) + 1
        for t, cnt in sorted(type_counts.items()):
            names = {0: "minima", 1: "1-saddles", 2: "2-saddles", 3: "maxima"}
            label = names.get(t, f"type-{t}")
            lines.append(f"    {label:12s}: {cnt}")
        return "\n".join(lines)
