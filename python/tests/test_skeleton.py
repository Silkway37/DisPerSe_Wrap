"""Tests for the Skeleton data model and I/O."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from disperse_wrapper.io.skeleton_io import _parse_ascii_ndskl
from disperse_wrapper.skeleton import Arc, CriticalPoint, Skeleton

# ---------------------------------------------------------------------------
# Skeleton object tests
# ---------------------------------------------------------------------------


class TestSkeleton:
    @pytest.fixture()
    def simple_skeleton(self) -> Skeleton:
        cps = [
            CriticalPoint(id=0, position=np.array([0.0, 0.0, 0.0]), cp_type=0, value=0.1),
            CriticalPoint(id=1, position=np.array([50.0, 50.0, 50.0]), cp_type=3, value=1.5),
            CriticalPoint(id=2, position=np.array([10.0, 20.0, 30.0]), cp_type=1, value=0.5),
            CriticalPoint(id=3, position=np.array([80.0, 80.0, 80.0]), cp_type=2, value=0.9),
        ]
        arc_pts = np.array([[0.0, 0.0, 0.0], [25.0, 25.0, 25.0], [50.0, 50.0, 50.0]])
        arcs = [
            Arc(id=0, node_ids=(0, 1), points=arc_pts),
            Arc(id=1, node_ids=(2, 3), points=arc_pts[:2]),
        ]
        return Skeleton(
            ndims=3,
            critical_points=cps,
            arcs=arcs,
            bbox_min=np.zeros(3),
            bbox_max=np.full(3, 100.0),
        )

    def test_n_nodes(self, simple_skeleton: Skeleton) -> None:
        assert simple_skeleton.n_nodes == 4

    def test_n_arcs(self, simple_skeleton: Skeleton) -> None:
        assert simple_skeleton.n_arcs == 2

    def test_total_length(self, simple_skeleton: Skeleton) -> None:
        assert simple_skeleton.total_length > 0

    def test_nodes_of_type(self, simple_skeleton: Skeleton) -> None:
        assert len(simple_skeleton.minima) == 1
        assert len(simple_skeleton.maxima) == 1
        assert len(simple_skeleton.saddles_1) == 1
        assert len(simple_skeleton.saddles_2) == 1

    def test_summary_str(self, simple_skeleton: Skeleton) -> None:
        s = simple_skeleton.summary()
        assert "Nodes" in s
        assert "Arcs" in s

    def test_repr(self, simple_skeleton: Skeleton) -> None:
        r = repr(simple_skeleton)
        assert "Skeleton" in r

    def test_to_numpy(self, simple_skeleton: Skeleton) -> None:
        d = simple_skeleton.to_numpy()
        assert d["cp_positions"].shape == (4, 3)
        assert d["arc_node_ids"].shape == (2, 2)
        assert d["arc_lengths"].shape == (2,)

    def test_type_name(self) -> None:
        cp = CriticalPoint(id=0, position=np.zeros(3), cp_type=3)
        assert cp.type_name == "maximum"

    def test_arc_length_computed_from_points(self) -> None:
        pts = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        arc = Arc(id=0, node_ids=(0, 1), points=pts)
        assert abs(arc.length - 5.0) < 1e-9


# ---------------------------------------------------------------------------
# ASCII .a.NDskl parsing
# ---------------------------------------------------------------------------


class TestAsciiParsing:
    def test_parse_minimal(self, minimal_skeleton_text: str) -> None:
        sk = _parse_ascii_ndskl(minimal_skeleton_text)
        assert sk.ndims == 3
        assert sk.n_nodes == 4
        assert sk.n_arcs == 2

    def test_critical_point_positions(self, minimal_skeleton_text: str) -> None:
        sk = _parse_ascii_ndskl(minimal_skeleton_text)
        pos0 = sk.critical_points[0].position
        np.testing.assert_allclose(pos0, [10.0, 20.0, 30.0])

    def test_critical_point_types(self, minimal_skeleton_text: str) -> None:
        sk = _parse_ascii_ndskl(minimal_skeleton_text)
        assert sk.critical_points[0].cp_type == 0   # minimum
        assert sk.critical_points[1].cp_type == 3   # maximum

    def test_arc_node_ids(self, minimal_skeleton_text: str) -> None:
        sk = _parse_ascii_ndskl(minimal_skeleton_text)
        assert sk.arcs[0].node_ids == (0, 1)
        assert sk.arcs[1].node_ids == (2, 3)

    def test_arc_points(self, minimal_skeleton_text: str) -> None:
        sk = _parse_ascii_ndskl(minimal_skeleton_text)
        assert sk.arcs[0].points.shape == (3, 3)

    def test_bbox(self, minimal_skeleton_text: str) -> None:
        sk = _parse_ascii_ndskl(minimal_skeleton_text)
        np.testing.assert_allclose(sk.bbox_min, [0, 0, 0])
        np.testing.assert_allclose(sk.bbox_max, [100, 100, 100])

    def test_bad_header_raises(self) -> None:
        with pytest.raises(ValueError, match="NDSKEL"):
            _parse_ascii_ndskl("WRONG_HEADER\n")


# ---------------------------------------------------------------------------
# Round-trip JSON
# ---------------------------------------------------------------------------


class TestJsonRoundTrip:
    def test_save_load(
        self,
        minimal_skeleton_text: str,
        tmp_path: Path,
    ) -> None:
        from disperse_wrapper.api import load_skeleton, save_skeleton

        sk_orig = _parse_ascii_ndskl(minimal_skeleton_text)
        out = tmp_path / "skeleton.json"
        save_skeleton(sk_orig, out, fmt="json")

        assert out.exists()
        sk_loaded = load_skeleton(out)

        assert sk_loaded.n_nodes == sk_orig.n_nodes
        assert sk_loaded.n_arcs == sk_orig.n_arcs
        np.testing.assert_allclose(
            sk_loaded.critical_points[0].position,
            sk_orig.critical_points[0].position,
        )

    def test_json_is_valid(
        self,
        minimal_skeleton_text: str,
        tmp_path: Path,
    ) -> None:
        from disperse_wrapper.api import save_skeleton

        sk = _parse_ascii_ndskl(minimal_skeleton_text)
        out = tmp_path / "sk.json"
        save_skeleton(sk, out, fmt="json")
        data = json.loads(out.read_text())
        assert "critical_points" in data
        assert "arcs" in data


# ---------------------------------------------------------------------------
# Round-trip NPZ
# ---------------------------------------------------------------------------


class TestNpzRoundTrip:
    def test_save_load(
        self,
        minimal_skeleton_text: str,
        tmp_path: Path,
    ) -> None:
        from disperse_wrapper.api import load_skeleton, save_skeleton

        sk_orig = _parse_ascii_ndskl(minimal_skeleton_text)
        out = tmp_path / "skeleton.npz"
        save_skeleton(sk_orig, out, fmt="npz")

        assert out.exists()
        sk_loaded = load_skeleton(out)

        assert sk_loaded.n_nodes == sk_orig.n_nodes
        assert sk_loaded.n_arcs == sk_orig.n_arcs


# ---------------------------------------------------------------------------
# NetworkX export (skip if not installed)
# ---------------------------------------------------------------------------


class TestNetworkX:
    def test_to_networkx(self, minimal_skeleton_text: str) -> None:
        pytest.importorskip("networkx")
        sk = _parse_ascii_ndskl(minimal_skeleton_text)
        G = sk.to_networkx()
        assert G.number_of_nodes() == sk.n_nodes
        assert G.number_of_edges() == sk.n_arcs

    def test_to_networkx_missing_raises(
        self,
        minimal_skeleton_text: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import sys

        monkeypatch.setitem(sys.modules, "networkx", None)
        sk = _parse_ascii_ndskl(minimal_skeleton_text)
        with pytest.raises(ImportError, match="networkx"):
            sk.to_networkx()
