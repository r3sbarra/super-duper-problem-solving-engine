"""Regression guard for the video / computational-imaging domain (wave 3).

Locks in three properties of the new domain:
1. Every video solution_direction is SELF-DESCRIBING: it strips back to its own
   action primitive plus object_video (SKILL.md step 3). A solution that strips
   to empty can never be retrieved.
2. The 11 new action primitives + object_video exist and fire on their cue.
3. Novel, rephrased user-style queries (not string matches of problem_context)
   retrieve a top-1 solution that carries the structurally expected primitive.
"""
from __future__ import annotations

import pytest

from super_solver.vectorize.domain_knowledge import DOMAIN_KNOWLEDGE
from super_solver.vectorize.primitives import (
    ACTION_PRIMITIVES,
    OBJECT_PRIMITIVES,
    match_by_primitives,
    strip_to_primitives,
)

VIDEO_ACTIONS = {
    "warp_by_flow",
    "propagate_from_keyframe",
    "decompose_low_rank",
    "canonicalise_to_uv",
    "warp_piecewise_affine",
    "blend_temporal_overlap",
    "allocate_by_roi",
    "reuse_cached_features",
    "downsample_pyramid",
    "quantise_model",
    "stream_ring_buffer",
}

# (novel rephrased query, primitive a sensible top-1 must carry)
NOVEL_PROBES = [
    ("apply the dance moves and clothes from one clip onto a video of my friend", "warp_by_flow"),
    ("make a 4 hour video cheaper to process on weak hardware", "quantise_model"),
    ("stop a per-frame render from flickering", "blend_temporal_overlap"),
    ("take the outfit from clip A and put it on the person in clip B", "canonicalise_to_uv"),
    ("only re-render one frame in ten and fill the rest in", "propagate_from_keyframe"),
    ("cut the moving person out of a static shot without a green screen", "decompose_low_rank"),
    ("the expensive network runs on every frame and it is too slow", "reuse_cached_features"),
    ("the whole clip does not fit in GPU memory", "stream_ring_buffer"),
    ("warping with one homography looks wrong because the arms move differently", "warp_piecewise_affine"),
    ("spend the GPU time on the face and ignore the empty wall", "allocate_by_roi"),
]


def _sources():
    return [
        {"title": e["problem_context"][:60], "solution": e["solution_direction"]}
        for e in DOMAIN_KNOWLEDGE
    ]


def test_video_primitives_registered():
    actions = {p for _, p in ACTION_PRIMITIVES}
    objects = {p for _, p in OBJECT_PRIMITIVES}
    assert VIDEO_ACTIONS <= actions, VIDEO_ACTIONS - actions
    assert "object_video" in objects


@pytest.mark.parametrize("cue,expected", [
    ("optical flow warping", "warp_by_flow"),
    ("sparse keyframes are propagated", "propagate_from_keyframe"),
    ("low-rank plus sparse residual decomposition", "decompose_low_rank"),
    ("unwrap to a UV texture atlas", "canonicalise_to_uv"),
    ("piecewise-affine mesh warp", "warp_piecewise_affine"),
    ("motion-compensated temporal filtering", "blend_temporal_overlap"),
    ("allocate compute by ROI tile", "allocate_by_roi"),
    ("reuse cached features", "reuse_cached_features"),
    ("downsample pyramid coarse-to-fine", "downsample_pyramid"),
    ("int8 quantisation and distillation", "quantise_model"),
    ("bounded ring buffer streaming pipeline", "stream_ring_buffer"),
])
def test_primitive_cues_fire(cue, expected):
    assert expected in strip_to_primitives(cue)


def test_video_solutions_are_self_describing():
    # A VIDEO entry is one whose problem_context strips to a video action.
    # Only those solutions are required to carry object_video: `downsample_pyramid`
    # is a shared signal-processing atom also used by non-video math entries,
    # so it cannot be the test for membership.
    video_entries = [
        e for e in DOMAIN_KNOWLEDGE
        if strip_to_primitives(e["problem_context"]) & VIDEO_ACTIONS
    ]
    bad = []
    for entry in video_entries:
        prims = strip_to_primitives(entry["solution_direction"])
        if not (prims & VIDEO_ACTIONS):
            bad.append(("solution lacks a video action", entry["solution_direction"][:60]))
        if "object_video" not in prims:
            bad.append(("missing object_video", entry["solution_direction"][:60]))
    assert not bad, bad
    assert len(video_entries) >= 12, f"only {len(video_entries)} video entries"


@pytest.mark.parametrize("query,expected", NOVEL_PROBES)
def test_novel_video_probe_retrieves_expected_structure(query, expected):
    hits = match_by_primitives(query, _sources(), top_k=3)
    top = hits[0]
    assert not top.get("empty_match"), f"query stripped to nothing: {query}"
    assert expected in top["source_primitives"], (
        f"{query!r}: top match lacked {expected}; "
        f"prims={top['source_primitives']} solution={top['content'][:80]!r}"
    )


def test_non_video_probe_does_not_pick_video_entry():
    """A maths query must not be hijacked by the new object_video type."""
    hits = match_by_primitives("prove there are infinitely many primes", _sources(), top_k=1)
    assert not (set(hits[0]["source_primitives"]) & VIDEO_ACTIONS)
