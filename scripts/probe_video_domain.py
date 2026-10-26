"""Probe the video / computational-imaging domain of the primitive matcher.

Checks (1) every video solution strips back to its own intended primitive
(self-describing check) and (2) novel, rephrased user queries retrieve a
structurally sensible solution. Run from the repo root:

    .venv/bin/python scripts/probe_video_domain.py
"""

from __future__ import annotations

import sys

from super_solver.vectorize.domain_knowledge import DOMAIN_KNOWLEDGE
from super_solver.vectorize.primitives import match_by_primitives, strip_to_primitives

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

# (novel query, expected primitive that a sensible top match should carry)
PROBES = [
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


def main() -> int:
    sources = [{"title": e["problem_context"][:60], "solution": e["solution_direction"]} for e in DOMAIN_KNOWLEDGE]
    print(f"sources: {len(sources)}")

    print("\n--- self-describing check (video solutions) ---")
    bad = []
    video_entries = [e for e in DOMAIN_KNOWLEDGE if strip_to_primitives(e["problem_context"]) & VIDEO_ACTIONS]
    for e in video_entries:
        prims = strip_to_primitives(e["solution_direction"])
        acts = sorted(p for p in prims if p in VIDEO_ACTIONS)
        ok = "object_video" in prims and acts
        if not ok:
            bad.append(e["solution_direction"][:70])
        print(f"{'OK ' if ok else 'BAD'} {acts} obj_video={'object_video' in prims} :: {e['solution_direction'][:70]}")
    print(f"video entries: {len(video_entries)}  not-self-describing: {len(bad)}")

    print("\n--- novel rephrased probes ---")
    correct = 0
    for q, expected in PROBES:
        hits = match_by_primitives(q, sources, top_k=3)
        top = hits[0]
        prims = set(top["source_primitives"])
        ok = expected in prims
        correct += ok
        print(
            f"\nQ: {q}\n   target prims : {sorted(strip_to_primitives(q))}\n"
            f"   top score    : {top['primitive_overlap']:.3f}  expected={expected} {'HIT' if ok else 'MISS'}\n"
            f"   top solution : {top['content'][:130]}\n"
            f"   runner-up    : {hits[1]['primitive_overlap']:.3f} {hits[1]['content'][:90]}"
        )
    print(f"\nprobes: {correct}/{len(PROBES)} top-1 structurally correct")
    return 0 if correct == len(PROBES) and not bad else 1


if __name__ == "__main__":
    sys.exit(main())
