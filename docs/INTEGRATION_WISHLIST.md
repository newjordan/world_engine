# Overworld Integration Wishlist

Status: wishlist / idea inventory.

This document is not a results report and does not promote any item as proven.
It is a backlog of ambitious integration directions grounded in the current
spin-persistence handoff: pose-admitted temporal anchoring works in the
controlled yaw return sweep, while long-pan re-acquisition and walkable
translation remain the hard frontier.

## Wishlist

1. Multi-hypothesis loop-closure search for long-pan re-acquisition.
2. Content-defined "I know this place" detection before anchoring.
3. Pose-admitted temporal anchoring as the default memory gate.
4. Explicit no-match / failed-loop-closure state instead of silent fallback.
5. Temporal keyframe braid: build each 4-frame anchor chunk from multiple remembered views.
6. Full-pan MP4 review generator for every serious run.
7. Human-readable auto-report pages with videos, plots, source hashes, and verdicts.
8. Live browser demo with memory ON/OFF toggle.
9. Anchor confidence HUD: pose error, response, accept/reject, atlas size, fps.
10. Black-and-white surreal primitive benchmark world.
11. Sparse landmark/object permanence probes.
12. Homography+ compositor for small translation.
13. Depth-warp atlas using monodepth on generated frames.
14. Gaussian / point-cloud dream memory for walkable persistence.
15. Memory minimap rendered from the explicit atlas/geometry store.
16. Collision and navigation from remembered geometry.
17. Long-session atlas garbage collection with confidence decay.
18. CARTOGRAPHER vertical slice: finite anchors, mutating dream, get-home objective.
19. Anchor economy where remembering costs scarce resources.
20. Player-authored landmarks or "memory stamps."
21. Confidence-driven dream weather: fog, shimmer, desaturation, instability.
22. Loop-closure gameplay encounters where memory is the win condition.
23. Replay debugger for frames, candidate keyframes, admissions, rejects, and appends.
24. Gauntlet leaderboard across revisit, atlas_nowb, anchor, homography, depth, and geometry arms.
25. Condition switchboard for biome, surreal_bw, OOD primitives, and future trained worlds.
26. Failure anthology page showing drift, smear, wrong-pose anchors, rejects, and recoveries.
27. Training-data factory from failed long-pan / wrong-pose cases.
28. Memory-tolerance LoRA that tries to flip the known failed arms.
29. Register-token or adapter-based learned memory channel beyond raw KV editing.
30. Public "world persistence kit": benchmark, demo, paper draft, MP4s, and reproducible scripts.
