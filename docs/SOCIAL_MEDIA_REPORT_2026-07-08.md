# Overworld Persistence Update - Social Report

**Date:** 2026-07-08
**Status:** public-facing progress note
**Media package:** `bench_out/social_2026-07-08/`

## Share Video

- Vertical: `bench_out/social_2026-07-08/overworld_turnaround_vertical_1080x1920.mp4`
- Landscape: `bench_out/social_2026-07-08/overworld_turnaround_landscape_1080p.mp4`
- Preview still: `bench_out/social_2026-07-08/overworld_turnaround_preview.jpg`
- Stills sheet: `bench_out/social_2026-07-08/overworld_turnaround_stills.png`

Recommended use:

- Use the vertical `1080x1920` export for Reels, Shorts, TikTok, and phone-first
  feeds.
- Use the landscape `1920x1080` export for YouTube, LinkedIn, X, and desktop-first
  posts.

The share exports are presentation wrappers around the recorded 9.6 second,
30 fps turnaround clip. The original source video is
`bench_out/warm/turnaround_360/atlas_before_after_h264.mp4`.

## Short Caption

Overworld persistence update: turning a generated scene into a mapped world.

The system keeps a pose-indexed visual memory, registers the current view against
that memory, and uses it to hold structure as the camera returns to a known place.
The next step is temporal anchoring: feeding four-frame memory chunks back through
the model's normal video input path.

Goal: a live, player-controlled world that stays recognizable as you look away,
return, and keep exploring.

## Longer Writeup

We are building permanence for interactive video world models.

The current prototype treats the model as a real-time renderer and adds a memory
layer around it. As the camera moves, the system records pose-keyed visual
keyframes, registers the current frame against the closest remembered view, and
uses that memory to stabilize what the player sees when they return.

The concept is simple: generated worlds need a map. The map does not replace the
model; it gives the model a persistent external structure to refer to.

Recent progress focused on three pieces:

1. A pose-indexed atlas for storing remembered views.
2. Registration and confidence scoring for matching a live frame back to memory.
3. Front-door anchoring experiments that feed reconstructed memory frames through
   the model's normal video input path.

The newest engineering step is a four-frame temporal anchor. Instead of repeating
one reconstructed frame, the next runner builds a short memory-consistent video
chunk. That is closer to how the model naturally receives visual context.

What we want to see next: a live browser demo where you spin away, come back, and
the scene still reads as the same place. From there, the path is a small permanence
gauntlet: spin, glance, patrol, then translation.

## Concepts Applied

- **Pose-indexed memory:** store views by camera heading so a returned pose can
  retrieve the right remembered frame.
- **Visual registration:** align remembered content to the current view before
  using it.
- **Confidence-gated compositing:** apply memory where the match is strong and
  keep the live model output where it is not.
- **Front-door anchoring:** feed reconstructed memory through the same image/video
  path the model already expects.
- **Temporal anchoring:** represent memory as a four-frame chunk, not a static
  repeated frame.

## Where This Is Headed

- A shareable live demo with a memory toggle.
- A compact permanence benchmark: spin, glance, patrol, and later out-and-back
  movement.
- Geometry-aware memory for translation, using depth or point-cloud style world
  state when pure rotation is no longer enough.
- A game prototype where mapping the world becomes the core mechanic.

## Suggested Post Text

Building persistence for interactive video world models.

This clip shows the current direction: generated scenes backed by a pose-indexed
visual memory layer. The model keeps rendering; the memory system gives the world a
map.

Next target: live permanence in the browser, then a small gauntlet for spin,
glance, patrol, and movement.

## Provenance

- Source video: `bench_out/warm/turnaround_360/atlas_before_after_h264.mp4`
- Source review page: `bench_out/warm/turnaround_360/index.html`
- Current controlled summary: `docs/ANCHOR_RESULTS.md`
- Next condition plan: `docs/ANCHOR4_PLAN.md`
- This is a social summary and media package, not a new benchmark result.
