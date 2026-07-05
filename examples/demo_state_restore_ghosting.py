# uv run --dev python examples/demo_state_restore_ghosting.py   (needs a seed in bench_assets/)
"""Visualize the get_state/load_state VAE-state fix: ghosting at a snapshot/restore
boundary, BEFORE (kv+frame_ts only, old behavior) vs AFTER (full restore incl VAE).

Both restores decode the IDENTICAL latent from the IDENTICAL restored KV + seed, so the
only difference is the TAEHV streaming-decoder memory -> the ghosting is isolated.
"""
import glob
import sys
import numpy as np
import torch
import cv2

sys.path.insert(0, "examples")
from permanence_bench import psnr, ssim  # reuse tested metrics
from world_engine import WorldEngine, CtrlInput

SEED = 1234
eng = WorldEngine("Overworld/Waypoint-1.5-1B", device="cuda")

bgr = cv2.imread(sorted(glob.glob("bench_assets/*.png"))[0])
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
seed_x4 = torch.from_numpy(np.repeat(rgb[None], 4, 0)).cuda()

def last(four):
    return four[-1].detach().cpu().numpy()

# 1) settle a scene, snapshot it
torch.manual_seed(SEED)
eng.reset()
eng.append_frame(seed_x4)
snap = None
for _ in range(8):
    snap = last(eng.gen_frame(ctrl=CtrlInput()))
S = eng.get_state()                              # kv + frame_ts + vae

# 2) branch: hard turn -> fills KV + streaming decoder with very different imagery
branch = None
for _ in range(16):
    branch = last(eng.gen_frame(ctrl=CtrlInput(mouse=[0.5, 0.0])))

# 3a) BEFORE fix: restore kv + frame_ts only (old buggy load_state) -> stale decoder
eng.kv_cache.load_state(S["kv_cache"])
eng.frame_ts.copy_(S["frame_ts"])
torch.manual_seed(SEED)
before = last(eng.gen_frame(ctrl=CtrlInput()))

# 3b) AFTER fix: full restore (kv + frame_ts + vae) -> identical latent, clean decode
eng.load_state(S)
torch.manual_seed(SEED)
after = last(eng.gen_frame(ctrl=CtrlInput()))

# metrics
print(f"snapshot vs BEFORE  : PSNR {psnr(snap, before):5.2f}  SSIM {ssim(snap, before):.3f}")
print(f"snapshot vs AFTER   : PSNR {psnr(snap, after):5.2f}  SSIM {ssim(snap, after):.3f}")
print(f"BEFORE vs AFTER      : PSNR {psnr(before, after):5.2f}  SSIM {ssim(before, after):.3f}  (the ghosting delta)")
print(f"branch vs BEFORE     : PSNR {psnr(branch, before):5.2f}  (lower = more branch bleed)")
print(f"branch vs AFTER      : PSNR {psnr(branch, after):5.2f}")

# contact sheet
def panel(img, label):
    p = cv2.cvtColor(img, cv2.COLOR_RGB2BGR).copy()
    for c, th in [((0, 0, 0), 7), ((255, 255, 255), 2)]:
        cv2.putText(p, label, (24, 56), cv2.FONT_HERSHEY_SIMPLEX, 1.5, c, th)
    return p

sheet = np.concatenate([
    panel(snap,   "1. snapshot (restore target)"),
    panel(branch, "2. branch end (discarded)"),
    panel(before, "3. BEFORE fix (ghosted)"),
    panel(after,  "4. AFTER fix (clean)"),
], axis=1)
sheet = cv2.resize(sheet, None, fx=0.5, fy=0.5)
cv2.imwrite("bench_out/ghosting_before_after.png", sheet)
print("wrote bench_out/ghosting_before_after.png")
