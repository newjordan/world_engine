# uv run --dev python examples/anchor_visual.py --out bench_out/anchor/visual_review
"""
Frames-first visual review packet for Phase 10 anchoring.

This reruns selected trials under the Phase 10 condition and saves raw frames plus
annotated contact sheets. It is a visual review artifact, not a replacement for the
n=6 aggregate CSV produced by anchor_probe.py.
"""
import argparse
import csv
import hashlib
import html
import json
import os
import pathlib
import shlex
import sys

sys.path.insert(0, "examples")

import cv2
import numpy as np
import torch

from anchor import (AnchorStore, blend_u8, confidence_alpha, micro_yaw_grid,
                    reconstruct_from_keyframe, reconstruct_temporal_batch,
                    repeat_as_x4)
from atlas import AtlasStore, _clear_pins
from permanence_bench import RealEngine, _noop, load_seeds, psnr, ssim
from rigid import RigidStore, band_mask, rigid_step


ANCHOR_ARMS = {
    "anchor", "anchor_blend", "anchor_k1", "anchor_k4",
    "anchor4_linear", "anchor4_blend", "anchor4_k4",
    "anchor_full", "anchor_full_blend", "anchor_confmap", "anchor_full_far",
    "anchor4_full", "anchor4_full_blend", "anchor4_full_far",
}
ATLAS_ARMS = {"atlas_nowb"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _sha256(path):
    try:
        data = pathlib.Path(path).read_bytes()
    except OSError:
        return "missing"
    return hashlib.sha256(data).hexdigest()[:16]


def _cadence(arm):
    if arm == "anchor_k1":
        return 1
    if arm in ("anchor_k4", "anchor4_k4"):
        return 4
    return None


def _is_anchor4(arm):
    return arm.startswith("anchor4_")


def _retrieve_mode(arm):
    return "farthest" if arm.endswith("_far") else "nearest"


def _mask_mode(arm):
    if arm.startswith("anchor_full") or arm.startswith("anchor4_full"):
        return "full"
    if arm == "anchor_confmap":
        return "confmap"
    return "band"


def _summarize_micro_infos(micro_infos, anchor_count):
    projected = [i for i in micro_infos if i.get("projected")]
    resp = [i["resp"] for i in micro_infos if i.get("resp") is not None]
    alpha = [i.get("alpha", 0.0) for i in projected]
    all_projected = len(projected) == len(micro_infos)
    return {
        "projected": bool(projected),
        "dx_px": float(np.mean([i["dx_px"] for i in projected])) if projected else None,
        "resp": float(np.mean(resp)) if resp else None,
        "reject": None if all_projected else ("empty" if not projected else "partial"),
        "anchor": bool(projected),
        "alpha": float(np.mean(alpha)) if alpha else 0.0,
        "anchor_count": anchor_count,
        "micro_infos": micro_infos,
    }


def _append_anchor(eng, store, current_rgb, arm, args, anchor_count,
                   current_x4=None, target_yaws=None):
    kf = store.query(float(eng.engine.camera_yaw), mode=_retrieve_mode(arm))
    info = {"projected": False, "dx_px": None, "resp": None, "reject": "empty",
            "anchor": False, "alpha": 0.0, "anchor_count": anchor_count}
    if kf is None:
        return current_rgb, info

    if _is_anchor4(arm):
        if current_x4 is None:
            current_x4 = repeat_as_x4(current_rgb)
        if target_yaws is None:
            target_yaws = [float(eng.engine.camera_yaw)] * len(current_x4)
        recon_x4, micro_infos = reconstruct_temporal_batch(
            store, current_x4, target_yaws,
            retrieve=_retrieve_mode(arm),
            blend=(arm in ("anchor4_blend", "anchor4_full_blend")),
            resp_min=args.resp_min,
            resp_full=args.resp_full,
            max_shift_frac=args.max_shift_frac,
            feather=args.pixel_feather,
            mask_mode=_mask_mode(arm),
        )
        if not any(i.get("projected") for i in micro_infos):
            info = _summarize_micro_infos(micro_infos, anchor_count)
            info.update(anchor=False)
            return current_rgb, info

        ctrl = eng._CtrlInput(button=set(), mouse=(0.0, 0.0))
        four = eng.engine.append_frame(torch.from_numpy(recon_x4).to(eng.device), ctrl=ctrl)
        appended = eng.last_rgb(four)
        anchor_count += 1
        info = _summarize_micro_infos(micro_infos, anchor_count)
        info.update(anchor=True)
        return appended, info

    recon, info = reconstruct_from_keyframe(
        kf, current_rgb,
        resp_min=args.resp_min,
        max_shift_frac=args.max_shift_frac,
        feather=args.pixel_feather,
        mask_mode=_mask_mode(arm),
    )
    if not info["projected"]:
        info.update(anchor=False, alpha=0.0, anchor_count=anchor_count)
        return current_rgb, info

    alpha = 1.0
    if arm in ("anchor_blend", "anchor_full_blend"):
        alpha = confidence_alpha(info.get("resp"), args.resp_min, args.resp_full)
        recon = blend_u8(recon, current_rgb, alpha)

    ctrl = eng._CtrlInput(button=set(), mouse=(0.0, 0.0))
    four = eng.engine.append_frame(torch.from_numpy(repeat_as_x4(recon)).to(eng.device), ctrl=ctrl)
    appended = eng.last_rgb(four)
    anchor_count += 1
    info.update(anchor=True, alpha=alpha, anchor_count=anchor_count)
    return appended, info


def run_arm(eng, seed_x4, seed, arm, args):
    is_anchor = arm in ANCHOR_ARMS
    is_atlas = arm in ATLAS_ARMS
    H = args.K // 2
    torch.manual_seed(seed)
    eng.reset()
    eng.append_seed(seed_x4)

    anchor_store = AnchorStore(tau_insert=args.tau_insert) if is_anchor else None
    rigid_store = None
    rigid_mask = None
    if is_atlas:
        eng.attach_atlas(AtlasStore(tau_insert=args.tau_insert, n_max=100_000))
        rigid_store = RigidStore(tau_insert=args.tau_insert)
        _, _, _, lh, lw = eng.engine.frm_shape
        rigid_mask = band_mask(lh, lw, device=eng.device, feather=args.feather)

    def gen_plain(mouse=(0.0, 0.0)):
        return eng.gen(dict(button=set(), mouse=mouse))

    def gen_atlas_nowb(mouse, capture=False, project=False):
        ctrl = eng._CtrlInput(button=set(), mouse=list(mouse))
        four, info = rigid_step(
            eng.engine, ctrl, rigid_store, rigid_mask,
            project=project,
            capture=capture,
            retrieve="nearest",
            lam=args.lam,
            resp_min=args.resp_min,
            max_shift_frac=args.max_shift_frac,
            write_back=False,
        )
        return four, info

    start = None
    last_yaw_delta = 0.0
    for i, _spec in enumerate(_noop(args.settle)):
        yaw_before = float(eng.engine.camera_yaw)
        if is_atlas:
            four, _info = gen_atlas_nowb((0.0, 0.0), capture=(i == args.settle - 1))
        else:
            four = gen_plain((0.0, 0.0))
        yaw_after = float(eng.engine.camera_yaw)
        last_yaw_delta = yaw_after - yaw_before
        start = eng.last_rgb(four)
    if is_anchor:
        anchor_store.insert(eng.all_rgb(four), float(eng.engine.camera_yaw),
                            yaw_delta=last_yaw_delta)
    if is_atlas:
        eng.atlas_capture()

    for i in range(H):
        yaw_before = float(eng.engine.camera_yaw)
        if is_atlas:
            four, _info = gen_atlas_nowb((+args.yaw_mag, 0.0), capture=(i % args.M == 0))
        else:
            four = gen_plain((+args.yaw_mag, 0.0))
        yaw_after = float(eng.engine.camera_yaw)
        yaw_delta = yaw_after - yaw_before
        if is_anchor and i % args.M == 0:
            anchor_store.insert(eng.all_rgb(four), float(eng.engine.camera_yaw),
                                yaw_delta=yaw_delta)
        if is_atlas and i % args.M == 0:
            eng.atlas_capture()

    cadence = _cadence(arm)
    anchor_infos = []
    anchor_count = 0
    return_pre_anchor = None
    last_rgb = None
    last_x4 = None
    last_target_yaws = None
    for b in range(H):
        yaw_before = float(eng.engine.camera_yaw)
        if is_atlas:
            eng.atlas_activate(offset=args.restamp_offset, k=args.atlas_k,
                               align_pose=True, retrieve="nearest")
            four, _info = gen_atlas_nowb((-args.yaw_mag, 0.0), project=True)
        else:
            four = gen_plain((-args.yaw_mag, 0.0))
        yaw_after = float(eng.engine.camera_yaw)
        yaw_delta = yaw_after - yaw_before
        last_x4 = eng.all_rgb(four)
        last_target_yaws = micro_yaw_grid(yaw_after, yaw_delta, n=len(last_x4))
        last_rgb = eng.last_rgb(four)
        return_pre_anchor = last_rgb

        if is_anchor and cadence and (b % cadence == 0):
            last_rgb, ainfo = _append_anchor(
                eng, anchor_store, last_rgb, arm, args, anchor_count,
                current_x4=last_x4, target_yaws=last_target_yaws)
            anchor_count = ainfo["anchor_count"]
            if ainfo["anchor"]:
                ainfo = dict(ainfo)
                ainfo["heading"] = round((H - 1 - b) * args.yaw_mag, 6)
                anchor_infos.append(ainfo)

    if is_anchor and (cadence is None or ((H - 1) % cadence != 0)):
        last_rgb, ainfo = _append_anchor(
            eng, anchor_store, last_rgb, arm, args, anchor_count,
            current_x4=last_x4, target_yaws=last_target_yaws)
        anchor_count = ainfo["anchor_count"]
        if ainfo["anchor"]:
            ainfo = dict(ainfo)
            ainfo["heading"] = 0.0
            anchor_infos.append(ainfo)

    if is_atlas:
        _clear_pins(eng.engine.kv_cache)

    post = {}
    for p in range(1, args.post_frames + 1):
        four = gen_plain((0.0, 0.0))
        f = eng.last_rgb(four)
        if p in (1, 4, 8, 16, 32, 64):
            post[p] = f

    shown = last_rgb if anchor_infos else return_pre_anchor
    final_info = anchor_infos[-1] if anchor_infos else {}
    return {
        "start": start,
        "return_pre_anchor": return_pre_anchor,
        "intervention_or_return": shown,
        "post": post,
        "anchor_infos": anchor_infos,
        "metrics": {
            "post16_psnr": psnr(start, post[16]),
            "post64_psnr": psnr(start, post[64]),
            "post16_ssim": ssim(start, post[16]),
            "anchors": len(anchor_infos),
            "final_dx": final_info.get("dx_px"),
            "final_resp": final_info.get("resp"),
        },
    }


def resize_for_panel(img, width=360):
    h, w = img.shape[:2]
    return cv2.resize(img, (width, int(round(h * width / w))), interpolation=cv2.INTER_AREA)


def label_panel(img, title, subtitle=""):
    panel = resize_for_panel(img)
    h, w = panel.shape[:2]
    bar = np.zeros((44, w, 3), np.uint8)
    cv2.putText(bar, title, (8, 20), FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(bar, subtitle, (8, 38), FONT, 0.43, (210, 230, 255), 1, cv2.LINE_AA)
    return np.vstack([bar, panel])


def diff_heat(a, b):
    diff = np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)), axis=2)
    diff = np.clip(diff * 3.0, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(diff, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def write_png(path, rgb):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def write_contact_sheet(path, arm_results):
    rows = []
    for arm, data in arm_results.items():
        start = data["start"]
        m = data["metrics"]
        info = []
        if m["anchors"]:
            info.append(f"anchors {m['anchors']}")
        if m["final_dx"] is not None:
            info.append(f"dx {m['final_dx']:.1f}px")
        if m["final_resp"] is not None:
            info.append(f"resp {m['final_resp']:.3f}")
        row = [
            label_panel(start, f"{arm}: start"),
            label_panel(data["intervention_or_return"], "return / anchor", " ".join(info)),
            label_panel(data["post"][16], "post16", f"{m['post16_psnr']:.2f} dB vs start"),
            label_panel(data["post"][64], "post64", f"{m['post64_psnr']:.2f} dB vs start"),
            label_panel(diff_heat(start, data["post"][16]), "post16 error heat", "brighter = more error"),
        ]
        rows.append(np.hstack(row))
    sheet = np.vstack(rows)
    write_png(path, sheet)


def load_aggregate_rows(path):
    if not path or not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def aggregate_post16(rows, scene, seed, arm):
    vals = [float(r["psnr"]) for r in rows
            if r["scene"] == scene and int(r["seed"]) == seed
            and r["arm"] == arm and r["phase"] == "post" and int(r["idx"]) == 16]
    return vals[0] if vals else None


def html_page(out_dir, cases, manifest, aggregate_rows):
    cards = []
    for case in cases:
        scene = case["scene"]
        seed = case["seed"]
        slug = case["slug"]
        rows = []
        for arm, data in case["arms"].items():
            m = data["metrics"]
            agg = aggregate_post16(aggregate_rows, scene, seed, arm)
            agg_cell = "" if agg is None else f"{agg:.2f}"
            rows.append(
                f"<tr><td>{html.escape(arm)}</td><td>{m['post16_psnr']:.2f}</td>"
                f"<td>{m['post64_psnr']:.2f}</td><td>{agg_cell}</td>"
                f"<td>{m['anchors']}</td>"
                f"<td>{'' if m['final_dx'] is None else f'{m['final_dx']:.1f}'}</td>"
                f"<td>{'' if m['final_resp'] is None else f'{m['final_resp']:.3f}'}</td></tr>"
            )
        cards.append(f"""
<section>
  <h2>{html.escape(case['label'])}</h2>
  <p class="boundary">Controlled visual rerun: scene <code>{html.escape(scene)}</code>,
  seed <code>{seed}</code>. The aggregate n=6 result remains the linked CSV;
  this sheet exists so the failure/success modes are visually inspectable.</p>
  <a href="{slug}/contact_sheet.png"><img class="sheet" src="{slug}/contact_sheet.png"></a>
  <table>
    <tr><th>arm</th><th>visual post16</th><th>visual post64</th><th>aggregate post16</th><th>anchors</th><th>final dx</th><th>final resp</th></tr>
    {''.join(rows)}
  </table>
</section>
""")

    doc = f"""<!doctype html>
<meta charset="utf-8">
<title>Anchor Visual Review</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 28px; color: #111827; }}
h1 {{ margin-bottom: 6px; }}
.boundary {{ background: #fff7e6; border: 1px solid #e8c36a; padding: 10px 12px; max-width: 1180px; }}
.sheet {{ width: min(100%, 1900px); border: 1px solid #cbd5e1; display: block; margin: 14px 0; }}
table {{ border-collapse: collapse; margin: 10px 0 28px; min-width: 900px; }}
th, td {{ border: 1px solid #cbd5e1; padding: 7px 9px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th {{ background: #e5edf6; }}
code {{ background: #eef2f7; padding: 1px 4px; }}
a {{ color: #1d4ed8; }}
</style>
<h1>Anchor Visual Review</h1>
<p class="boundary">Frames-first review packet from controlled reruns of selected
Phase 10 trials. These are real generated frames from <code>examples/anchor_visual.py</code>.
They are visual explanations, not a replacement for the n=6 aggregate CSV.</p>
<p>
Raw aggregate: <a href="{html.escape(os.path.relpath(manifest['aggregate_csv'], out_dir))}">{html.escape(os.path.basename(manifest['aggregate_csv']))}</a> |
Visual manifest: <a href="manifest.json">manifest.json</a> |
Condition source: <code>{html.escape(manifest['condition_source'])}</code>
</p>
{''.join(cards)}
<h2>Provenance</h2>
<pre>{html.escape(json.dumps(manifest, indent=2))}</pre>
"""
    pathlib.Path(out_dir, "index.html").write_text(doc)


def parse_cases(spec):
    cases = []
    for item in spec.split(","):
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError("cases must be scene:seed:label comma list")
        scene, seed, label = parts
        cases.append({"scene": scene, "seed": int(seed), "label": label,
                      "slug": f"{scene}_{seed}_{label}"})
    return cases


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Overworld/Waypoint-1.5-1B")
    ap.add_argument("--out", default="bench_out/anchor/visual_review")
    ap.add_argument("--aggregate-csv", default="bench_out/anchor/anchor.csv")
    ap.add_argument("--condition-source", default="docs/ANCHOR_PLAN.md")
    ap.add_argument("--cases", default="seed_01:1235:anchor_k4_good,seed_00:1234:anchor_k4_bad")
    ap.add_argument("--arms", default="revisit,atlas_nowb,anchor,anchor_k4")
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--post-frames", type=int, default=64)
    ap.add_argument("--settle", type=int, default=8)
    ap.add_argument("--yaw-mag", type=float, default=0.2)
    ap.add_argument("--M", type=int, default=1)
    ap.add_argument("--tau-insert", type=float, default=0.05)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--feather", type=int, default=1)
    ap.add_argument("--pixel-feather", type=int, default=8)
    ap.add_argument("--resp-min", type=float, default=0.05)
    ap.add_argument("--resp-full", type=float, default=0.55)
    ap.add_argument("--max-shift-frac", type=float, default=0.35)
    ap.add_argument("--restamp-offset", type=int, default=4)
    ap.add_argument("--atlas-k", type=int, default=1)
    ap.add_argument("--assets", default="bench_assets")
    ap.add_argument("--quant", default=None)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cases = parse_cases(args.cases)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    print(f"Condition source: {args.condition_source}")
    print("Run label: visual_review")
    print(f"Command: {shlex.join(sys.argv)}")
    print("Source hashes: "
          f"anchor_visual={_sha256(__file__)} "
          f"anchor={_sha256('examples/anchor.py')} "
          f"anchor_probe={_sha256('examples/anchor_probe.py')} "
          f"plan={_sha256(args.condition_source)}")

    eng = RealEngine(args.model, "cuda", args.quant, max(args.atlas_k, 1),
                     pin_all_layers=True)
    seed_lookup = dict(load_seeds(max(int(c["scene"].split("_")[-1]) for c in cases) + 1,
                                  args.assets))
    sigmas = [float(x) for x in eng.engine.scheduler_sigmas.detach().cpu()]
    print(f"Resolved condition: model={args.model} quant={args.quant} device=cuda "
          f"K={args.K} post_frames={args.post_frames} settle={args.settle} "
          f"yaw_mag={args.yaw_mag} M={args.M} restamp_offset={args.restamp_offset} "
          f"scheduler_sigmas={sigmas} arms={arms} cases={args.cases}", flush=True)

    manifest = {
        "condition_source": args.condition_source,
        "run_label": "visual_review",
        "command": shlex.join(sys.argv),
        "model": args.model,
        "quant": args.quant,
        "device": "cuda",
        "scheduler_sigmas": sigmas,
        "arms": arms,
        "cases": cases,
        "aggregate_csv": args.aggregate_csv,
        "source_hashes": {
            "anchor_visual": _sha256(__file__),
            "anchor": _sha256("examples/anchor.py"),
            "anchor_probe": _sha256("examples/anchor_probe.py"),
            "plan": _sha256(args.condition_source),
        },
    }

    done_cases = []
    for case in cases:
        scene = case["scene"]
        seed = case["seed"]
        if scene not in seed_lookup:
            raise ValueError(f"unknown scene {scene}; loaded {sorted(seed_lookup)}")
        out_case = pathlib.Path(args.out, case["slug"])
        out_case.mkdir(parents=True, exist_ok=True)
        arm_results = {}
        for arm in arms:
            print(f"  visual {scene} seed {seed} {arm}", flush=True)
            data = run_arm(eng, seed_lookup[scene], seed, arm, args)
            arm_results[arm] = data
            arm_dir = out_case / arm
            write_png(str(arm_dir / "start.png"), data["start"])
            write_png(str(arm_dir / "return_or_anchor.png"), data["intervention_or_return"])
            write_png(str(arm_dir / "post16.png"), data["post"][16])
            write_png(str(arm_dir / "post64.png"), data["post"][64])
            write_png(str(arm_dir / "post16_error_heat.png"), diff_heat(data["start"], data["post"][16]))
        write_contact_sheet(str(out_case / "contact_sheet.png"), arm_results)
        done = dict(case)
        done["arms"] = arm_results
        done_cases.append(done)

    pathlib.Path(args.out, "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    aggregate_rows = load_aggregate_rows(args.aggregate_csv)
    html_page(args.out, done_cases, manifest, aggregate_rows)
    print(f"wrote {args.out}/index.html")
    print(f"wrote {args.out}/manifest.json")


if __name__ == "__main__":
    main()
