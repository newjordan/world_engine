# uv run --dev python examples/analyze_pose_restamp.py
"""
Quantitative analysis of the TWO fundamental flaws in the current uniform-shift
pose restamp (restamp_pose_k), and a per-token perspective-correct alternative.

FLAW 1: Uniform shift assumes camera yaw = translation, but yaw = rotation.
  -> per-token displacement varies by 2-4x across the latent grid.
  -> restamp_pose_k applies the SAME dx_norm to all 512 tokens.

FLAW 2: No calibration between mouse-velocity units and latent-pixel shift.
  -> camera_yaw = sum(mouse[0]) has unknown relationship to actual scene motion.
  -> dx_norm = 2*dyaw/W assumes 1:1 mouse-unit-to-latent-pixel mapping.

FLAW 3: Spatial RoPE aliases severely (3.2 full rotations across [-1,1]).
  -> 1 pixel of miscalibration = 72° phase error on highest freq pair.
  -> Both flaws above are amplified into catastrophic attention errors.
"""
import math
import torch
import numpy as np


# === Model RoPE geometry (Waypoint-1.5-1B) ===
D_HEAD = 64
N_XY = D_HEAD // 8    # 8 freq-pairs per spatial axis
N_X = N_XY            # 8 x-pairs
N_Y = N_XY            # 8 y-pairs
N_T = D_HEAD // 4     # 16 t-pairs
HALF = D_HEAD // 2    # 32
H, W = 16, 32         # latent grid

nyq = 0.8
max_freq = min(H, W) * nyq
n = (N_XY + 1) // 2
XY = (torch.linspace(1.0, max_freq / 2, n) * math.pi).repeat_interleave(2)[:N_XY]
THETA = 10000.0
INV_T = 1.0 / (THETA ** (torch.arange(0, N_T, 2, dtype=torch.float32) / N_T))
INV_T = INV_T.repeat_interleave(2)


def restamp_pose_uniform(k, dx_norm, dy_norm, dt):
    """Current restamp_pose_k: uniform phase shift on ALL tokens."""
    phi = k.new_zeros(HALF, dtype=torch.float32)
    phi[:N_X] = XY[:N_X] * dx_norm
    phi[N_X:N_X + N_Y] = XY[:N_Y] * dy_norm
    phi[N_X + N_Y:] = INV_T * dt
    cos, sin = phi.cos(), phi.sin()
    kf = k.float()
    k0, k1 = kf[..., :HALF], kf[..., HALF:]
    return torch.cat((k0 * cos - k1 * sin, k1 * cos + k0 * sin), dim=-1).to(k.dtype)


def restamp_pose_per_token(k, grid_x_norm, grid_y_norm, dx_norm_per_token, dt):
    """Perspective-correct restamp: each token gets its own phase shift based on
    its grid position and the rotation geometry. This is what the correct math demands.

    grid_x_norm: [..., T] per-token normalized x position
    grid_y_norm: [..., T] per-token normalized y position
    dx_norm_per_token: [..., T] per-token displacement (from rotation geometry)
    """
    T = k.shape[-2]
    # Per-token phi: [T, HALF]
    phi = k.new_zeros(T, HALF, dtype=torch.float32)
    # Broadcast per-token x displacement to x freq-pairs
    phi[:, :N_X] = (dx_norm_per_token[..., None] * XY[:N_X][None, :]).float()
    phi[:, N_X:N_X + N_Y] = 0.0  # no vertical shift under pure yaw
    phi[:, N_X + N_Y:] = INV_T[None, :].float() * dt
    cos, sin = phi.cos(), phi.sin()  # [T, HALF]
    kf = k.float()  # [B, H, T, D_HEAD]
    k0, k1 = kf[..., :HALF], kf[..., HALF:]
    out = torch.cat((k0 * cos - k1 * sin, k1 * cos + k0 * sin), dim=-1)
    return out.to(k.dtype)


def perspective_displacement(yaw_rad, fov_deg=90):
    """Compute per-token horizontal displacement (in normalized units) for a camera
    yaw rotation, assuming perspective projection."""
    half_fov = math.radians(fov_deg / 2)
    displacements = []
    for gx in range(W):
        x_norm = (2 * gx + 1) / W - 1
        view_ang = x_norm * half_fov
        new_ang = view_ang + yaw_rad
        new_x_norm = math.tan(new_ang) / math.tan(half_fov)
        displacements.append(new_x_norm - x_norm)
    return np.array(displacements)


def dot_product_ratio(k_orig, k_restamped):
    """Average cosine similarity between original and restamped keys."""
    return torch.nn.functional.cosine_similarity(
        k_orig.flatten(0, -2), k_restamped.flatten(0, -2), dim=-1
    ).mean().item()


def main():
    print("=" * 72)
    print("POSE RESTAMP ANALYSIS: uniform-shift vs perspective-correct")
    print("=" * 72)

    torch.manual_seed(42)
    k = torch.randn(1, 1, W * H, D_HEAD)  # 512 tokens, d_head=64

    # Per-token grid positions
    grid_x = torch.arange(W).repeat(H)  # [W*H]
    grid_x_norm = (2.0 * grid_x.float() + 1) / W - 1  # [512]

    fov_deg = 90

    print(f"\nModel: Waypoint-1.5-1B  d_head={D_HEAD}  grid={H}x{W}  FoV={fov_deg}°")
    print(f"Spatial freqs: {XY.tolist()}")
    print(f"Max spatial freq = {XY.max():.1f} rad = {XY.max()/(2*math.pi):.1f} rotations over [-1,1]")
    print()

    # === Sweep yaw angles ===
    print(f"{'yaw_deg':>8} {'mean_disp':>10} {'std_disp':>10} {'uniform_cos':>12} {'per_tok_cos':>12} {'ratio':>8}")
    print("-" * 72)

    for yaw_deg in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
        yaw_rad = math.radians(yaw_deg)
        disps = perspective_displacement(yaw_rad, fov_deg)  # [W]
        # Expand to all rows
        disp_per_token = torch.from_numpy(np.tile(disps, H)).float()  # [512]
        mean_disp = disp_per_token.mean().item()
        std_disp = disp_per_token.std().item()

        # Uniform restamp uses the MEAN displacement
        dx_norm_uniform = mean_disp
        k_uniform = restamp_pose_uniform(k, dx_norm_uniform, 0.0, 0)
        cos_uniform = dot_product_ratio(k, k_uniform)

        # Per-token restamp uses per-token displacement
        k_per_tok = restamp_pose_per_token(k, grid_x_norm, None, disp_per_token, 0)
        cos_per_tok = dot_product_ratio(k, k_per_tok)

        ratio = cos_per_tok / cos_uniform if abs(cos_uniform) > 1e-8 else float("inf")
        print(f"{yaw_deg:8.1f} {mean_disp:10.4f} {std_disp:10.4f} {cos_uniform:12.4f} {cos_per_tok:12.4f} {ratio:8.3f}")

    print()
    print("INTERPRETATION:")
    print("- 'uniform_cos' = how well the uniform restamp preserves the original key")
    print("  (1.0 = identical, 0 = orthogonal). Lower = MORE DESTRUCTION.")
    print("- 'per_tok_cos' = same but for perspective-correct per-token restamp.")
    print("- If per_tok >> uniform, the uniform shift is DESTROYING key information")
    print("  that a per-token rotation would preserve.")

    # === Calibration gap ===
    print("\n" + "=" * 72)
    print("CALIBRATION GAP: mouse-velocity units vs latent-pixel shift")
    print("=" * 72)
    print()
    print("camera_yaw = sum(mouse[0]) across all gen_frame calls.")
    print("For K=32, yaw_mag=0.2: peak camera_yaw = 16 * 0.2 = 3.2 (mouse units)")
    print("restamp converts: dx_norm = 2 * camera_yaw / W = 2 * 3.2 / 32 = 0.2")
    print()
    print("But the ACTUAL scene displacement per mouse unit is determined by the")
    print("model's learned dynamics — there is NO calibration factor.")
    print()
    print("If the true mouse-to-pixel calibration is ALFA (pixels per mouse-unit),")
    print("then the restamp applies dx_norm = 0.2 but the correct value would be")
    print("dx_norm = 2 * ALFA * 3.2 / 32. The ratio is 1/ALFA.")
    print()

    for alfa in [0.1, 0.25, 0.5, 1.0, 2.0]:
        correct_dn = 2.0 * alfa * 3.2 / W
        applied_dn = 0.2
        residual = abs(correct_dn - applied_dn)
        max_phase_err = XY.max().item() * residual
        print(f"  ALFA={alfa:5.2f}: correct dx_norm={correct_dn:.4f}, applied=0.2, "
              f"residual={residual:.4f}, max phase err={math.degrees(max_phase_err):.1f}°")

    print()
    print("At ALFA≠1, the restamp OVERSHOOTS or UNDERSHOOTS the correct spatial phase,")
    print("introducing a residual phase error that scales with the highest spatial freq.")
    print("Given spatial aliasing (3.2 rotations), even 10% calibration error = 23° phase")
    print("error on the highest freq pair, destroying that pair's contribution.")


if __name__ == "__main__":
    main()
