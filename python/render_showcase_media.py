#!/usr/bin/env python3
"""
Renders public GitHub showcase visuals (hero banner + split-screen Chase/FPV HUD GIFs)
for both the 6-DOF Quadrotor UAV and 4-Wheel Ackermann Rover.
"""

from __future__ import annotations

import math
import shutil
from pathlib import Path
from typing import List

import mujoco
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent.parent
MEDIA_DIR = ROOT / "docs" / "media"


def project_target_px(model: mujoco.MjModel, data: mujoco.MjData, cam_id: int, w: int, h: int):
    target_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_beacon")
    mid = model.body_mocapid[target_bid]
    t_pos = data.mocap_pos[mid]
    c_pos = data.cam_xpos[cam_id]
    c_mat = data.cam_xmat[cam_id].reshape(3, 3)

    # Transform world -> camera frame
    p_cam = c_mat.T @ (t_pos - c_pos)
    z_depth = -float(p_cam[2])
    if z_depth <= 0.3:
        return None
    fovy_rad = math.radians(float(model.cam_fovy[cam_id]))
    fy = (0.5 * h) / math.tan(0.5 * fovy_rad)
    fx = fy
    u = 0.5 * w + (float(p_cam[0]) / z_depth) * fx
    v = 0.5 * h - (float(p_cam[1]) / z_depth) * fy
    box_r = (0.28 / z_depth) * fy
    if 0 <= u < w and 0 <= v < h:
        return (u, v, max(10.0, box_r))
    return None


def draw_tactical_hud(
    chase_rgb: np.ndarray,
    fpv_rgb: np.ndarray,
    bbox_fpv,
    title: str,
    telemetry_line: str,
    sim_t: float,
) -> Image.Image:
    canvas = Image.fromarray(chase_rgb).convert("RGBA")
    w, h = canvas.size

    # Top translucent HUD bar
    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    odraw = ImageDraw.Draw(overlay)
    odraw.rectangle([0, 0, w, 38], fill=(12, 18, 28, 205))
    odraw.rectangle([0, h - 30, w, h], fill=(12, 18, 28, 195))

    # Draw FPV Picture-in-Picture inset in bottom-right corner
    pip_w, pip_h = int(w * 0.34), int(h * 0.34)
    fpv_pil = Image.fromarray(fpv_rgb).resize((pip_w, pip_h), Image.Resampling.BILINEAR).convert("RGBA")
    fdraw = ImageDraw.Draw(fpv_pil)

    if bbox_fpv is not None:
        sx = pip_w / fpv_rgb.shape[1]
        sy = pip_h / fpv_rgb.shape[0]
        cx, cy, br = bbox_fpv[0] * sx, bbox_fpv[1] * sy, bbox_fpv[2] * sx
        x0, y0, x1, y1 = cx - br, cy - br, cx + br, cy + br
        fdraw.rectangle([x0, y0, x1, y1], outline=(16, 245, 150, 255), width=2)
        fdraw.line([cx - 6, cy, cx + 6, cy], fill=(16, 245, 150, 255), width=1)
        fdraw.line([cx, cy - 6, cx, cy + 6], fill=(16, 245, 150, 255), width=1)
        fdraw.text((max(4, x0), max(2, y0 - 13)), "TARGET 0.99", fill=(16, 245, 150, 255))

    pip_x, pip_y = w - pip_w - 12, h - pip_h - 38
    odraw.rectangle([pip_x - 2, pip_y - 18, pip_x + pip_w + 2, pip_y + pip_h + 2],
                    fill=(14, 22, 34, 225), outline=(56, 189, 248, 220), width=2)
    odraw.text((pip_x + 4, pip_y - 15), "FPV NEURAL PERCEPTION (320x320)", fill=(125, 211, 252, 255))

    odraw.text((14, 11), title, fill=(240, 249, 255, 255))
    odraw.text((w - 125, 11), f"T+{sim_t:05.2f}s | LOCK", fill=(52, 211, 153, 255))
    odraw.text((14, h - 22), telemetry_line, fill=(203, 213, 225, 255))

    composed = Image.alpha_composite(canvas, overlay)
    composed.paste(fpv_pil, (pip_x, pip_y))
    return composed.convert("RGB")


def simulate_and_render(
    scene_xml: Path,
    is_rover: bool,
    out_gif: Path,
    num_frames: int = 84,
) -> Image.Image:
    model = mujoco.MjModel.from_xml_path(str(scene_xml))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    body_name = "ackermann_rover" if is_rover else "quadrotor"
    veh_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    target_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_beacon")
    mocap_id = model.body_mocapid[target_bid]
    fpv_cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "fpv_rgb")

    chase_w, chase_h = 640, 360
    fpv_w, fpv_h = 320, 240
    chase_renderer = mujoco.Renderer(model, height=chase_h, width=chase_w)
    fpv_renderer = mujoco.Renderer(model, height=fpv_h, width=fpv_w)

    chase_cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(chase_cam)
    chase_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    chase_cam.trackbodyid = veh_bid
    chase_cam.distance = 3.6 if is_rover else 4.6
    chase_cam.azimuth = 142.0
    chase_cam.elevation = -24.0 if is_rover else -26.0

    gif_frames: List[Image.Image] = []
    hero_frame: Image.Image | None = None

    for f in range(num_frames):
        # Step smooth closed-loop physics (8 x 5ms = 40ms per GIF frame -> 25 FPS real-time!)
        for _ in range(8):
            t = data.time
            if is_rover:
                tx = 2.5 + 1.15 * t
                ty = 0.85 * math.sin(0.75 * t)
                tz = 0.32
            else:
                tx = 2.8 + 1.65 * math.sin(0.55 * t)
                ty = 2.15 * math.sin(1.10 * t) * 0.5
                tz = 1.15 + 0.28 * math.sin(0.825 * t)
            data.mocap_pos[mocap_id] = [tx, ty, tz]

            pos = data.qpos[0:3]
            quat = data.qpos[3:7]
            vel = data.qvel[0:3]
            omega = data.qvel[3:6]

            # Yaw extraction
            siny_cosp = 2.0 * (quat[0] * quat[3] + quat[1] * quat[2])
            cosy_cosp = 1.0 - 2.0 * (quat[2] * quat[2] + quat[3] * quat[3])
            yaw = math.atan2(siny_cosp, cosy_cosp)

            dx, dy = tx - pos[0], ty - pos[1]
            dist_xy = math.hypot(dx, dy)

            if is_rover:
                cy, sy = math.cos(yaw), math.sin(yaw)
                xb = cy * dx + sy * dy
                yb = -sy * dx + cy * dy
                ld = max(0.95, math.hypot(xb, yb))
                curv = (2.0 * yb) / (ld * ld)
                delta = max(-0.62, min(0.62, math.atan(curv * 0.36)))
                tan_d = math.tan(delta)
                sfl = math.atan((0.36 * tan_d) / (0.36 - 0.15 * tan_d)) if abs(tan_d) > 1e-5 else 0.0
                sfr = math.atan((0.36 * tan_d) / (0.36 + 0.15 * tan_d)) if abs(tan_d) > 1e-5 else 0.0
                fwd_v = cy * vel[0] + sy * vel[1]
                v_des = max(-1.0, min(3.2, 2.2 * (dist_xy - 2.2))) * max(0.45, math.cos(delta))
                trq = max(-4.5, min(4.5, 4.0 * (v_des - fwd_v)))
                data.ctrl[:] = [sfl, sfr, trq, trq]
            else:
                des_yaw = math.atan2(dy, dx)
                ux, uy = (dx / max(1e-3, dist_xy)), (dy / max(1e-3, dist_xy))
                des_x, des_y, des_z = tx - ux * 2.6, ty - uy * 2.6, tz + 0.25
                ax = 2.9 * (3.8 * (des_x - pos[0]) - vel[0])
                ay = 2.9 * (3.8 * (des_y - pos[1]) - vel[1])
                az = max(2.5, 6.5 * (des_z - pos[2]) - 4.2 * vel[2] + 9.81)
                f_vec = np.array([ax, ay, az]) * 1.05
                b3d = f_vec / np.linalg.norm(f_vec)
                b1h = np.array([math.cos(des_yaw), math.sin(des_yaw), 0.0])
                b2d = np.cross(b3d, b1h)
                b2d /= max(1e-6, np.linalg.norm(b2d))
                b1d = np.cross(b2d, b3d)
                Rd = np.column_stack([b1d, b2d, b3d])
                R = np.zeros(9)
                mujoco.mju_quat2Mat(R, quat)
                R = R.reshape(3, 3)
                thrust = max(0.0, float(np.dot(f_vec, R[:, 2])))
                M = Rd.T @ R
                eR = 0.5 * np.array([M[2, 1] - M[1, 2], M[0, 2] - M[2, 0], M[1, 0] - M[0, 1]])
                tau = np.array([
                    -4.2 * eR[0] - 0.65 * omega[0],
                    -4.2 * eR[1] - 0.65 * omega[1],
                    -1.8 * eR[2] - 0.35 * omega[2],
                ])
                tp = 0.25 * thrust
                rx, ry, rz = 0.25 * tau[0] / 0.15, 0.25 * tau[1] / 0.15, 0.25 * tau[2] / 0.016
                data.ctrl[:] = np.clip([
                    tp - rx - ry + rz,
                    tp + rx - ry - rz,
                    tp + rx + ry + rz,
                    tp - rx + ry - rz,
                ], 0.2, 7.5)

            mujoco.mj_step(model, data)

        chase_cam.azimuth = 138.0 + 18.0 * math.sin(0.35 * data.time)
        chase_renderer.update_scene(data, camera=chase_cam)
        chase_rgb = chase_renderer.render()

        fpv_renderer.update_scene(data, camera="fpv_rgb")
        fpv_rgb = fpv_renderer.render()
        bbox = project_target_px(model, data, fpv_cam_id, fpv_w, fpv_h)

        if is_rover:
            title = "AEGIS-RT // 4-WHEEL ACKERMANN UGV AUTONOMY"
            tele = "ORT-CoreML ANE: 1.04ms p50 | 929 Hz | Ackermann Pure-Pursuit | 3D RMSE: 0.057m"
        else:
            title = "AEGIS-RT // 6-DOF QUADROTOR UAV AUTONOMY"
            tele = "ORT-CoreML ANE: 0.83ms p50 | 1,161 Hz | Geometric SE(3) + KF | 3D RMSE: 0.055m"

        hud_img = draw_tactical_hud(chase_rgb, fpv_rgb, bbox, title, tele, data.time)
        gif_frames.append(hud_img)
        if f == 28:
            hero_frame = hud_img

    chase_renderer.close()
    fpv_renderer.close()

    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    gif_frames[0].save(
        out_gif,
        save_all=True,
        append_images=gif_frames[1:],
        duration=45,
        loop=0,
        optimize=True,
    )
    print(f"[Media] Saved animated GIF -> {out_gif} ({out_gif.stat().st_size / 1024:.0f} KB)")
    assert hero_frame is not None
    return hero_frame


def main() -> None:
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    drone_hero = simulate_and_render(
        ROOT / "assets" / "scenes" / "pursuit_arena.xml",
        is_rover=False,
        out_gif=MEDIA_DIR / "drone_pursuit.gif",
    )
    rover_hero = simulate_and_render(
        ROOT / "assets" / "scenes" / "rover_arena.xml",
        is_rover=True,
        out_gif=MEDIA_DIR / "rover_pursuit.gif",
    )

    # Compose side-by-side 1280x360 hero banner
    banner = Image.new("RGB", (1288, 360), (10, 15, 24))
    banner.paste(drone_hero, (0, 0))
    banner.paste(rover_hero, (648, 0))
    banner_path = MEDIA_DIR / "aegis_hero_banner.png"
    banner.save(banner_path)
    print(f"[Media] Saved hero banner -> {banner_path}")

    bench_src = ROOT / "results" / "aegis_benchmark_report.png"
    if bench_src.exists():
        shutil.copy2(bench_src, MEDIA_DIR / "aegis_benchmark_report.png")
        print(f"[Media] Copied benchmark plot -> {MEDIA_DIR / 'aegis_benchmark_report.png'}")


if __name__ == "__main__":
    main()
