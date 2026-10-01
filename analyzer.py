"""
Hockey movement analyzer using MediaPipe Pose Landmarker (Tasks API).
Extracts biomechanical metrics from video and compares to NHL pro profiles.
"""

import cv2
import json
import numpy as np
import math
import os
from pathlib import Path

from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker, PoseLandmarkerOptions, PoseLandmark, PoseLandmarksConnections,
    _RunningMode as RunningMode,
)
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe import Image, ImageFormat

from pro_profiles import PRO_PROFILES, SKATING_METRIC_WEIGHTS, SHOOTING_METRIC_WEIGHTS

MODEL_PATH        = Path(__file__).parent / "pose_landmarker_lite.task"
CALIBRATION_FILE  = Path(__file__).parent / "calibration_data.json"


# ─── Calibration helpers ──────────────────────────────────────────────────────

def load_calibration_overrides() -> dict:
    """Load saved per-pro measurement overrides from disk."""
    if CALIBRATION_FILE.exists():
        try:
            return json.loads(CALIBRATION_FILE.read_text())
        except Exception:
            return {}
    return {}


def get_pro_ref(pro_key: str, mode: str) -> dict:
    """
    Return the reference values for a pro player in the given mode.
    Calibrated (measured) values override the hardcoded estimates wherever
    they exist, so comparisons are always apples-to-apples.
    """
    profile = PRO_PROFILES[pro_key]
    ref = dict(profile["skating"] if mode == "skating" else profile["shooting"])
    overrides = load_calibration_overrides()
    ref.update(overrides.get(pro_key, {}).get(mode, {}))
    return ref

# Landmark indices
L = PoseLandmark


# ─── Drawing helpers ──────────────────────────────────────────────────────────

CONNECTIONS = [
    (L.LEFT_SHOULDER, L.RIGHT_SHOULDER),
    (L.LEFT_SHOULDER, L.LEFT_ELBOW),   (L.LEFT_ELBOW, L.LEFT_WRIST),
    (L.RIGHT_SHOULDER, L.RIGHT_ELBOW), (L.RIGHT_ELBOW, L.RIGHT_WRIST),
    (L.LEFT_SHOULDER, L.LEFT_HIP),     (L.RIGHT_SHOULDER, L.RIGHT_HIP),
    (L.LEFT_HIP, L.RIGHT_HIP),
    (L.LEFT_HIP, L.LEFT_KNEE),         (L.LEFT_KNEE, L.LEFT_ANKLE),
    (L.RIGHT_HIP, L.RIGHT_KNEE),       (L.RIGHT_KNEE, L.RIGHT_ANKLE),
    (L.LEFT_ANKLE, L.LEFT_HEEL),       (L.LEFT_HEEL, L.LEFT_FOOT_INDEX),
    (L.RIGHT_ANKLE, L.RIGHT_HEEL),     (L.RIGHT_HEEL, L.RIGHT_FOOT_INDEX),
]

NEUTRAL_COLOR = (50, 140, 230)    # orange-amber in BGR — user skeleton colour
USER_ORANGE   = (50, 140, 230)    # same: consistent skater line colour

def _pair(a, b):
    ai = a if isinstance(a, int) else int(a)
    bi = b if isinstance(b, int) else int(b)
    return (min(ai, bi), max(ai, bi))

# Connection → (metric_key, pro_profile_key, tolerance_deg)
SKATING_CONN_MAP = {
    _pair(L.LEFT_SHOULDER,  L.LEFT_HIP):    ("forward_lean",      "forward_lean_deg",         10),
    _pair(L.RIGHT_SHOULDER, L.RIGHT_HIP):   ("forward_lean",      "forward_lean_deg",         10),
    _pair(L.LEFT_HIP,       L.LEFT_KNEE):   ("left_hip_ext",      "hip_extension_deg",        20),
    _pair(L.RIGHT_HIP,      L.RIGHT_KNEE):  ("right_hip_ext",     "hip_extension_deg",        20),
    _pair(L.LEFT_KNEE,      L.LEFT_ANKLE):  ("left_knee_angle",   "knee_bend_angle_deg",      25),
    _pair(L.RIGHT_KNEE,     L.RIGHT_ANKLE): ("right_knee_angle",  "knee_bend_angle_deg",      25),
    _pair(L.LEFT_ANKLE,     L.LEFT_HEEL):   ("left_ankle_angle",  "ankle_plantarflexion_deg", 20),
    _pair(L.RIGHT_ANKLE,    L.RIGHT_HEEL):  ("right_ankle_angle", "ankle_plantarflexion_deg", 20),
}

SHOOTING_CONN_MAP = {
    _pair(L.LEFT_SHOULDER,  L.RIGHT_SHOULDER): ("shoulder_tilt",     "shoulder_tilt_deg",  10),
    _pair(L.LEFT_HIP,       L.RIGHT_HIP):      ("hip_rotation",      "hip_rotation_deg",   15),
    _pair(L.LEFT_SHOULDER,  L.LEFT_HIP):       ("forward_lean",      "forward_lean_deg",   12),
    _pair(L.RIGHT_SHOULDER, L.RIGHT_HIP):      ("forward_lean",      "forward_lean_deg",   12),
    _pair(L.LEFT_SHOULDER,  L.LEFT_ELBOW):     ("left_elbow_angle",  None,                 None),
    _pair(L.LEFT_ELBOW,     L.LEFT_WRIST):     ("left_elbow_angle",  None,                 None),
    _pair(L.RIGHT_SHOULDER, L.RIGHT_ELBOW):    ("right_elbow_angle", None,                 None),
    _pair(L.RIGHT_ELBOW,    L.RIGHT_WRIST):    ("right_elbow_angle", None,                 None),
}

def metric_pct_score(user_val, target_val, tolerance):
    return max(0.0, 100.0 - abs(user_val - target_val) / (tolerance + 1e-8) * 100.0)

def score_to_bgr(pct):
    """<50 → red, 50-79 → orange, ≥80 → green (BGR). Used for callout box colours only."""
    if pct >= 80: return (50, 210, 50)
    if pct >= 50: return (50, 140, 230)
    return (40, 50, 220)

def connection_color(a, b, metrics, pro_ref, mode):
    """All skeleton connections are drawn in USER_ORANGE — clear, consistent skater colour."""
    return USER_ORANGE

def _draw_callout_box(img, jx, jy, bx, by, box_w, box_h,
                      label, user_str, pro_str, color, body_is_right):
    """
    Draw a pre-positioned callout box and a leader line from (jx,jy) to it.
    body_is_right=True  → box is to the LEFT  of the body, leader exits from box right edge.
    body_is_right=False → box is to the RIGHT of the body, leader exits from box left edge.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    FS_LABEL, FS_USER, FS_PRO = 0.58, 0.80, 0.62
    PAD, ROW_GAP = 22, 8

    x1, y1 = bx - 6, by - 4
    x2, y2 = bx + box_w, by + box_h

    # Dark translucent background
    rx1, ry1 = max(0, x1), max(0, y1)
    rx2, ry2 = min(img.shape[1], x2), min(img.shape[0], y2)
    roi = img[ry1:ry2, rx1:rx2]
    if roi.size > 0:
        bg = np.zeros_like(roi); bg[:] = (8, 16, 28)
        cv2.addWeighted(bg, 0.85, roi, 0.15, 0, roi)
        img[ry1:ry2, rx1:rx2] = roi
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # Leader line: exits whichever edge of the box faces the body
    anchor_x = x2 if body_is_right else x1
    anchor_y = (y1 + y2) // 2
    cv2.line(img, (jx, jy), (anchor_x, anchor_y), color, 2)
    cv2.circle(img, (jx, jy), 9,  color, -1)
    cv2.circle(img, (jx, jy), 12, color,  2)

    # Text rows
    (_, lh), lb = cv2.getTextSize(label,    font, FS_LABEL, 1)
    (_, uh), ub = cv2.getTextSize(user_str, font, FS_USER,  2)
    (_, ph), pb = cv2.getTextSize(pro_str,  font, FS_PRO,   1) if pro_str else ((0, 0), 0)

    cy = by + lh + PAD // 2
    cv2.putText(img, label,    (bx, cy), font, FS_LABEL, (180, 205, 225), 1, cv2.LINE_AA)
    cy += uh + ROW_GAP + ub
    cv2.putText(img, user_str, (bx, cy), font, FS_USER,  color, 2, cv2.LINE_AA)
    if pro_str:
        cy += ph + ROW_GAP + pb
        cv2.putText(img, pro_str, (bx, cy), font, FS_PRO, (130, 165, 195), 1, cv2.LINE_AA)


_CALLOUT_SEGMENT_MAP = {
    "KNEE BEND":     [(L.LEFT_HIP,       L.LEFT_KNEE),
                      (L.LEFT_KNEE,      L.LEFT_ANKLE)],
    "FORWARD LEAN":  [(L.LEFT_HIP,       L.LEFT_SHOULDER)],
    "HIP EXTENSION": [(L.LEFT_SHOULDER,  L.LEFT_HIP),
                      (L.LEFT_HIP,       L.LEFT_KNEE)],
    "SHIN ANGLE":    [(L.LEFT_KNEE,      L.LEFT_ANKLE)],
    "HIP ROTATION":  [(L.LEFT_HIP,       L.RIGHT_HIP)],
    "SHOULDER TILT": [(L.LEFT_SHOULDER,  L.RIGHT_SHOULDER)],
    "ELBOW ANGLE":   [(L.LEFT_SHOULDER,  L.LEFT_ELBOW),
                      (L.LEFT_ELBOW,     L.LEFT_WRIST)],
}


def annotate_frame(frame, lm_coords, w, h, metrics, pro_ref, mode, session_support=None):
    """Draw color-coded skeleton and stacked callout boxes outside the body region.
    Returns callout_meta — a list of dicts for interactive JS hover/highlight."""
    def px(idx):
        x, y = lm_coords[idx]
        return (int(x * w), int(y * h))

    # Skeleton — all connections in user orange, clearly on the body
    for a, b in CONNECTIONS:
        cv2.line(frame, px(a), px(b), USER_ORANGE, 2, cv2.LINE_AA)

    # Joint dots — slightly larger, same orange
    for x, y in lm_coords:
        cv2.circle(frame, (int(x * w), int(y * h)), 4, USER_ORANGE, -1)
        cv2.circle(frame, (int(x * w), int(y * h)), 4, (255, 255, 255), 1)

    # ── Pick supporting leg (more bent knee = weight-bearing leg on the ice) ──
    # Using knee ANGLE from the metrics dict: the support leg always has a smaller
    # angle (more deeply flexed) than the push leg, which is extending.
    # When the angle difference is small (< 20°, e.g. transition frames), we fall back
    # to the session-dominant support leg computed from frames with clear distinction.
    l_knee_angle = metrics.get("left_knee_angle", 180)
    r_knee_angle = metrics.get("right_knee_angle", 180)
    angle_diff = abs(l_knee_angle - r_knee_angle)

    # Ankle-proximity tiebreaker: when knee angles are ambiguous (< 20° apart),
    # the leg whose ankle is closest to the hip midpoint is bearing the body weight.
    lhx_, lhy_ = px(L.LEFT_HIP); rhx_, rhy_ = px(L.RIGHT_HIP)
    mid_hip_x = (lhx_ + rhx_) / 2
    l_ankle_dist = abs(px(L.LEFT_ANKLE)[0]  - mid_hip_x)
    r_ankle_dist = abs(px(L.RIGHT_ANKLE)[0] - mid_hip_x)

    if angle_diff >= 20:
        # Clear signal from knee angles — use that
        _support = 'LEFT' if l_knee_angle <= r_knee_angle else 'RIGHT'
    else:
        # Ambiguous knee angles — ankle closer to body center is the weight-bearing leg
        _support = 'LEFT' if l_ankle_dist <= r_ankle_dist else 'RIGHT'

    print(f"[SUP LEG] L_knee={l_knee_angle:.1f}° R_knee={r_knee_angle:.1f}° diff={angle_diff:.1f}° "
          f"L_ankle_dist={l_ankle_dist:.0f} R_ankle_dist={r_ankle_dist:.0f} "
          f"→ support={_support}", flush=True)
    if _support == 'LEFT':        # LEFT knee more bent = LEFT is support leg
        sup_hip_i, sup_knee_i, sup_ankle_i = L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE
        sup_heel_i                          = L.LEFT_HEEL
        psh_hip_i, psh_knee_i, psh_ankle_i = L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE
        sup_knee_key  = "left_knee_angle"
        sup_ankle_key = "left_ankle_angle"
        sup_shin_key  = "left_shin_angle"
        psh_hip_key   = "right_hip_ext"
    else:                                   # RIGHT knee more bent = RIGHT is support leg
        sup_hip_i, sup_knee_i, sup_ankle_i = L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE
        sup_heel_i                          = L.RIGHT_HEEL
        psh_hip_i, psh_knee_i, psh_ankle_i = L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE
        sup_knee_key  = "right_knee_angle"
        sup_ankle_key = "right_ankle_angle"
        sup_shin_key  = "right_shin_angle"
        psh_hip_key   = "left_hip_ext"

    # Segment midpoint helper — leader line anchors here so it points at the measured bone
    def _seg_mid(a_i, b_i):
        ax, ay = px(a_i); bx_, by_ = px(b_i)
        return ((ax + bx_) // 2, (ay + by_) // 2)

    # Mid-torso points (used for legacy average references)
    lhx, lhy = px(L.LEFT_HIP);   rhx, rhy = px(L.RIGHT_HIP)
    lsx, lsy = px(L.LEFT_SHOULDER); rsx, rsy = px(L.RIGHT_SHOULDER)
    mid_hip_px  = ((lhx + rhx) // 2, (lhy + rhy) // 2)
    mid_sh_px   = ((lsx + rsx) // 2, (lsy + rsy) // 2)
    mid_torso_px = ((mid_hip_px[0] + mid_sh_px[0]) // 2,
                    (mid_hip_px[1] + mid_sh_px[1]) // 2)

    # Forward lean — use same-side (supporting leg) shoulder & hip for accurate side-view line
    fl_sh_i   = L.LEFT_SHOULDER if sup_hip_i == L.LEFT_HIP else L.RIGHT_SHOULDER
    fl_hip_px = px(sup_hip_i)
    fl_sh_px  = px(fl_sh_i)
    fl_mid_px = ((fl_hip_px[0] + fl_sh_px[0]) // 2, (fl_hip_px[1] + fl_sh_px[1]) // 2)

    def _ip(p): return [int(p[0]), int(p[1])]
    def _s(*pairs): return [[_ip(px(a)), _ip(px(b))] for a, b in pairs]

    # ── Callout definitions ───────────────────────────────────────────────────
    # New format: (anchor_px, segs_list, label, metric_key, target, tol)
    # segs_list = list of [[x1,y1],[x2,y2]] already-computed pixel segment pairs
    # Multiple segments per metric so we show the actual joint angle shape
    if mode == "skating":
        l_ankle_px   = px(L.LEFT_ANKLE)
        r_ankle_px   = px(L.RIGHT_ANKLE)
        mid_ankle_px = ((l_ankle_px[0] + r_ankle_px[0]) // 2,
                        (l_ankle_px[1] + r_ankle_px[1]) // 2)
        raw_callouts = [
            # KNEE BEND / ANKLE DORSIFLEXION — supporting leg: thigh + shin + foot chain
            (_seg_mid(sup_hip_i, sup_knee_i),
             _s((sup_hip_i, sup_knee_i), (sup_knee_i, sup_ankle_i), (sup_ankle_i, sup_heel_i)),
             "KNEE BEND / ANKLE DORSIFLEXION", sup_knee_key, pro_ref.get("knee_bend_angle_deg", 105), 25),
            # FORWARD LEAN — same-side hip→shoulder for accurate side-view torso line
            (fl_mid_px,
             [[_ip(fl_hip_px), _ip(fl_sh_px)]],
             "FORWARD LEAN", "forward_lean", pro_ref.get("forward_lean_deg", 15), 10),
            # SHIN ANGLE — supporting leg shin only
            (_seg_mid(sup_knee_i, sup_ankle_i),
             _s((sup_knee_i, sup_ankle_i)),
             "SHIN ANGLE", sup_shin_key, pro_ref.get("shin_angle_deg", 20), 10),
            # HIP EXTENSION — push leg: thigh + shin
            (_seg_mid(psh_hip_i, psh_knee_i),
             _s((psh_hip_i, psh_knee_i), (psh_knee_i, psh_ankle_i)),
             "HIP EXTENSION", psh_hip_key, pro_ref.get("hip_extension_deg", 175), 20),
            # STRIDE LENGTH — ankle-to-ankle horizontal span
            (mid_ankle_px,
             [[_ip(l_ankle_px), _ip(r_ankle_px)]],
             "STRIDE LENGTH", "stride_length_ratio", pro_ref.get("stride_length_ratio", 1.2), 0.3),
            # SYMMETRY — both left and right legs for visual L/R comparison
            (_seg_mid(L.LEFT_HIP, L.LEFT_KNEE),
             _s((L.LEFT_HIP, L.LEFT_KNEE), (L.LEFT_KNEE, L.LEFT_ANKLE),
                (L.RIGHT_HIP, L.RIGHT_KNEE), (L.RIGHT_KNEE, L.RIGHT_ANKLE)),
             "SYMMETRY", "symmetry", None, None),
        ]
    else:
        raw_callouts = [
            (_seg_mid(L.LEFT_HIP, L.RIGHT_HIP),
             _s((L.LEFT_HIP, L.RIGHT_HIP)),
             "HIP ROTATION", "hip_rotation", pro_ref.get("hip_rotation_deg", 85), 15),
            (_seg_mid(L.LEFT_SHOULDER, L.RIGHT_SHOULDER),
             _s((L.LEFT_SHOULDER, L.RIGHT_SHOULDER)),
             "SHOULDER TILT", "shoulder_tilt", pro_ref.get("shoulder_tilt_deg", 15), 10),
            (_seg_mid(L.LEFT_ELBOW, L.LEFT_WRIST),
             _s((L.LEFT_SHOULDER, L.LEFT_ELBOW), (L.LEFT_ELBOW, L.LEFT_WRIST)),
             "ELBOW ANGLE", "left_elbow_angle", None, None),
        ]

    # ── Pre-compute box sizes ─────────────────────────────────────────────────
    font = cv2.FONT_HERSHEY_SIMPLEX
    FS_LABEL, FS_USER, FS_PRO = 0.58, 0.80, 0.62
    PAD, ROW_GAP = 22, 8

    specs = []
    for anchor_px, segs_list, label, metric_key, target, tol in raw_callouts:
        user_val = metrics.get(metric_key, 0)
        if label == "STRIDE LENGTH":
            user_str = f"{user_val:.2f}×"
            pro_str  = f"Pro: {target:.2f}×" if target is not None else ""
        else:
            user_str = f"{user_val:.0f} deg"
            pro_str  = f"Pro: {target:.0f} deg" if target is not None else ""
        color = (score_to_bgr(metric_pct_score(user_val, target, tol))
                 if target is not None else NEUTRAL_COLOR)

        (lw, lh), _  = cv2.getTextSize(label,    font, FS_LABEL, 1)
        (uw, uh), ub = cv2.getTextSize(user_str, font, FS_USER,  2)
        (pw, ph), pb = (cv2.getTextSize(pro_str, font, FS_PRO, 1)
                        if pro_str else ((0, 0), 0))
        box_w = max(lw, uw, pw) + PAD * 2
        box_h = lh + ub + uh + (ROW_GAP + pb + ph if pro_str else 0) + PAD + ROW_GAP * 2
        specs.append((anchor_px, segs_list, label, user_str, pro_str, color, box_w, box_h))

    # ── Place boxes far from body, with extra margin ──────────────────────────
    MARGIN      = 8
    BOX_MARGIN  = 40
    GAP_BETWEEN = 12
    body_xs     = [int(x * w) for x, _ in lm_coords]
    body_left   = min(body_xs)
    body_right  = max(body_xs)
    right_space = w - body_right - MARGIN - BOX_MARGIN
    left_space  = body_left - MARGIN - BOX_MARGIN
    place_right = right_space >= left_space

    max_box_w = max(s[6] for s in specs)

    if place_right:
        bx = body_right + MARGIN + BOX_MARGIN
        bx = min(bx, w - max_box_w - MARGIN)
    else:
        bx = body_left - max_box_w - MARGIN - BOX_MARGIN
        bx = max(bx, MARGIN)

    # ── Stack boxes centred vertically on frame ───────────────────────────────
    total_h   = sum(s[7] for s in specs) + GAP_BETWEEN * (len(specs) - 1)
    stack_top = max(MARGIN, min(h - total_h - MARGIN, h // 2 - total_h // 2))

    cy = stack_top
    callout_meta = []
    for spec in specs:
        anchor_px, segs_list, label, user_str, pro_str, color, box_w, box_h = spec
        jx, jy = anchor_px

        cv2.circle(frame, (jx, jy), 6, color, -1)
        cv2.circle(frame, (jx, jy), 6, (255, 255, 255), 1)

        _draw_callout_box(frame, jx, jy, bx, cy, max_box_w, box_h,
                          label, user_str, pro_str, color,
                          body_is_right=not place_right)

        # segs_list already contains the correct pixel segment pairs
        segs = segs_list

        # Pro-target segment for JS canvas click overlay
        pro_seg = []
        try:
            if mode == "skating":
                if label == "KNEE BEND / ANKLE DORSIFLEXION":
                    # Thigh (actual) + shin rotated to target knee angle + foot continuing from target ankle
                    t_knee  = pro_ref.get("knee_bend_angle_deg", 105)
                    t_ankle = pro_ref.get("ankle_plantarflexion_deg", 155)
                    # pa = where ankle WOULD BE at target knee angle (shin pivots around knee)
                    pa = _pro_target_point(px(sup_hip_i), px(sup_knee_i), px(sup_ankle_i), t_knee)
                    # Translate heel to maintain same foot shape, anchored at pa (target ankle)
                    _ankle_arr = np.array(px(sup_ankle_i), dtype=float)
                    _heel_arr  = np.array(px(sup_heel_i),  dtype=float)
                    _pa_arr    = np.array(pa, dtype=float)
                    _heel_at_pa = tuple((_pa_arr + (_heel_arr - _ankle_arr)).astype(int))
                    # ph = where heel would be at target ankle angle, pivoting around pa
                    ph = _pro_target_point(px(sup_knee_i), pa, _heel_at_pa, t_ankle)
                    pro_seg = {"segments": [
                        [_ip(px(sup_hip_i)),  _ip(px(sup_knee_i))],  # thigh (actual)
                        [_ip(px(sup_knee_i)), _ip(pa)],               # shin at target knee angle
                        [_ip(pa),             _ip(ph)],               # foot from target ankle (connected)
                    ]}
                elif label == "FORWARD LEAN":
                    t = pro_ref.get("forward_lean_deg", 15)
                    fhx, fhy = fl_hip_px
                    fsx, fsy = fl_sh_px
                    tlen  = math.hypot(fsx - fhx, fsy - fhy)
                    lsign = 1 if (fsx - fhx) >= 0 else -1
                    pro_seg = [[fhx, fhy],
                               [fhx + lsign * int(math.sin(math.radians(t)) * tlen),
                                fhy - int(math.cos(math.radians(t)) * tlen)]]
                elif label == "SHIN ANGLE":
                    t = pro_ref.get("shin_angle_deg", 20)
                    ankle_px_ = px(sup_ankle_i)
                    knee_px_  = px(sup_knee_i)
                    shin_len  = math.hypot(knee_px_[0] - ankle_px_[0], knee_px_[1] - ankle_px_[1])
                    pro_knee_x = ankle_px_[0] + int(math.sin(math.radians(t)) * shin_len)
                    pro_knee_y = ankle_px_[1] - int(math.cos(math.radians(t)) * shin_len)
                    pro_seg = [_ip(ankle_px_), [pro_knee_x, pro_knee_y]]
                elif label == "HIP EXTENSION":
                    # Pro target: thigh of push leg at target hip extension angle.
                    # Hip extension = angle(shoulder-hip-knee), so reference is shoulder.
                    t = pro_ref.get("hip_extension_deg", 175)
                    psh_sh_i = L.LEFT_SHOULDER if psh_hip_i == L.LEFT_HIP else L.RIGHT_SHOULDER
                    # pa = where knee WOULD BE at target hip extension (thigh pivots around hip)
                    pa = _pro_target_point(px(psh_sh_i), px(psh_hip_i), px(psh_knee_i), t)
                    # Show thigh (hip→target_knee) + shin continuing from target knee to actual ankle
                    pro_seg = {"segments": [
                        [_ip(px(psh_hip_i)), _ip(pa)],            # thigh at target extension
                        [_ip(pa), _ip(px(psh_ankle_i))],          # shin from target knee to ankle
                    ]}
                elif label == "SYMMETRY":
                    pro_seg = []
                elif label == "STRIDE LENGTH":
                    # Pro target stride = ratio × torso_len, centred on mid-ankle
                    t_ratio   = pro_ref.get("stride_length_ratio", 1.2)
                    lhx_, lhy_ = px(L.LEFT_HIP);   rhx_, rhy_ = px(L.RIGHT_HIP)
                    lsx_, lsy_ = px(L.LEFT_SHOULDER); rsx_, rsy_ = px(L.RIGHT_SHOULDER)
                    m_hip_ = ((lhx_ + rhx_) / 2, (lhy_ + rhy_) / 2)
                    m_sh_  = ((lsx_ + rsx_) / 2, (lsy_ + rsy_) / 2)
                    torso_len_ = math.hypot(m_sh_[0] - m_hip_[0], m_sh_[1] - m_hip_[1])
                    target_px  = int(t_ratio * torso_len_)
                    mx = mid_ankle_px[0];  ay = mid_ankle_px[1]
                    pro_seg = [[mx - target_px // 2, ay], [mx + target_px // 2, ay]]
            else:
                if label == "SHOULDER TILT":
                    t   = pro_ref.get("shoulder_tilt_deg", 15)
                    lsh = np.array(px(L.LEFT_SHOULDER), dtype=float)
                    rsh = np.array(px(L.RIGHT_SHOULDER), dtype=float)
                    sh_len = float(np.linalg.norm(rsh - lsh))
                    anchor = lsh if lsh[1] <= rsh[1] else rsh
                    lsign  = 1 if (rsh[0] - lsh[0]) >= 0 else -1
                    pro_seg = [_ip(anchor),
                               [int(anchor[0] + lsign * math.cos(math.radians(t)) * sh_len),
                                int(anchor[1] + math.sin(math.radians(t)) * sh_len)]]
                elif label == "ELBOW ANGLE":
                    pw_ = _pro_target_point(px(L.LEFT_SHOULDER), px(L.LEFT_ELBOW), px(L.LEFT_WRIST), 90)
                    pro_seg = [_ip(px(L.LEFT_ELBOW)), _ip(pw_)]
        except Exception:
            pro_seg = []

        callout_meta.append({
            "label":    label,
            "user_str": user_str,
            "pro_str":  pro_str,
            "joint":    [jx, jy],
            "box":      [bx - 6, cy - 4, max_box_w + 6, box_h + 4],
            "color":    "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0]),
            "segments": segs,
            "pro_seg":  pro_seg,
        })
        cy += box_h + GAP_BETWEEN

    # Pro-target overlay is shown interactively on click in the JS canvas — not baked in.
    return callout_meta


# ─── Pro target overlay ───────────────────────────────────────────────────────

PRO_GREEN = (50, 230, 80)

def _pro_target_point(A, B, C, target_deg):
    """
    Return the point D where angle(A-B-D) = target_deg, |BD| = |BC|,
    on the same rotational side as C relative to ray B→A.
    All inputs are (x, y) integer tuples; returns an (x, y) integer tuple.
    """
    A = np.array(A, dtype=float)
    B = np.array(B, dtype=float)
    C = np.array(C, dtype=float)
    bc_len = np.linalg.norm(C - B)
    if bc_len < 1:
        return tuple(C.astype(int))
    ba_hat = (A - B) / (np.linalg.norm(A - B) + 1e-8)
    # Determine which side of B→A the current distal point C lies on
    sign = 1 if (ba_hat[0] * (C - B)[1] - ba_hat[1] * (C - B)[0]) >= 0 else -1
    a = math.radians(target_deg) * sign
    ca, sa = math.cos(a), math.sin(a)
    d = np.array([ba_hat[0] * ca - ba_hat[1] * sa,
                  ba_hat[0] * sa + ba_hat[1] * ca])
    return tuple((B + d * bc_len).astype(int))


def _draw_pro_overlay(frame, lm_coords, w, h, pro_ref, mode):
    """Draw green lines showing the ideal pro limb positions."""
    def px(idx):
        x, y = lm_coords[idx]
        return (int(x * w), int(y * h))

    def seg(p1, p2):
        cv2.line(frame, p1, p2, PRO_GREEN, 2, cv2.LINE_AA)
        cv2.circle(frame, p2, 5, PRO_GREEN, -1)

    if mode == "skating":
        t_knee = pro_ref.get("knee_bend_angle_deg", 105)
        t_hip  = pro_ref.get("hip_extension_deg",  175)
        t_lean = pro_ref.get("forward_lean_deg",    15)

        # Knee bend: pro lower-leg direction from each knee
        for hip_i, knee_i, ankle_i in [
            (L.LEFT_HIP,  L.LEFT_KNEE,  L.LEFT_ANKLE),
            (L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE),
        ]:
            pro_ankle = _pro_target_point(px(hip_i), px(knee_i), px(ankle_i), t_knee)
            seg(px(knee_i), pro_ankle)

        # Hip extension: pro upper-leg direction from each hip
        for sh_i, hip_i, knee_i in [
            (L.LEFT_SHOULDER,  L.LEFT_HIP,  L.LEFT_KNEE),
            (L.RIGHT_SHOULDER, L.RIGHT_HIP, L.RIGHT_KNEE),
        ]:
            pro_knee = _pro_target_point(px(sh_i), px(hip_i), px(knee_i), t_hip)
            seg(px(hip_i), pro_knee)

        # Forward lean: pro torso line from mid-hip
        lhx, lhy = px(L.LEFT_HIP)
        rhx, rhy = px(L.RIGHT_HIP)
        lsx, lsy = px(L.LEFT_SHOULDER)
        rsx, rsy = px(L.RIGHT_SHOULDER)
        mhx, mhy = (lhx + rhx) // 2, (lhy + rhy) // 2
        msx, msy = (lsx + rsx) // 2, (lsy + rsy) // 2
        torso_len = math.hypot(msx - mhx, msy - mhy)
        lean_sign = 1 if (msx - mhx) >= 0 else -1
        pro_sx = mhx + lean_sign * int(math.sin(math.radians(t_lean)) * torso_len)
        pro_sy = mhy - int(math.cos(math.radians(t_lean)) * torso_len)
        seg((mhx, mhy), (pro_sx, pro_sy))

    else:  # shooting
        # Elbow angle: pro forearm direction from each elbow
        t_elbow = 90   # typical elbow angle at release
        for sh_i, elbow_i, wrist_i in [
            (L.LEFT_SHOULDER,  L.LEFT_ELBOW,  L.LEFT_WRIST),
            (L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST),
        ]:
            pro_wrist = _pro_target_point(px(sh_i), px(elbow_i), px(wrist_i), t_elbow)
            seg(px(elbow_i), pro_wrist)

        # Shoulder tilt: pro shoulder axis anchored at the higher shoulder
        t_tilt = pro_ref.get("shoulder_tilt_deg", 15)
        lsh = np.array(px(L.LEFT_SHOULDER))
        rsh = np.array(px(L.RIGHT_SHOULDER))
        sh_len = float(np.linalg.norm(rsh - lsh))
        # Anchor on whichever shoulder is higher (lower y = higher in frame)
        if lsh[1] <= rsh[1]:
            anchor, far = lsh, rsh
            lean_sign = 1 if (rsh[0] - lsh[0]) >= 0 else -1
        else:
            anchor, far = rsh, lsh
            lean_sign = 1 if (lsh[0] - rsh[0]) >= 0 else -1
        pro_far_x = int(anchor[0] + lean_sign * math.cos(math.radians(t_tilt)) * sh_len)
        pro_far_y = int(anchor[1] + math.sin(math.radians(t_tilt)) * sh_len)
        seg(tuple(anchor), (pro_far_x, pro_far_y))

    pass  # Legend shown in the UI, not baked into every frame


# ─── Geometry helpers ─────────────────────────────────────────────────────────

def angle_between(a, b, c):
    """Angle at vertex b, given three (x,y) tuples."""
    ba = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
    bc = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
    cos_a = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8)
    return math.degrees(math.acos(np.clip(cos_a, -1, 1)))


def pt(lms, idx, w, h):
    l = lms[idx]
    return (l.x * w, l.y * h)


def vertical_angle(top, bot):
    dy = bot[1] - top[1]
    dx = bot[0] - top[0]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-8))


# ─── Per-frame metrics ────────────────────────────────────────────────────────

def extract_frame_metrics(lms, w, h, world_lms=None):
    l_hip   = pt(lms, L.LEFT_HIP,   w, h)
    r_hip   = pt(lms, L.RIGHT_HIP,  w, h)
    l_knee  = pt(lms, L.LEFT_KNEE,  w, h)
    r_knee  = pt(lms, L.RIGHT_KNEE, w, h)
    l_ankle = pt(lms, L.LEFT_ANKLE, w, h)
    r_ankle = pt(lms, L.RIGHT_ANKLE, w, h)
    l_heel  = pt(lms, L.LEFT_HEEL,  w, h)
    r_heel  = pt(lms, L.RIGHT_HEEL, w, h)
    l_foot  = pt(lms, L.LEFT_FOOT_INDEX,  w, h)
    r_foot  = pt(lms, L.RIGHT_FOOT_INDEX, w, h)
    l_sh    = pt(lms, L.LEFT_SHOULDER,  w, h)
    r_sh    = pt(lms, L.RIGHT_SHOULDER, w, h)
    l_elbow = pt(lms, L.LEFT_ELBOW,  w, h)
    r_elbow = pt(lms, L.RIGHT_ELBOW, w, h)
    l_wrist = pt(lms, L.LEFT_WRIST,  w, h)
    r_wrist = pt(lms, L.RIGHT_WRIST, w, h)

    mid_hip = ((l_hip[0]+r_hip[0])/2, (l_hip[1]+r_hip[1])/2)
    mid_sh  = ((l_sh[0]+r_sh[0])/2,   (l_sh[1]+r_sh[1])/2)

    m = {}

    # Knee angles
    m["left_knee_angle"]  = angle_between(l_hip, l_knee, l_ankle)
    m["right_knee_angle"] = angle_between(r_hip, r_knee, r_ankle)
    m["knee_angle"] = (m["left_knee_angle"] + m["right_knee_angle"]) / 2

    # Hip extension
    m["left_hip_ext"]  = angle_between(l_sh, l_hip, l_knee)
    m["right_hip_ext"] = angle_between(r_sh, r_hip, r_knee)
    m["hip_extension"] = max(m["left_hip_ext"], m["right_hip_ext"])

    # Forward lean
    m["forward_lean"] = vertical_angle(mid_sh, mid_hip)

    # Stride length ratio (foot x-spread / torso length)
    torso_len = math.hypot(mid_sh[0] - mid_hip[0], mid_sh[1] - mid_hip[1]) + 1e-6
    m["stride_length_ratio"] = abs(l_ankle[0] - r_ankle[0]) / torso_len

    # Bilateral symmetry (knee bend similarity)
    knee_diff = abs(m["left_knee_angle"] - m["right_knee_angle"])
    m["symmetry"] = max(0.0, 1.0 - knee_diff / 180.0)

    # Ankle angles
    m["left_ankle_angle"]  = angle_between(l_knee, l_ankle, l_foot)
    m["right_ankle_angle"] = angle_between(r_knee, r_ankle, r_foot)

    # Shin angle — forward tilt of the shin from vertical.
    # vertical_angle(bottom, top): angle of the bottom→top vector from vertical.
    # ankle is bottom, knee is top → measures how far forward the shin is leaning.
    m["left_shin_angle"]  = vertical_angle(l_ankle, l_knee)
    m["right_shin_angle"] = vertical_angle(r_ankle, r_knee)
    # Use the supporting leg (more bent knee = lower to ice)
    if m["left_knee_angle"] <= m["right_knee_angle"]:
        m["shin_angle"] = m["left_shin_angle"]
    else:
        m["shin_angle"] = m["right_shin_angle"]

    # Heel lift
    avg_hip_y = mid_hip[1]
    m["max_heel_lift"] = max(
        max(0, avg_hip_y - l_heel[1]),
        max(0, avg_hip_y - r_heel[1]),
    ) / (h + 1e-6)

    # ── Shooting metrics — use 3D world landmarks when available ─────────────
    # 2D projections of shoulder tilt and hip rotation collapse to near-zero
    # whenever the camera isn't directly in front of the player. World landmarks
    # give metric-space (x,y,z) coordinates that are camera-angle independent.
    if world_lms is not None:
        try:
            ls = world_lms[int(L.LEFT_SHOULDER)]
            rs = world_lms[int(L.RIGHT_SHOULDER)]
            # Shoulder tilt: how much one shoulder drops below the other in 3D.
            # MediaPipe world y-axis points downward, so higher y = lower in space.
            vert_sh  = abs(rs.y - ls.y)
            horiz_sh = math.sqrt((rs.x - ls.x)**2 + (rs.z - ls.z)**2) + 1e-8
            m["shoulder_tilt"] = math.degrees(math.atan2(vert_sh, horiz_sh))

            lh = world_lms[int(L.LEFT_HIP)]
            rh = world_lms[int(L.RIGHT_HIP)]
            # Hip rotation: angle of the hip line from the lateral (x) axis.
            # Hips perpendicular to the camera → large z-separation → ~90°.
            # Hips squared to camera → large x-separation → ~0°.
            m["hip_rotation"] = abs(math.degrees(math.atan2(
                abs(rh.z - lh.z),
                abs(rh.x - lh.x) + 1e-8,
            )))
        except Exception:
            # Fallback to 2D if world landmark access fails
            sh_vec  = (r_sh[0]-l_sh[0], r_sh[1]-l_sh[1])
            m["shoulder_tilt"] = abs(math.degrees(math.atan2(abs(sh_vec[1]), abs(sh_vec[0])+1e-8)))
            hip_vec = (r_hip[0]-l_hip[0], r_hip[1]-l_hip[1])
            m["hip_rotation"]  = abs(math.degrees(math.atan2(hip_vec[1], hip_vec[0]+1e-8)))
    else:
        sh_vec  = (r_sh[0]-l_sh[0], r_sh[1]-l_sh[1])
        m["shoulder_tilt"] = abs(math.degrees(math.atan2(abs(sh_vec[1]), abs(sh_vec[0])+1e-8)))
        hip_vec = (r_hip[0]-l_hip[0], r_hip[1]-l_hip[1])
        m["hip_rotation"]  = abs(math.degrees(math.atan2(hip_vec[1], hip_vec[0]+1e-8)))

    # Weight transfer (horizontal COM position)
    m["weight_transfer"] = mid_hip[0] / w

    # Elbow extension
    m["left_elbow_angle"]  = angle_between(l_sh, l_elbow, l_wrist)
    m["right_elbow_angle"] = angle_between(r_sh, r_elbow, r_wrist)

    # Release point (wrist height relative to hip)
    l_wh = (mid_hip[1] - l_wrist[1]) / (h + 1e-6)
    r_wh = (mid_hip[1] - r_wrist[1]) / (h + 1e-6)
    m["release_point"] = max(l_wh, r_wh)

    return m


def aggregate_metrics(frames):
    if not frames:
        return {}
    keys = frames[0].keys()
    # Angle metrics should never be 0 in a valid skating frame — skip zeros
    ANGLE_KEYS = {
        "left_knee_angle", "right_knee_angle", "knee_angle",
        "left_hip_ext", "right_hip_ext", "hip_extension",
        "forward_lean", "left_ankle_angle", "right_ankle_angle",
        "left_elbow_angle", "right_elbow_angle",
        "left_shin_angle", "right_shin_angle", "shin_angle",
    }
    # These metrics should use the peak value across frames, not the mean.
    # release_point peaks at the moment of release; shoulder_tilt peaks during the shot.
    # Averaging these across all frames collapses them to near-zero.
    PEAK_KEYS = {"release_point", "shoulder_tilt"}
    result = {}
    for k in keys:
        vals = [f[k] for f in frames if k in f]
        if k in ANGLE_KEYS:
            vals = [v for v in vals if v > 1.0]   # drop zero/near-zero angle readings
        if k in PEAK_KEYS:
            result[k] = float(np.max(vals)) if vals else 0.0
        else:
            result[k] = float(np.mean(vals)) if vals else 0.0
    return result


# ─── Scoring & feedback ───────────────────────────────────────────────────────

def score_vs_profile(user_metrics, pro_key, mode):
    profile = PRO_PROFILES[pro_key]
    ref     = get_pro_ref(pro_key, mode)   # calibrated values override hardcoded estimates
    feedbacks, scores = [], []

    if mode == "skating":
        # Knee bend — normalized: 180° (straight) = 0, pro target = 100
        # User gets credit for any bend, not a cliff-edge at ±40° from pro
        uk = user_metrics.get("knee_angle", 130)
        tk = ref["knee_bend_angle_deg"]
        diff = uk - tk
        norm_knee = max(0.0, (180 - uk) / max(1, 180 - tk))
        ks = max(0, min(100, round(norm_knee * 100)))
        scores.append(("knee_bend", ks, SKATING_METRIC_WEIGHTS["knee_bend"]))
        if diff > 10:
            feedbacks.append({"metric":"Knee Bend","status":"too_high","icon":"⬆️",
                "value":f"Your knee angle: {uk:.0f}° vs {profile['name']}'s target ~{tk}°",
                "tip":f"Bend deeper. {profile['name']} sustains an unusually low skating position — your quads should burn.",
                "drill":"60-second wall sits immediately before on-ice sessions to pre-load the position."})
        elif diff < -10:
            feedbacks.append({"metric":"Knee Bend","status":"too_low","icon":"⬇️",
                "value":f"Your knee angle: {uk:.0f}° vs {profile['name']}'s target ~{tk}°",
                "tip":"Very deep bend — confirm you maintain it through the full stride rather than just at one moment.",
                "drill":"Video your full stride cycle and check depth is consistent, not just at the low point."})
        else:
            feedbacks.append({"metric":"Knee Bend","status":"good","icon":"✅",
                "value":f"Knee angle {uk:.0f}° — close to {profile['name']}'s profile",
                "tip":"Solid bend. Build endurance so you hold this depth in the third period.",
                "drill":"Resistance-band skating drills to maintain depth under load."})

        # Forward lean
        ul = user_metrics.get("forward_lean", 5)
        tl = ref["forward_lean_deg"]
        dl = ul - tl
        ls = max(0, 100 - abs(dl) * 4)
        scores.append(("forward_lean", ls, SKATING_METRIC_WEIGHTS["forward_lean"]))
        if dl < -5:
            feedbacks.append({"metric":"Forward Lean","status":"too_upright","icon":"🧍",
                "value":f"Your lean: {ul:.1f}° vs {profile['name']}'s ~{tl}°",
                "tip":f"{profile['name']} leans aggressively forward to maximize horizontal propulsion. Upright posture kills top speed.",
                "drill":"Lean shins hard into boot tongues when pushing — if you're not touching the tongue you're too upright."})
        elif dl > 8:
            feedbacks.append({"metric":"Forward Lean","status":"too_forward","icon":"⤵️",
                "value":f"Your lean: {ul:.1f}° vs {profile['name']}'s ~{tl}°",
                "tip":"Excessive lean can destabilize edge control. Find the balance: aggressive but controlled.",
                "drill":"Puck-handling drills naturally regulate lean — incorporate them in skating sessions."})
        else:
            feedbacks.append({"metric":"Forward Lean","status":"good","icon":"✅",
                "value":f"Forward lean {ul:.1f}° — well-calibrated",
                "tip":f"Matches {profile['name']}'s aggressive-yet-balanced profile.",
                "drill":"Maintain this lean through direction changes and acceleration bursts."})

        # Stride length — normalized: 0 = no stride, pro target = 100, above target = 100
        us = user_metrics.get("stride_length_ratio", 1.0)
        ts = ref["stride_length_ratio"]
        ds = us - ts
        norm_stride = min(1.0, us / max(0.1, ts))
        ss = max(0, round(norm_stride * 100))
        scores.append(("stride_length", ss, SKATING_METRIC_WEIGHTS["stride_length"]))
        if ds < -0.15:
            feedbacks.append({"metric":"Stride Length","status":"short","icon":"📏",
                "value":f"Stride ratio: {us:.2f} vs {profile['name']}'s ~{ts:.2f}",
                "tip":f"{profile['name']} achieves exceptional reach per stride. Short strides mean more energy per meter.",
                "drill":"Cone drill: set cones wider than comfortable and force full extension to reach each one."})
        else:
            feedbacks.append({"metric":"Stride Length","status":"good","icon":"✅",
                "value":f"Stride length ratio {us:.2f} — good reach",
                "tip":"Pair with complete ankle plantarflexion (toe flick) on every push to extract maximum propulsion.",
                "drill":"Resistance bands around ankles to strengthen full-extension push-off."})

        # Hip extension — normalized: 90° (right angle) = 0, pro target = 100
        uh = user_metrics.get("hip_extension", 150)
        th = ref["hip_extension_deg"]
        dh = th - uh
        norm_hip = max(0.0, (uh - 90) / max(1, th - 90))
        hs = max(0, min(100, round(norm_hip * 100)))
        scores.append(("hip_extension", hs, SKATING_METRIC_WEIGHTS["hip_extension"]))
        if dh > 10:
            feedbacks.append({"metric":"Hip Extension","status":"incomplete","icon":"🦵",
                "value":f"Your max hip extension: {uh:.0f}° vs {profile['name']}'s ~{th}°",
                "tip":f"{profile['name']} achieves near-full hip extension on every single stride. Incomplete extension wastes the most powerful phase of your push.",
                "drill":"Single-leg deadlifts and hip-flexor stretching, then 'tall' push-offs on ice."})
        else:
            feedbacks.append({"metric":"Hip Extension","status":"good","icon":"✅",
                "value":f"Hip extension {uh:.0f}° — strong",
                "tip":f"Excellent extension. Focus on replicating it under fatigue in late-game situations.",
                "drill":"Third-period simulation drills maintaining technique when tired."})

        # Ankle dorsiflexion (plantarflexion angle at push-off)
        ua = (user_metrics.get("left_ankle_angle", 0) + user_metrics.get("right_ankle_angle", 0)) / 2
        ta = ref.get("ankle_plantarflexion_deg", 155)
        da = ua - ta
        anks = max(0, min(100, round(100 - abs(da) * 2.5)))
        scores.append(("ankle_dorsiflexion", anks, SKATING_METRIC_WEIGHTS["ankle_dorsiflexion"]))
        if da < -15:
            feedbacks.append({"metric":"Ankle Dorsiflexion","status":"incomplete","icon":"🦶",
                "value":f"Your ankle angle: {ua:.0f}° vs {profile['name']}'s ~{ta:.0f}°",
                "tip":f"{profile['name']} achieves deep ankle flexion pressing the shin into the boot tongue — this loads the calf and generates the 'toe flick' that adds propulsion at the end of each stride.",
                "drill":"Ankle mobility circles + calf raises on a step to build plantarflexion range."})
        elif da > 15:
            feedbacks.append({"metric":"Ankle Dorsiflexion","status":"too_stiff","icon":"🦶",
                "value":f"Your ankle angle: {ua:.0f}° vs {profile['name']}'s ~{ta:.0f}°",
                "tip":"Very high ankle angle — ensure you are fully pressing through the toe at the end of each push-off rather than lifting the heel early.",
                "drill":"Toe-press drills on ice: hold each push until your toe is the last thing to leave the ice."})
        else:
            feedbacks.append({"metric":"Ankle Dorsiflexion","status":"good","icon":"✅",
                "value":f"Ankle angle {ua:.0f}° — solid push-off flexion",
                "tip":"Good ankle engagement. Focus on timing — the toe flick should be the final movement of each stride.",
                "drill":"Single-leg balance on skate blade (off-ice) to improve proprioception and ankle stability."})

        # Symmetry
        usym = user_metrics.get("symmetry", 0.8)
        tsym = ref["glute_drive_symmetry"]
        syms = max(0, min(100, usym * 100))
        scores.append(("symmetry", syms, SKATING_METRIC_WEIGHTS["symmetry"]))
        if usym < 0.85:
            feedbacks.append({"metric":"Bilateral Symmetry","status":"asymmetric","icon":"⚖️",
                "value":f"Symmetry score: {usym:.2f} (target ≥{tsym:.2f})",
                "tip":"Left-right imbalance detected. Asymmetric skating is predictable and limits agility in both directions.",
                "drill":"Single-leg skating: 20 strides left only, then 20 right. Video-compare the two."})
        else:
            feedbacks.append({"metric":"Bilateral Symmetry","status":"good","icon":"✅",
                "value":f"Good balance ({usym:.2f})",
                "tip":f"{profile['name']} has near-perfect symmetry. Maintain yours through fatigue and tight turns.",
                "drill":"Backward skating drills reinforce symmetrical movement patterns."})

    elif mode == "shooting":
        # Hip rotation
        uhr = user_metrics.get("hip_rotation", 20)
        thr = ref["hip_rotation_deg"]
        dhr = thr - uhr
        hrs = max(0, min(100, round(100 - dhr * 1.5)))
        scores.append(("hip_rotation", hrs, SHOOTING_METRIC_WEIGHTS["hip_rotation"]))
        if dhr > 15:
            feedbacks.append({"metric":"Hip Rotation","status":"insufficient","icon":"🌀",
                "value":f"Hip rotation: {uhr:.0f}° vs {profile['name']}'s ~{thr}°",
                "tip":f"{profile['name']}'s shot power starts in the legs and hips — not the arms. Drive your back hip through on every shot.",
                "drill":"Medicine ball rotational throws. Feel the hip-shoulder sequence before adding the stick."})
        else:
            feedbacks.append({"metric":"Hip Rotation","status":"good","icon":"✅",
                "value":f"Hip rotation {uhr:.0f}° — strong kinetic chain",
                "tip":"Good hip engagement. Ensure hips initiate BEFORE your arms pull — sequence is everything.",
                "drill":"Slow-motion shadow shots focusing on hips moving first."})

        # Shoulder alignment
        ust = user_metrics.get("shoulder_tilt", 5)
        tst = ref["shoulder_tilt_deg"]
        sts = max(0, 100 - abs(ust-tst) * 2)
        scores.append(("shoulder_alignment", sts, SHOOTING_METRIC_WEIGHTS["shoulder_alignment"]))
        feedbacks.append({"metric":"Shoulder Alignment","status":"good" if abs(ust-tst)<5 else "adjust","icon":"🎯",
            "value":f"Shoulder tilt: {ust:.1f}° vs {profile['name']}'s ~{tst}°",
            "tip":f"{profile['name']} uses shoulder tilt to generate extra stick flex and conceal the shot direction.",
            "drill":"Mirror drills: practice shoulder pre-set position before each shot."})

        # Weight transfer
        uwt = user_metrics.get("weight_transfer", 0.5)
        twt = ref["weight_transfer_pct"] / 100
        wts = max(0, 100 - abs(uwt - twt) * 200)
        scores.append(("weight_transfer", wts, SHOOTING_METRIC_WEIGHTS["weight_transfer"]))
        feedbacks.append({"metric":"Weight Transfer","status":"good" if abs(uwt-twt)<0.1 else "adjust","icon":"⚖️",
            "value":f"Body position {uwt*100:.0f}% horizontal (target ~{twt*100:.0f}%)",
            "tip":f"{profile['name']} drives ~{ref['weight_transfer_pct']}% of weight onto the front foot at release. Commit through the shot.",
            "drill":"Shoot off a resistance band that pulls you backward — forces full weight commit."})

        # Release point
        urp = user_metrics.get("release_point", 0.3)
        trp = ref["release_point_body_ratio"]
        rps = max(0, 100 - abs(urp-trp) * 80)
        scores.append(("release_point", rps, SHOOTING_METRIC_WEIGHTS["release_point"]))
        feedbacks.append({"metric":"Release Point","status":"good" if abs(urp-trp)<0.08 else "adjust","icon":"🏒",
            "value":f"Release height ratio: {urp:.2f} vs {profile['name']}'s ~{trp:.2f}",
            "tip":f"{profile['name']} releases at mid-body height to maximize stick flex and shot velocity.",
            "drill":"Mark a strip of tape on the boards at hip height — release exactly at that point each shot."})

    # Weighted overall — capped at 100 so the score is always interpretable as a percentage
    total_w = sum(w for _, _, w in scores)
    overall = min(100, sum(s * w for _, s, w in scores) / (total_w + 1e-8))

    key_trait_feedback = {
        "metric": f"Signature Traits of {profile['name']}",
        "status": "info",
        "icon": "⭐",
        "value": "Characteristics to develop:",
        "traits": ref["key_traits"],
        "cues": ref.get("coaching_cues", []),
    }

    return {
        "pro_name": profile["name"],
        "pro_style": profile["style"],
        "pro_color": profile["color"],
        "overall_score": round(overall),
        "score_breakdown": {name: round(s) for name, s, _ in scores},
        "feedbacks": feedbacks,
        "key_traits": key_trait_feedback,
    }


# ─── Calibration pipeline ────────────────────────────────────────────────────

# Which profile fields can be derived from pose landmarks, and how to label them.
CALIBRATION_FIELDS = {
    "skating": {
        "knee_bend_angle_deg":      "Knee Bend Angle (°)",
        "hip_extension_deg":        "Hip Extension (°)",
        "forward_lean_deg":         "Forward Lean (°)",
        "stride_length_ratio":      "Stride Length Ratio",
        "ankle_plantarflexion_deg": "Ankle Plantarflexion (°)",
        "glute_drive_symmetry":     "Bilateral Symmetry (0–1)",
        "stride_cadence_hz":        "Stride Cadence (Hz)",
    },
    "shooting": {
        "shoulder_tilt_deg":        "Shoulder Tilt (°)",
        "hip_rotation_deg":         "Hip Rotation (°)",
        "forward_lean_deg":         "Forward Lean (°)",
        "weight_transfer_pct":      "Weight Transfer (%)",
        "release_point_body_ratio": "Release Point Ratio",
    },
}

# Fields that cannot be derived from pose landmarks — shown for info only.
NON_MEASURABLE = {
    "skating":  ["stride_cadence_hz"],   # can estimate but low confidence
    "shooting": ["stick_flex_pct", "wrist_snap_velocity"],
}


def calibrate_video(video_path: str, mode: str) -> dict:
    """
    Extract representative biomechanical values from a pro player's video
    using the same MediaPipe pipeline as the user analysis.

    Returns a dict with:
      measured   — {field: value} for all measurable fields
      frame_count, scan_frames, confidence, notes
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"error": f"Cannot open video: {video_path}"}

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.20,
        min_pose_presence_confidence=0.20,
        min_tracking_confidence=0.20,
    )

    SCAN_STEP   = max(1, int(fps / 5))
    all_metrics = []
    total_scan  = 0
    total_fi    = 0

    with PoseLandmarker.create_from_options(options) as detector:
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if total_fi % SCAN_STEP == 0:
                total_scan += 1
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                res = detector.detect(Image(image_format=ImageFormat.SRGB, data=rgb))
                if res.pose_landmarks:
                    lms       = res.pose_landmarks[0]
                    world_lms = res.pose_world_landmarks[0] if res.pose_world_landmarks else None
                    m = extract_frame_metrics(lms, w, h, world_lms)
                    all_metrics.append(m)
            total_fi += 1
    cap.release()

    if not all_metrics:
        return {"error": "No pose detected — ensure the athlete is clearly visible and well-lit throughout the clip."}

    confidence = len(all_metrics) / max(1, total_scan)
    measured   = {}
    notes      = []

    if mode == "skating":
        knee_a  = [m["knee_angle"]          for m in all_metrics if m.get("knee_angle", 0)         > 30]
        hip_e   = [m["hip_extension"]        for m in all_metrics if m.get("hip_extension", 0)      > 90]
        leans   = [m["forward_lean"]         for m in all_metrics]
        str_rs  = [m["stride_length_ratio"]  for m in all_metrics]
        ankles  = ([m["left_ankle_angle"]    for m in all_metrics if m.get("left_ankle_angle", 0)   > 10]
                 + [m["right_ankle_angle"]   for m in all_metrics if m.get("right_ankle_angle", 0)  > 10])
        syms    = [m["symmetry"]             for m in all_metrics]

        if len(knee_a) >= 3:
            # 15th percentile = the characteristic deep-bend position
            measured["knee_bend_angle_deg"] = round(float(np.percentile(knee_a, 15)), 1)
        else:
            notes.append("Too few valid knee angle readings — use a clip with more continuous skating.")

        if len(hip_e) >= 3:
            # 85th percentile = representative full extension
            measured["hip_extension_deg"] = round(float(np.percentile(hip_e, 85)), 1)

        if leans:
            measured["forward_lean_deg"] = round(float(np.median(leans)), 1)

        if str_rs:
            # 75th percentile = typical extended stride (not recovery)
            measured["stride_length_ratio"] = round(float(np.percentile(str_rs, 75)), 2)

        if ankles:
            # 85th percentile = push-off plantarflexion
            measured["ankle_plantarflexion_deg"] = round(float(np.percentile(ankles, 85)), 1)

        if syms:
            measured["glute_drive_symmetry"] = round(float(np.mean(syms)), 3)

        # Stride cadence: count feet-together events (recovery phase) per second
        if len(str_rs) >= 4:
            THRESH  = 0.20
            below   = [r < THRESH for r in str_rs]
            # Rising edge: was below threshold, now above → one stride cycle completed
            cycles  = sum(1 for i in range(1, len(below)) if below[i - 1] and not below[i])
            dur_sec = total_fi / fps
            if cycles >= 2 and dur_sec > 0:
                measured["stride_cadence_hz"] = round(cycles / dur_sec, 2)
            else:
                notes.append("Too few stride cycles detected for cadence estimate — use a 5+ second clip.")

    elif mode == "shooting":
        sh_tilts = [m.get("shoulder_tilt", 0) for m in all_metrics]
        if not any(t > 5 for t in sh_tilts):
            return {"error": "No shooting motion detected — shoulder tilt never exceeded 5°. Ensure the clip shows an actual shot."}

        peak_i    = int(np.argmax(sh_tilts))
        peak_meta = all_metrics[peak_i]

        # Small window around peak for stability
        half = max(1, min(3, len(all_metrics) // 6))
        lo   = max(0, peak_i - half)
        hi   = min(len(all_metrics), peak_i + half + 1)
        win  = all_metrics[lo:hi]

        measured["shoulder_tilt_deg"]        = round(float(sh_tilts[peak_i]), 1)
        measured["hip_rotation_deg"]         = round(float(np.mean([m.get("hip_rotation", 0) for m in win])), 1)
        measured["forward_lean_deg"]         = round(float(peak_meta.get("forward_lean", 0)), 1)
        measured["weight_transfer_pct"]      = round(float(peak_meta.get("weight_transfer", 0.5) * 100), 1)
        measured["release_point_body_ratio"] = round(float(max(m.get("release_point", 0) for m in win)), 3)

        if peak_i == 0:
            notes.append("Shot peak at first frame — start the clip slightly before the shot motion begins.")
        if sh_tilts[peak_i] < 10:
            notes.append("Low peak shoulder tilt — confirm this is an actual shot rather than a skating clip.")

    else:
        return {"error": f"Unknown mode: {mode}"}

    if confidence < 0.70:
        notes.append(f"Pose detected in only {confidence * 100:.0f}% of frames — try a brighter, less-obstructed clip.")

    return {
        "mode":        mode,
        "measured":    measured,
        "frame_count": len(all_metrics),
        "scan_frames": total_scan,
        "confidence":  round(confidence, 2),
        "notes":       notes,
    }


# ─── Main analysis pipeline ───────────────────────────────────────────────────

def analyze_video(video_path, mode="skating", output_dir="processed"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(video_path).stem

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.20,
        min_pose_presence_confidence=0.20,
        min_tracking_confidence=0.20,
    )

    # ── Pass 1: coarse scan of the entire video (~5 frames/sec) ──────────────
    SCAN_STEP   = max(1, int(fps / 5))
    scan_frames = []   # (frame_idx, bgr, lm_coords, metrics)

    with PoseLandmarker.create_from_options(options) as detector:
        fi = 0
        while cap.isOpened():
            ret, bgr = cap.read()
            if not ret:
                break
            if fi % SCAN_STEP == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                res = detector.detect(Image(image_format=ImageFormat.SRGB, data=rgb))
                if res.pose_landmarks:
                    lms       = res.pose_landmarks[0]
                    world_lms = res.pose_world_landmarks[0] if res.pose_world_landmarks else None
                    lm_coords = [(lm.x, lm.y) for lm in lms]
                    metrics   = extract_frame_metrics(lms, w, h, world_lms)
                    scan_frames.append((fi, bgr.copy(), lm_coords, metrics))
            fi += 1
    cap.release()

    if not scan_frames:
        return {
            "error": "No pose detected. Ensure the skater/shooter is clearly visible and well-lit.",
            "annotated_frames": [],
            "comparisons": {},
            "user_metrics": {},
            "total_frames_analyzed": 0,
        }

    # ── Pass 2: identify the active motion region ─────────────────────────────
    scans_per_sec = fps / SCAN_STEP

    # ── Stride-ratio signals — primary motion discriminator ──────────────────
    # stride_length_ratio = |left_ankle_x – right_ankle_x| / torso_length.
    #
    # The single most reliable skating/shooting discriminator found through
    # empirical testing on real video:
    #
    #   stride_ratio_min  — MINIMUM ratio across the whole video.
    #     Skaters ALWAYS bring their feet together during stride recovery
    #     (ratio → 0.00–0.15).  Shooters never do: their feet stay at least
    #     shoulder-width apart throughout (ratio ≥ 0.20+).
    #     Tested across 7 videos: skating min = 0.00–0.09, shooting min = 0.27–0.30.
    #
    #   stride_ratio_std  — global standard deviation (secondary signal).
    #     Skating produces high variance (0.38–0.47) from the stride cycle.
    #     Shooting produces low variance (0.14–0.19) from a stable stance.
    #
    # All signals are camera-angle independent (ankle positions relative to each
    # other and to torso length).
    all_stride_ratios = [e[3].get("stride_length_ratio", 0.0) for e in scan_frames]
    stride_ratio_min  = float(min(all_stride_ratios))  if all_stride_ratios else 1.0
    stride_ratio_std  = float(np.std(all_stride_ratios))  if all_stride_ratios else 0.0
    stride_ratio_mean = float(np.mean(all_stride_ratios)) if all_stride_ratios else 0.0

    if mode == "skating":
        # ── Stride + arm-swing quality scorer ────────────────────────────────
        # Only valid skating poses are considered (reasonable knee/hip values).
        valid = [
            e for e in scan_frames
            if 60 < e[3].get("knee_angle", 0) < 180 and e[3].get("hip_extension", 0) > 5
        ]

        if len(valid) < 3:
            candidates = valid or scan_frames
            best_avg   = 0.0
        else:
            # Per-frame stride quality (0–1):
            #   knee bend depth   35 % — lower angle = deeper, more powerful bend
            #   hip extension     30 % — fuller extension = stronger push-off
            #   stride spread     25 % — wider foot separation = longer, visible stride
            #   forward lean      10 % — 10-20° is ideal skating lean
            def _stride_q(met):
                kq = max(0.0, min(1.0, (180 - met.get("knee_angle", 180)) / 80.0))
                hq = max(0.0, min(1.0, (met.get("hip_extension", 90) - 90) / 85.0))
                sq = min(1.0, met.get("stride_length_ratio", 0) / 1.3)
                lq = max(0.0, 1.0 - abs(met.get("forward_lean", 0) - 15) / 20.0)
                return kq * 0.35 + hq * 0.30 + sq * 0.25 + lq * 0.10

            # Arm-swing velocity between consecutive valid frames (normalised 0–1).
            # Arms swinging freely = the stride is at full effort, not just gliding.
            lw_i = int(L.LEFT_WRIST);  rw_i = int(L.RIGHT_WRIST)
            arm_vel = [0.0]
            for i in range(1, len(valid)):
                p, c = valid[i-1][2], valid[i][2]
                lv = math.hypot(c[lw_i][0]-p[lw_i][0], c[lw_i][1]-p[lw_i][1])
                rv = math.hypot(c[rw_i][0]-p[rw_i][0], c[rw_i][1]-p[rw_i][1])
                arm_vel.append((lv + rv) / 2)
            max_arm = max(arm_vel) + 1e-8

            scores = [
                _stride_q(valid[i][3]) * 0.80 + (arm_vel[i] / max_arm) * 0.20
                for i in range(len(valid))
            ]

            # Sliding window: pick the contiguous stretch (~60 % of valid frames,
            # min 6) with the highest average quality score.
            WIN = max(6, min(len(valid), int(len(valid) * 0.60)))
            if len(valid) <= WIN:
                candidates = valid
                best_avg   = sum(scores) / len(scores) if scores else 0.0
            else:
                best_start, best_avg = 0, -1.0
                for s in range(len(valid) - WIN + 1):
                    avg = sum(scores[s : s + WIN]) / WIN
                    if avg > best_avg:
                        best_avg, best_start = avg, s
                candidates = valid[best_start : best_start + WIN]

        # ── Skating motion gate ───────────────────────────────────────────────
        # Three signals, all required:
        #
        # 1. _has_stride_cycle()  ← PRIMARY
        #    In skating, the feet snap together (ratio→0) IMMEDIATELY after or
        #    before full extension (ratio→1+).  There is always a frame with
        #    ratio > 0.65 within ±2 scan frames of the minimum.
        #    In shooting, the feet approach a low stance GRADUALLY (0.8→0.54→
        #    0.23→0.01) so the nearest high value is 4+ frames away.
        #    Empirically verified against all 7 test videos.
        #
        # 2. stride_ratio_std ≥ 0.25  ← SECONDARY
        #    Catches gliding clips where the skater never fully extends.
        #
        # 3. best_avg ≥ 0.20  ← pose quality floor.

        def _has_stride_cycle(ratios, min_thresh=0.15, high_thresh=0.65, proximity=2):
            """True if the global minimum (< min_thresh) has a high value
            (> high_thresh) within ±proximity scan frames — the hallmark of
            the skating stride cycle (extension then immediate recovery)."""
            if not ratios or min(ratios) >= min_thresh:
                return False
            min_idx = min(range(len(ratios)), key=lambda i: ratios[i])
            lo = max(0, min_idx - proximity)
            hi = min(len(ratios), min_idx + proximity + 1)
            return any(ratios[i] > high_thresh for i in range(lo, hi))

        cycle_ok = _has_stride_cycle(all_stride_ratios)
        print(f"[SKATE GATE] best_avg={best_avg:.3f} "
              f"stride_ratio_min={stride_ratio_min:.3f} stride_ratio_std={stride_ratio_std:.3f} "
              f"cycle_ok={cycle_ok} candidates={len(candidates)} valid={len(valid)}", flush=True)

        if best_avg < 0.20 or stride_ratio_std < 0.25 or not cycle_ok:
            return {
                "error": (
                    "No skating stride detected. Make sure the video shows active "
                    "skating with visible knee bend, hip drive, and stride. "
                    "If the skater is just standing or gliding, try a clip where "
                    "they are accelerating or striding at full speed."
                ),
                "annotated_frames": [], "comparisons": {},
                "user_metrics": {}, "total_frames_analyzed": 0,
            }

    else:  # shooting
        # ── Two-component release detector ───────────────────────────────────
        # All three shooting metrics normalised to 0-100 so none dominates:
        #   shoulder_tilt  0-45°  → ×2.22
        #   hip_rotation   0-90°  → ×1.11
        #   release_point  0-0.5  → ×200
        # Velocity term catches the actual firing moment (rapid metric change).

        # Shoulder_tilt is the most reliable camera-angle-independent shooting
        # signal (3D world landmarks measure actual vertical shoulder drop).
        # hip_rotation is removed from scoring entirely: it reads ~80° for any
        # video where the subject travels across the frame (hips perpendicular to
        # camera), making it useless as a discriminator.
        def _val_score(met):
            return (met.get("shoulder_tilt",              0) * 4.44 * 0.55
                  + max(0, met.get("release_point",       0)) * 200 * 0.45)

        def _vel_score(met, prev):
            drp = abs(max(0, met.get("release_point", 0)) - max(0, prev.get("release_point", 0)))
            dst = abs(met.get("shoulder_tilt", 0) - prev.get("shoulder_tilt", 0))
            return drp * 400 * 0.60 + dst * 4.0 * 0.40

        val_scores = [_val_score(e[3]) for e in scan_frames]
        vel_scores = [0.0] + [
            _vel_score(scan_frames[i][3], scan_frames[i-1][3])
            for i in range(1, len(scan_frames))
        ]
        combined = [v + vel_scores[i] for i, v in enumerate(val_scores)]
        peak_i   = max(range(len(combined)), key=lambda i: combined[i])

        # ── Shooting motion gate ──────────────────────────────────────────────
        peak_sh_tilt = scan_frames[peak_i][3].get("shoulder_tilt", 0)

        # Local window around peak — at least 8 scan frames so we always have
        # enough context even when the peak lands at the very start or end.
        MIN_LOCAL = 8
        lhalf   = max(1, int(scans_per_sec * 0.5))
        w_start = max(0, peak_i - lhalf)
        w_end   = min(len(scan_frames), peak_i + lhalf + 1)
        if w_end - w_start < MIN_LOCAL:
            w_end   = min(len(scan_frames), w_start + MIN_LOCAL)
            w_start = max(0, w_end - MIN_LOCAL)
        local_window  = scan_frames[w_start:w_end]
        local_ratios  = [e[3].get("stride_length_ratio", 0.0) for e in local_window]
        local_min     = min(local_ratios) if local_ratios else 1.0
        local_std     = float(np.std(local_ratios)) if local_ratios else 0.0

        all_sh_tilts = [e[3].get("shoulder_tilt", 0) for e in scan_frames]
        print(f"[SHOOT GATE] peak_i={peak_i}/{len(scan_frames)} peak_sh_tilt={peak_sh_tilt:.1f}° "
              f"global_min={stride_ratio_min:.3f} local_min={local_min:.3f} local_std={local_std:.3f} "
              f"global_std={stride_ratio_std:.3f} "
              f"sh_tilt avg={sum(all_sh_tilts)/len(all_sh_tilts):.1f}° "
              f"combined_peak={combined[peak_i]:.1f}", flush=True)

        # Signal 1 (PRIMARY): feet came together near the shot peak.
        # Skating recovery always drops ratio below 0.12.
        # Shooting stances never do (tested minimum: 0.27 across all shot videos).
        if local_min < 0.12:
            return {
                "error": (
                    "No shooting motion detected — this looks like a skating clip. "
                    "Make sure the video shows an actual shot: wind-up, weight "
                    "transfer through the puck, and follow-through."
                ),
                "annotated_frames": [], "comparisons": {},
                "user_metrics": {}, "total_frames_analyzed": 0,
            }

        # Signal 2 (SECONDARY): high variance in the local window = feet still cycling.
        if local_std > 0.25:
            return {
                "error": (
                    "No shooting motion detected — this looks like a skating clip. "
                    "Make sure the video shows an actual shot: wind-up, weight "
                    "transfer through the puck, and follow-through."
                ),
                "annotated_frames": [], "comparisons": {},
                "user_metrics": {}, "total_frames_analyzed": 0,
            }

        # Signal 3: some shoulder engagement required for a real shot.
        if peak_sh_tilt < 8.0:
            return {
                "error": (
                    "No shooting motion detected. Make sure the video shows an "
                    "actual shot — wind-up, shoulder drop through the puck, and "
                    "follow-through. If you're just stickhandling or passing, "
                    "try a clip that includes the full shot motion."
                ),
                "annotated_frames": [], "comparisons": {},
                "user_metrics": {}, "total_frames_analyzed": 0,
            }

        # ── Dynamic load-start detection ─────────────────────────────────────
        # Scan back from the shoulder-tilt peak to find when the shot loaded.
        sh_at_peak     = peak_sh_tilt
        load_threshold = max(5.0, sh_at_peak * 0.35)
        max_lookback   = int(scans_per_sec * 4.0)
        load_start_i   = max(0, peak_i - int(scans_per_sec * 2.5))  # default fallback

        for i in range(peak_i - 1, max(0, peak_i - max_lookback) - 1, -1):
            if scan_frames[i][3].get("shoulder_tilt", 0) < load_threshold:
                load_start_i = min(i + 1, peak_i)
                break

        # Include one frame of setup before the load, then 1.5 s of follow-through.
        w_after    = max(2, int(scans_per_sec * 1.5))
        start_i    = max(0, load_start_i - 1)
        candidates = scan_frames[start_i : peak_i + w_after + 1]

    if not candidates:
        candidates = scan_frames   # fallback: use everything

    # ── Pass 3: evenly pick up to 15 frames from the active region ────────────
    MAX_SAMPLES = 15
    if len(candidates) <= MAX_SAMPLES:
        selected = candidates
    else:
        step     = (len(candidates) - 1) / (MAX_SAMPLES - 1)
        selected = [candidates[round(i * step)] for i in range(MAX_SAMPLES)]

    frame_metrics_list = [e[3] for e in selected]
    raw_frames         = [(e[1], e[2]) for e in selected]

    # ── Aggregate, score, annotate ────────────────────────────────────────────
    user_metrics = aggregate_metrics(frame_metrics_list)
    comparisons  = {key: score_vs_profile(user_metrics, key, mode) for key in PRO_PROFILES}
    best_match   = max(comparisons, key=lambda k: comparisons[k]["overall_score"])

    pro_ref = PRO_PROFILES[best_match]["skating" if mode == "skating" else "shooting"]

    annotated_frames    = []
    clean_frames        = []
    frame_callout_meta  = []
    for fi, (raw_frame, lm_coords) in enumerate(raw_frames):
        # Save clean (unannotated) frame for the Compare Poses interactive canvas
        clean_path = os.path.join(output_dir, f"{base_name}_frame_{fi:04d}_clean.jpg")
        cv2.imwrite(clean_path, raw_frame)
        clean_frames.append(clean_path)

        annotated = raw_frame.copy()
        # Skating: per-frame metrics (knee bend / hip extension change every stride).
        # Shooting: session-level metrics for callouts — shoulder_tilt and release_point
        # are peak metrics that only spike at the moment of release; showing per-frame
        # values makes most frames read 0°. Session metrics give the player their actual
        # best measurement on every annotated frame.
        ann_metrics = frame_metrics_list[fi] if mode == "skating" else user_metrics
        cmeta = annotate_frame(annotated, lm_coords, w, h, ann_metrics, pro_ref, mode)
        frame_callout_meta.append(cmeta)
        out_path = os.path.join(output_dir, f"{base_name}_frame_{fi:04d}.jpg")
        cv2.imwrite(out_path, annotated)
        annotated_frames.append(out_path)

    return {
        "user_metrics": {k: round(v, 3) for k, v in user_metrics.items()},
        "comparisons":  comparisons,
        "best_match":   best_match,
        "total_frames_analyzed": len(frame_metrics_list),
        "annotated_frames":   [os.path.basename(f) for f in annotated_frames],
        "clean_frames":       [os.path.basename(f) for f in clean_frames],
        "frame_callout_meta": frame_callout_meta,
        "mode": mode,
    }
