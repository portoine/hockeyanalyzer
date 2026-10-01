"""
Golf swing analyzer using MediaPipe Pose Landmarker.
Extracts biomechanical metrics from drive, chip, and putt videos
and compares against PGA Tour pro profiles.
"""

import cv2
import json
import numpy as np
import math
import os
from pathlib import Path

from mediapipe.tasks.python.vision.pose_landmarker import (
    PoseLandmarker, PoseLandmarkerOptions, PoseLandmark,
    _RunningMode as RunningMode,
)
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe import Image, ImageFormat

# Shared utilities from hockey analyzer
from analyzer import (
    extract_frame_metrics, MODEL_PATH, score_to_bgr,
    _pro_target_point, CONNECTIONS, angle_between, vertical_angle, pt,
)
from golf_profiles import (
    GOLF_PRO_PROFILES, DRIVE_METRIC_WEIGHTS, CHIP_METRIC_WEIGHTS,
    PUTT_METRIC_WEIGHTS, GOLF_CALIBRATION_FIELDS,
)

GOLF_CALIBRATION_FILE = Path(__file__).parent / "golf_calibration_data.json"
L = PoseLandmark

PRO_GREEN  = (80, 220, 80)
PHASE_COLORS = {
    "setup":         (200, 160,  60),
    "backswing":     ( 60, 160, 255),
    "impact":        ( 60, 220, 100),
    "follow_through": (200, 100, 220),
    "finish":        (255, 180,  60),
}


# ── Calibration helpers ───────────────────────────────────────────────────────

def load_golf_calibration() -> dict:
    if GOLF_CALIBRATION_FILE.exists():
        try:
            return json.loads(GOLF_CALIBRATION_FILE.read_text())
        except Exception:
            return {}
    return {}


def get_golf_pro_ref(pro_key: str, shot_type: str) -> dict:
    profile = GOLF_PRO_PROFILES[pro_key]
    ref     = dict(profile.get(shot_type, profile.get("drive", {})))
    overrides = load_golf_calibration()
    ref.update(overrides.get(pro_key, {}).get(shot_type, {}))
    return ref


# ── Golf-specific metric extraction ──────────────────────────────────────────

def extract_golf_metrics(lms, w, h, world_lms=None) -> dict:
    """
    Wraps extract_frame_metrics and adds golf-specific fields.
    Assumes right-handed golfer (lead arm = left arm).
    """
    m = extract_frame_metrics(lms, w, h, world_lms)

    # Re-label for golf context
    m["spine_angle"]      = m.get("forward_lean", 0)
    m["knee_flex"]        = m.get("knee_angle", 160)     # angle AT the knee
    m["stance_width"]     = m.get("stride_length_ratio", 0)
    m["lead_arm_angle"]   = m.get("left_elbow_angle", 170)   # right-handed
    m["trail_arm_angle"]  = m.get("right_elbow_angle", 120)
    m["weight_dist"]      = m.get("weight_transfer", 0.5) * 100  # 0-100%

    # Shoulder rotation via world landmarks (same formula as hip_rotation but for shoulders)
    if world_lms is not None:
        try:
            ls = world_lms[int(L.LEFT_SHOULDER)]
            rs = world_lms[int(L.RIGHT_SHOULDER)]
            m["shoulder_turn"] = abs(math.degrees(math.atan2(
                abs(rs.z - ls.z),
                abs(rs.x - ls.x) + 1e-8,
            )))
        except Exception:
            m["shoulder_turn"] = m.get("shoulder_tilt", 0)
    else:
        m["shoulder_turn"] = m.get("shoulder_tilt", 0)

    # Hip-to-shoulder ratio (X-factor proxy): low = better coil
    sh = m["shoulder_turn"]
    hp = m.get("hip_rotation", 0)
    m["hip_to_shoulder_ratio"] = round(hp / (sh + 1e-8), 3) if sh > 5 else 0.5

    return m


# ── Swing phase detection ────────────────────────────────────────────────────

def detect_swing_phases(all_metrics: list) -> list:
    """
    Returns a phase label for each entry in all_metrics.
    Phases: setup | backswing | impact | follow_through | finish
    """
    n = len(all_metrics)
    if n < 3:
        return ["setup"] * n

    sh_turns = [m.get("shoulder_turn", 0) for m in all_metrics]
    hip_rots  = [m.get("hip_rotation",  0) for m in all_metrics]
    wts       = [m.get("weight_transfer", 0.5) for m in all_metrics]

    # Top of backswing: max shoulder turn
    peak_i = int(np.argmax(sh_turns))

    # Setup end: last frame before backswing where shoulder_turn < 20% of peak
    setup_thresh = max(8.0, sh_turns[peak_i] * 0.20)
    setup_end = 0
    for i in range(peak_i):
        if sh_turns[i] < setup_thresh:
            setup_end = i

    # Impact: after peak, first frame where shoulder_turn drops below 55% of peak
    # AND weight transfer starts shifting lead-side
    impact_i = min(peak_i + 1, n - 1)
    for i in range(peak_i + 1, n):
        if sh_turns[i] < sh_turns[peak_i] * 0.55:
            impact_i = i
            break

    # Follow-through / finish split
    post_impact = n - impact_i
    follow_end  = impact_i + max(1, post_impact // 2)
    finish_i    = min(follow_end, n - 1)

    phases = []
    for i in range(n):
        if   i <= setup_end:   phases.append("setup")
        elif i <= peak_i:      phases.append("backswing")
        elif i <= impact_i:    phases.append("impact")
        elif i <= finish_i:    phases.append("follow_through")
        else:                  phases.append("finish")
    return phases


# ── Shot type detection ───────────────────────────────────────────────────────

def detect_shot_type(all_metrics: list) -> str:
    if not all_metrics:
        return "drive"
    sh_turns   = [m.get("shoulder_turn",  0) for m in all_metrics]
    stances    = [m.get("stance_width",   0) for m in all_metrics]
    hip_rots   = [m.get("hip_rotation",   0) for m in all_metrics]
    knee_flex  = [m.get("knee_flex",    160) for m in all_metrics]

    max_turn   = max(sh_turns)
    mean_stance = float(np.median(stances))
    max_hip    = max(hip_rots)

    if max_turn < 22 and max_hip < 15:
        return "putt"
    if max_turn < 55 or mean_stance < 0.72:
        return "chip"
    return "drive"


# ── Kinematic sequence scoring ────────────────────────────────────────────────

def kinematic_sequence_score(all_metrics: list, phases: list) -> float:
    """
    Returns 0-100. Good sequence = hips rotate before shoulders in transition.
    Checks: at the backswing peak, does hip_rotation peak AFTER shoulder_turn peaks?
    """
    sh_turns = [m.get("shoulder_turn", 0) for m in all_metrics]
    hip_rots  = [m.get("hip_rotation",  0) for m in all_metrics]
    n = len(all_metrics)
    if n < 4:
        return 60.0

    peak_sh  = int(np.argmax(sh_turns))
    peak_hip = int(np.argmax(hip_rots))

    # Ideal: hip peaks AFTER shoulder turn starts falling (hips lead downswing)
    # hip_peak should be >= shoulder_peak (hips still going while shoulders reverse)
    if peak_hip >= peak_sh:
        base_score = 85.0
    elif peak_hip >= peak_sh - 1:
        base_score = 70.0
    else:
        base_score = 50.0

    # Bonus: how much separation exists at the top
    if sh_turns[peak_sh] > 5:
        ratio = hip_rots[peak_sh] / sh_turns[peak_sh]
        # Ideal ratio ~0.4-0.55 (hips turn less than shoulders = coil)
        sep_score = max(0, 100 - abs(ratio - 0.47) * 200)
        base_score = base_score * 0.6 + sep_score * 0.4

    return round(float(base_score), 1)


# ── Skeleton drawing ──────────────────────────────────────────────────────────

GOLF_CONN_MAP = {
    # (landmark_a, landmark_b): (metric_key, pro_profile_key, tolerance)
    (min(int(L.LEFT_HIP),  int(L.LEFT_SHOULDER)),
     max(int(L.LEFT_HIP),  int(L.LEFT_SHOULDER))):  ("spine_angle", "spine_angle_deg",  12),
    (min(int(L.RIGHT_HIP), int(L.RIGHT_SHOULDER)),
     max(int(L.RIGHT_HIP), int(L.RIGHT_SHOULDER))): ("spine_angle", "spine_angle_deg",  12),
    (min(int(L.LEFT_HIP),  int(L.LEFT_KNEE)),
     max(int(L.LEFT_HIP),  int(L.LEFT_KNEE))):      ("knee_flex",   "knee_flex_deg",    20),
    (min(int(L.RIGHT_HIP), int(L.RIGHT_KNEE)),
     max(int(L.RIGHT_HIP), int(L.RIGHT_KNEE))):     ("knee_flex",   "knee_flex_deg",    20),
    (min(int(L.LEFT_ELBOW), int(L.LEFT_SHOULDER)),
     max(int(L.LEFT_ELBOW), int(L.LEFT_SHOULDER))): ("lead_arm_angle", "lead_arm_angle_deg", 15),
    (min(int(L.LEFT_ELBOW), int(L.LEFT_WRIST)),
     max(int(L.LEFT_ELBOW), int(L.LEFT_WRIST))):    ("lead_arm_angle", "lead_arm_angle_deg", 15),
}


def _golf_connection_color(a, b, metrics, pro_ref):
    ai = min(int(a), int(b))
    bi = max(int(a), int(b))
    entry = GOLF_CONN_MAP.get((ai, bi))
    if not entry:
        return (100, 130, 155)
    metric_key, pro_key, tol = entry
    user_val = metrics.get(metric_key)
    pro_val  = pro_ref.get(pro_key)
    if user_val is None or pro_val is None:
        return (100, 130, 155)
    pct = max(0.0, 100.0 - abs(user_val - pro_val) / (tol + 1e-8) * 100.0)
    return score_to_bgr(pct)


def _round_rect(img, x, y, w, h, r, color, thickness=-1):
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), color, thickness)
    cv2.rectangle(img, (x, y + r), (x + w, y + h - r), color, thickness)
    for cx, cy in [(x+r, y+r), (x+w-r, y+r), (x+r, y+h-r), (x+w-r, y+h-r)]:
        cv2.ellipse(img, (cx, cy), (r, r), 0, *{
            (x+r,y+r):   (180, 270),
            (x+w-r,y+r): (270, 360),
            (x+r,y+h-r): (90, 180),
            (x+w-r,y+h-r): (0, 90),
        }[(cx, cy)], 1, color, thickness)


# ── Frame annotation ──────────────────────────────────────────────────────────

_GOLF_CALLOUT_SEGMENTS = {
    "SPINE ANGLE":    [(L.LEFT_HIP,      L.LEFT_SHOULDER)],
    "KNEE FLEX":      [(L.LEFT_HIP,      L.LEFT_KNEE), (L.LEFT_KNEE, L.LEFT_ANKLE)],
    "SHOULDER TURN":  [(L.LEFT_SHOULDER, L.RIGHT_SHOULDER)],
    "HIP ROTATION":   [(L.LEFT_HIP,      L.RIGHT_HIP)],
    "LEAD ARM":       [(L.LEFT_SHOULDER, L.LEFT_ELBOW), (L.LEFT_ELBOW, L.LEFT_WRIST)],
    "WEIGHT SHIFT":   [(L.LEFT_HIP,      L.RIGHT_HIP)],
}

_GOLF_PHASE_CALLOUTS = {
    "setup":         ["SPINE ANGLE", "KNEE FLEX", "SHOULDER TURN"],
    "backswing":     ["SHOULDER TURN", "HIP ROTATION", "LEAD ARM"],
    "impact":        ["HIP ROTATION", "WEIGHT SHIFT", "SPINE ANGLE"],
    "follow_through": ["SHOULDER TURN", "WEIGHT SHIFT", "LEAD ARM"],
    "finish":        ["SHOULDER TURN", "WEIGHT SHIFT", "KNEE FLEX"],
}

_GOLF_METRIC_MAP = {
    "SPINE ANGLE":   ("spine_angle",    "spine_angle_deg",    "deg"),
    "KNEE FLEX":     ("knee_flex",      "knee_flex_deg",      "deg"),
    "SHOULDER TURN": ("shoulder_turn",  "shoulder_turn_deg",  "deg"),
    "HIP ROTATION":  ("hip_rotation",   "hip_rotation_deg",   "deg"),
    "LEAD ARM":      ("lead_arm_angle", "lead_arm_angle_deg", "deg"),
    "WEIGHT SHIFT":  ("weight_dist",    "weight_transfer_pct", "%"),
}


def annotate_golf_frame(frame, lm_coords, metrics, phase, pro_ref, w, h):
    """
    Draw golf skeleton + phase label + metric callout boxes.
    Returns (annotated_frame, callout_meta_list).
    """
    out = frame.copy()

    def px(idx):
        x, y = lm_coords[int(idx)]
        return (int(x * w), int(y * h))

    # Draw skeleton
    for (a, b) in CONNECTIONS:
        try:
            pa, pb = px(a), px(b)
            col = _golf_connection_color(a, b, metrics, pro_ref)
            cv2.line(out, pa, pb, col, 2, cv2.LINE_AA)
        except Exception:
            pass

    # Joint dots
    KEY_JOINTS = [L.LEFT_HIP, L.RIGHT_HIP, L.LEFT_KNEE, L.RIGHT_KNEE,
                  L.LEFT_SHOULDER, L.RIGHT_SHOULDER,
                  L.LEFT_ELBOW, L.RIGHT_ELBOW, L.LEFT_WRIST, L.RIGHT_WRIST]
    for j in KEY_JOINTS:
        try:
            pj = px(j)
            cv2.circle(out, pj, 4, (220, 220, 220), -1)
        except Exception:
            pass

    # Phase label banner
    phase_col = PHASE_COLORS.get(phase, (180, 180, 180))
    label_txt = phase.upper().replace("_", " ")
    cv2.rectangle(out, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(out, label_txt, (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, phase_col, 2, cv2.LINE_AA)

    # Pro overlay line (green) for spine angle
    try:
        lhp, rhp = px(L.LEFT_HIP), px(L.RIGHT_HIP)
        lsh, rsh = px(L.LEFT_SHOULDER), px(L.RIGHT_SHOULDER)
        mid_hip = ((lhp[0]+rhp[0])//2, (lhp[1]+rhp[1])//2)
        mid_sh  = ((lsh[0]+rsh[0])//2, (lsh[1]+rsh[1])//2)
        t_lean  = pro_ref.get("spine_angle_deg", 40)
        torso_len = math.hypot(mid_sh[0]-mid_hip[0], mid_sh[1]-mid_hip[1])
        sign = 1 if (mid_sh[0]-mid_hip[0]) >= 0 else -1
        pro_sh = (
            mid_hip[0] + sign * int(math.sin(math.radians(t_lean)) * torso_len),
            mid_hip[1] - int(math.cos(math.radians(t_lean)) * torso_len),
        )
        cv2.line(out, mid_hip, pro_sh, PRO_GREEN, 2, cv2.LINE_AA)
        cv2.circle(out, pro_sh, 5, PRO_GREEN, -1)
        cv2.putText(out, "GREEN=PRO TARGET", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, PRO_GREEN, 1, cv2.LINE_AA)
    except Exception:
        pass

    # Build callout boxes
    callouts_to_show = _GOLF_PHASE_CALLOUTS.get(phase, ["SPINE ANGLE", "KNEE FLEX", "HIP ROTATION"])
    raw_callouts = []
    for cname in callouts_to_show:
        if cname not in _GOLF_METRIC_MAP:
            continue
        met_key, pro_key, unit = _GOLF_METRIC_MAP[cname]
        user_val = metrics.get(met_key)
        pro_val  = pro_ref.get(pro_key)
        if user_val is None:
            continue

        pct   = max(0.0, 100.0 - abs(user_val - (pro_val or user_val)) / 15.0 * 100) if pro_val else 60.0
        color = score_to_bgr(pct)
        segs  = _GOLF_CALLOUT_SEGMENTS.get(cname, [])

        # Anchor joint for this callout
        anchor_j = segs[0][0] if segs else L.LEFT_HIP
        try:
            anchor_px = px(anchor_j)
        except Exception:
            continue

        user_str = f"{user_val:.0f}{unit}"
        pro_str  = f"Pro: {pro_val:.0f}{unit}" if pro_val is not None else ""
        raw_callouts.append((cname, user_str, pro_str, anchor_px, color, segs))

    # Place callout boxes (right side of frame, stacked)
    BOX_W, BOX_H = 145, 58
    MARGIN = 10
    callout_meta = []
    for ci, (label, user_str, pro_str, joint_px, color, segs) in enumerate(raw_callouts):
        bx = w - BOX_W - MARGIN
        by = MARGIN + 30 + ci * (BOX_H + 6)
        if by + BOX_H > h:
            break

        # Draw box
        overlay = out.copy()
        cv2.rectangle(overlay, (bx, by), (bx + BOX_W, by + BOX_H), (8, 15, 28), -1)
        cv2.addWeighted(overlay, 0.88, out, 0.12, 0, out)
        cv2.rectangle(out, (bx, by), (bx + BOX_W, by + BOX_H), color, 1)

        # Text
        cv2.putText(out, label,    (bx+8, by+16), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150,190,210), 1, cv2.LINE_AA)
        cv2.putText(out, user_str, (bx+8, by+36), cv2.FONT_HERSHEY_SIMPLEX, 0.60, color,         2, cv2.LINE_AA)
        if pro_str:
            cv2.putText(out, pro_str, (bx+8, by+52), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (80,130,160), 1, cv2.LINE_AA)

        # Leader line to joint
        box_mid_y = by + BOX_H // 2
        cv2.line(out, joint_px, (bx, box_mid_y), color, 1, cv2.LINE_AA)
        cv2.circle(out, joint_px, 7, color, -1)

        # Segment pixels for hover effect
        segs_px = []
        for (sa, sb) in segs:
            try:
                segs_px.append([list(px(sa)), list(px(sb))])
            except Exception:
                pass

        # Pro segment: spine angle reference
        pro_seg = []
        if segs_px:
            p1 = segs_px[0][0]
            p2 = segs_px[0][1]
            if len(segs_px) > 1:
                p2 = segs_px[-1][1]
            pro_seg = [p1, p2]

        color_hex = "#{:02x}{:02x}{:02x}".format(color[2], color[1], color[0])
        callout_meta.append({
            "label":    label,
            "user_str": user_str,
            "pro_str":  pro_str,
            "joint":    list(joint_px),
            "box":      [bx, by, BOX_W, BOX_H],
            "color":    color_hex,
            "segments": segs_px,
            "pro_seg":  pro_seg,
        })

    return out, callout_meta


# ── Scoring ───────────────────────────────────────────────────────────────────

def _pct_score(user, target, tol):
    return max(0.0, 100.0 - abs(user - target) / (tol + 1e-8) * 100.0)


def score_golf_vs_profile(phase_metrics: dict, user_metrics: dict,
                           pro_key: str, shot_type: str,
                           kinematic_score: float) -> dict:
    profile = GOLF_PRO_PROFILES[pro_key]
    ref     = get_golf_pro_ref(pro_key, shot_type)

    feedbacks   = []
    scores      = []

    def add(key, label, icon, user_val, target_val, tol, tip_good, tip_bad, drill):
        s = _pct_score(user_val, target_val, tol)
        scores.append((key, s))
        diff = user_val - target_val
        if abs(diff) < tol * 0.4:
            feedbacks.append({"metric": label, "status": "good",   "icon": icon,
                "value": f"{user_val:.1f} (target ~{target_val})",
                "tip": tip_good, "drill": drill})
        else:
            direction = "too high" if diff > 0 else "too low"
            feedbacks.append({"metric": label, "status": direction, "icon": icon,
                "value": f"Your {label.lower()}: {user_val:.1f} vs {profile['name']}'s ~{target_val}",
                "tip": tip_bad, "drill": drill})

    # ── Setup metrics ────────────────────────────────────────────────────────
    setup = phase_metrics.get("setup", user_metrics)

    sp = setup.get("spine_angle", user_metrics.get("spine_angle", 35))
    add("setup_spine", "Spine Angle at Address", "📐",
        sp, ref.get("spine_angle_deg", 40), 10,
        f"Good posture angle — matches {profile['name']}'s address position.",
        f"{profile['name']} sets up with a {ref.get('spine_angle_deg', 40)}° spine tilt for optimal rotation.",
        "Stand at address without a club and have someone check your spine angle matches target.")

    kf = setup.get("knee_flex", user_metrics.get("knee_flex", 163))
    add("setup_knee", "Knee Flex at Address", "🦵",
        kf, ref.get("knee_flex_deg", 163), 12,
        "Solid knee flex — athletic and ready to rotate.",
        "Slight athletic flex in both knees allows proper weight shift and rotation.",
        "Squat to a box slowly 3× at address position to find the right flex feel.")

    sw = setup.get("stance_width", user_metrics.get("stance_width", 0.9))
    tgt_sw = ref.get("stance_width_ratio", 1.0)
    add("setup_stance", "Stance Width", "↔️",
        sw, tgt_sw, 0.25,
        "Stance width suits this shot type.",
        f"{profile['name']} uses a ratio of ~{tgt_sw:.2f} for this shot.",
        "Use two clubs on the ground to mark ideal foot width at address.")

    # ── Backswing metrics ─────────────────────────────────────────────────────
    bs = phase_metrics.get("backswing", user_metrics)

    st = user_metrics.get("shoulder_turn", 0)
    tgt_st = ref.get("shoulder_turn_deg", 85)
    add("bs_shoulder", "Peak Shoulder Turn", "🔄",
        st, tgt_st, 18,
        f"Excellent shoulder turn — matching {profile['name']}'s range of motion.",
        f"{profile['name']} achieves ~{tgt_st}° of shoulder turn. More turn = more stored energy.",
        "Cross your arms on your chest and practice full rotation to a mirror.")

    hr = user_metrics.get("hip_rotation", 0)
    tgt_hr = ref.get("hip_rotation_deg", 45)
    add("bs_hip", "Hip Turn at Top", "🏌️",
        hr, tgt_hr, 15,
        "Good hip coil — creating separation from your shoulders.",
        f"Resist the hips while turning shoulders for maximum X-factor coil.",
        "Place a club across your hips and practice limited hip turn while turning shoulders fully.")

    la = user_metrics.get("lead_arm_angle", 165)
    tgt_la = ref.get("lead_arm_angle_deg", 170)
    add("bs_lead_arm", "Lead Arm Position", "💪",
        la, tgt_la, 15,
        "Lead arm staying connected — solid swing plane.",
        f"A straighter lead arm at the top maintains width and power in the swing arc.",
        "Practice swings with a glove under your lead arm to keep it connected to your chest.")

    # ── Impact / kinematic ────────────────────────────────────────────────────
    wt = user_metrics.get("weight_dist", 50)
    tgt_wt = ref.get("weight_transfer_pct", 78)
    add("imp_weight", "Weight Transfer", "⚖️",
        wt, tgt_wt, 15,
        f"Excellent weight transfer — moving like {profile['name']}.",
        f"{profile['name']} shifts ~{tgt_wt}% onto the lead foot through impact.",
        "Practice hitting shots with a weighted resistance band to feel proper lead-side pressure.")

    scores.append(("kinematic", kinematic_score))
    if kinematic_score >= 72:
        feedbacks.append({"metric": "Kinematic Sequence", "status": "good", "icon": "⚡",
            "value": f"Sequence score: {kinematic_score:.0f}/100",
            "tip": "Hips are initiating the downswing before the shoulders — this is the correct power chain.",
            "drill": "Slow-motion downswing drills focusing on feeling the hips move first."})
    else:
        feedbacks.append({"metric": "Kinematic Sequence", "status": "too low", "icon": "⚡",
            "value": f"Sequence score: {kinematic_score:.0f}/100",
            "tip": "Shoulders appear to be leading the downswing — try to feel the hips start first.",
            "drill": "Pause at the top of your backswing for 1 second, then initiate with your hips only."})

    # ── Finish ────────────────────────────────────────────────────────────────
    fin = phase_metrics.get("finish", user_metrics)
    fb = fin.get("symmetry", 0.85)
    tgt_fb = ref.get("finish_balance", 0.92)
    add("fin_balance", "Finish Balance", "🏆",
        fb * 100, tgt_fb * 100, 12,
        f"Balanced finish — fully committed through the shot.",
        f"{profile['name']} finishes in a fully balanced position with ~{tgt_fb*100:.0f}% stability.",
        "Hold your finish position for 3 full seconds after each practice swing.")

    # ── Phase-level score breakdown ───────────────────────────────────────────
    _score_phase_map = {
        "setup_spine": "setup",   "setup_knee":    "setup",   "setup_stance": "setup",
        "bs_shoulder": "backswing","bs_hip":        "backswing","bs_lead_arm":  "backswing",
        "imp_weight":  "impact",  "kinematic":     "impact",
        "fin_balance": "finish",
    }
    phase_score_groups: dict = {"setup": [], "backswing": [], "impact": [], "finish": []}
    for k, s in scores:
        ph = _score_phase_map.get(k, "setup")
        phase_score_groups[ph].append(s)
    phase_scores = {ph: round(sum(v)/len(v)) for ph, v in phase_score_groups.items() if v}

    # Overall score
    total_w = len(scores)
    overall = min(100, sum(s for _, s in scores) / (total_w + 1e-8))

    # Key traits
    key_trait_feedback = {
        "metric": f"Signature Traits of {profile['name']}",
        "status": "info",
        "icon": "⭐",
        "value": "Characteristics to develop:",
        "traits": ref.get("key_traits", []),
        "cues":   ref.get("coaching_cues", []),
    }

    # ── Build phase-keyed string feedbacks for the frontend ──────────────────
    _phase_map = {
        "setup_spine":  "setup",  "setup_knee":    "setup", "setup_stance": "setup",
        "bs_shoulder":  "backswing", "bs_hip":     "backswing", "bs_lead_arm": "backswing",
        "imp_weight":   "impact", "kinematic":     "impact",
        "fin_balance":  "finish",
    }
    phase_feedbacks: dict = {"setup": [], "backswing": [], "impact": [], "finish": []}
    for (key, _), fb in zip(scores, feedbacks):
        phase = _phase_map.get(key, "setup")
        status = fb.get("status", "")
        tip    = fb.get("tip", "")
        val    = fb.get("value", "")
        if status == "good":
            prefix = "✓ "
        elif status in ("too high", "too low"):
            prefix = "▲ "
        else:
            prefix = "– "
        line = f"{prefix}{val}" if val else f"{prefix}{tip}"
        if tip and val:
            line = f"{prefix}{val} — {tip}"
        phase_feedbacks[phase].append(line)

    # ── Metric comparisons table ───────────────────────────────────────────────
    _metric_keys = [
        ("spine_angle_deg",    "spine_angle",    "setup"),
        ("knee_flex_deg",      "knee_flex",       "setup"),
        ("stance_width_ratio", "stance_width",    "setup"),
        ("shoulder_turn_deg",  "shoulder_turn",   None),
        ("hip_rotation_deg",   "hip_rotation",    None),
        ("lead_arm_angle_deg", "lead_arm_angle",  None),
        ("weight_transfer_pct","weight_dist",     None),
        ("hip_to_shoulder_ratio","hip_to_shoulder_ratio", None),
        ("finish_balance",     "symmetry",        "finish"),
    ]
    metric_comps = {}
    for ref_key, user_key, src_phase in _metric_keys:
        pro_val = ref.get(ref_key)
        if pro_val is None:
            continue
        src = phase_metrics.get(src_phase, user_metrics) if src_phase else user_metrics
        user_val = src.get(user_key) if src else None
        if user_key == "symmetry" and user_val is not None:
            user_val = round(user_val * 100, 1)
            pro_val  = round(pro_val  * 100, 1)
        if user_val is not None:
            delta = round(user_val - pro_val, 2)
            pct   = round(max(0, 100 - abs(delta) / max(abs(pro_val), 1) * 100))
            metric_comps[ref_key] = {"user": user_val, "pro": pro_val, "delta": delta, "pct": pct}

    return {
        "pro_name":          profile["name"],
        "style":             profile["style"],   # frontend uses "style"
        "pro_style":         profile["style"],
        "pro_color":         profile["color"],
        "overall_score":     round(overall),
        "score_breakdown":   phase_scores,
        "feedbacks":         phase_feedbacks,
        "coaching_cues":     ref.get("coaching_cues", []),
        "key_traits":        key_trait_feedback,
        "metric_comparisons": metric_comps,
    }


# ── Aggregation helpers ───────────────────────────────────────────────────────

def _aggregate_phase(metrics_list):
    if not metrics_list:
        return {}
    result = {}
    for k in metrics_list[0]:
        vals = [m[k] for m in metrics_list if k in m and isinstance(m[k], (int, float))]
        if vals:
            result[k] = float(np.mean(vals))
    return result


# ── Main pipeline ─────────────────────────────────────────────────────────────

def analyze_golf_video(video_path: str, shot_type: str = "auto",
                        output_dir: str = "processed") -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    os.makedirs(output_dir, exist_ok=True)
    base = Path(video_path).stem[:8]

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=0.20,
        min_pose_presence_confidence=0.20,
        min_tracking_confidence=0.20,
    )

    # ── Pass 1: scan entire video at ~5 fps ──────────────────────────────────
    SCAN_STEP   = max(1, int(fps / 5))
    scan_frames = []   # (frame_idx, bgr, lm_coords, metrics)

    with PoseLandmarker.create_from_options(options) as det:
        fi = 0
        while cap.isOpened():
            ret, bgr = cap.read()
            if not ret:
                break
            if fi % SCAN_STEP == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                res = det.detect(Image(image_format=ImageFormat.SRGB, data=rgb))
                if res.pose_landmarks:
                    lms       = res.pose_landmarks[0]
                    wlms      = res.pose_world_landmarks[0] if res.pose_world_landmarks else None
                    lm_coords = [(lm.x, lm.y) for lm in lms]
                    met       = extract_golf_metrics(lms, w, h, wlms)
                    scan_frames.append((fi, bgr.copy(), lm_coords, met))
            fi += 1
    cap.release()

    if not scan_frames:
        return {
            "error": "No pose detected. Ensure the golfer is clearly visible, well-lit, and in frame throughout the clip.",
            "annotated_frames": [], "frame_callout_meta": [],
            "comparisons": {}, "user_metrics": {}, "total_frames_analyzed": 0,
            "mode": "golf",
        }

    all_metrics = [m for _, _, _, m in scan_frames]

    # ── Gate: require at least some swing motion ──────────────────────────────
    max_turn = max(m.get("shoulder_turn", 0) for m in all_metrics)
    if max_turn < 12:
        return {
            "error": "No golf swing detected — peak shoulder rotation was under 12°. Ensure the clip shows an actual swing.",
            "annotated_frames": [], "frame_callout_meta": [],
            "comparisons": {}, "user_metrics": {}, "total_frames_analyzed": 0,
            "mode": "golf",
        }

    # ── Phase detection + shot type ───────────────────────────────────────────
    phases = detect_swing_phases(all_metrics)

    if shot_type == "auto":
        shot_type = detect_shot_type(all_metrics)

    # ── Aggregate per-phase metrics ────────────────────────────────────────────
    phase_groups = {}
    for met, ph in zip(all_metrics, phases):
        phase_groups.setdefault(ph, []).append(met)

    phase_metrics = {ph: _aggregate_phase(mlist) for ph, mlist in phase_groups.items()}

    # Overall user metrics (aggregate across all frames)
    user_metrics = _aggregate_phase(all_metrics)

    # Kinematic sequence score
    kin_score = kinematic_sequence_score(all_metrics, phases)

    # ── Annotate up to 12 evenly spaced frames ─────────────────────────────
    n = len(scan_frames)
    indices_to_annotate = sorted(set(
        [0] +
        [int(i * (n - 1) / 11) for i in range(1, 11)] +
        [n - 1]
    ))[:12]

    annotated_frames   = []
    frame_callout_meta = []
    annotated_phases   = []

    for idx in indices_to_annotate:
        fi_raw, bgr, lm_coords, met = scan_frames[idx]
        phase = phases[idx]

        # Compare against all 3 pros; use Tiger as the annotated reference
        ref = get_golf_pro_ref("tiger_woods", shot_type)
        ann, cmeta = annotate_golf_frame(bgr, lm_coords, met, phase, ref, w, h)

        fname = f"{base}_golf_{idx:04d}.jpg"
        cv2.imwrite(os.path.join(output_dir, fname), ann)
        annotated_frames.append(fname)
        frame_callout_meta.append(cmeta)
        annotated_phases.append(phase)

    # ── Score vs all pros ──────────────────────────────────────────────────────
    comparisons = {}
    for pk in GOLF_PRO_PROFILES:
        comparisons[pk] = score_golf_vs_profile(
            phase_metrics, user_metrics, pk, shot_type, kin_score
        )

    best_match = max(comparisons, key=lambda k: comparisons[k]["overall_score"])

    return {
        "annotated_frames":    annotated_frames,
        "frame_callout_meta":  frame_callout_meta,
        "user_metrics":        user_metrics,
        "phase_metrics":       phase_metrics,
        "phases":              annotated_phases,
        "shot_type":           shot_type,
        "comparisons":         comparisons,
        "best_match":          best_match,
        "kinematic_score":     kin_score,
        "total_frames_analyzed": len(scan_frames),
        "mode":                "golf",
        "error":               None,
    }
