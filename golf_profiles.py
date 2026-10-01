"""
Biomechanical reference profiles for elite PGA Tour players.
Values are MediaPipe-compatible estimates derived from coaching research
and motion analysis studies. Use the calibration tool to replace these
with values measured directly from real pro footage through the same pipeline.

Metric notes:
  spine_angle_deg        — torso tilt from vertical (forward_lean in MediaPipe)
  knee_flex_deg          — average knee joint angle (lower = more flexed)
  stance_width_ratio     — ankle spread / torso length (stride_length_ratio reused)
  shoulder_turn_deg      — peak rotation of shoulder line (world-landmark atan2)
  hip_rotation_deg       — hip line rotation at same point (hip_rotation metric)
  lead_arm_angle_deg     — left elbow angle for right-handed golfer (left_elbow_angle)
  weight_transfer_pct    — horizontal COM position × 100 (weight_transfer × 100)
  hip_to_shoulder_ratio  — hip turn / shoulder turn — lower = better coil / X-factor
  finish_balance         — bilateral symmetry score at finish (symmetry metric)
"""

GOLF_PRO_PROFILES = {

    "tiger_woods": {
        "name":   "Tiger Woods",
        "team":   "TGR / Nike Era",
        "style":  "power & precision / course management",
        "color":  "#D4170F",   # Sunday red
        "number": 0,

        "drive": {
            "spine_angle_deg":             43,
            "knee_flex_deg":               163,
            "stance_width_ratio":          1.05,
            "shoulder_turn_deg":           90,
            "hip_rotation_deg":            45,
            "lead_arm_angle_deg":          170,
            "weight_transfer_pct":         78,
            "hip_to_shoulder_ratio":       0.50,
            "finish_balance":              0.92,
            "key_traits": [
                "Textbook kinematic sequence — hips initiate the downswing before shoulders unwind",
                "Exceptional X-factor hip/shoulder separation stores elastic energy at the top",
                "Aggressive squat-and-push leg drive through impact generates clubhead speed",
                "Spine angle maintained from address through impact — never stands up early",
                "Head stays behind the ball through impact, releasing only in follow-through",
                "Controlled, repeatable — minimal unnecessary movement anywhere in the swing",
            ],
            "coaching_cues": [
                "Feel your right knee driving toward the target on the downswing",
                "Maintain your spine angle — don't stand up out of your posture through impact",
                "Let the hips CLEAR before your shoulders unwind — hips lead, shoulders follow",
                "Keep the triangle of your arms and chest connected through the takeaway",
            ],
        },

        "chip": {
            "spine_angle_deg":             44,
            "knee_flex_deg":               167,
            "stance_width_ratio":          0.70,
            "shoulder_turn_deg":           42,
            "hip_rotation_deg":            12,
            "lead_arm_angle_deg":          162,
            "weight_transfer_pct":         68,
            "hip_to_shoulder_ratio":       0.29,
            "finish_balance":              0.96,
            "key_traits": [
                "Abbreviated backswing with a crisp, descending strike into the ball",
                "Weight favors the lead side throughout — eliminates any scooping motion",
                "Quiet lower body — arms and shoulder triangle do the work",
                "Hands lead the clubface through impact, creating forward shaft lean",
                "Consistent tempo regardless of distance — clock-pendulum rhythm",
            ],
            "coaching_cues": [
                "Set 60% of your weight forward at address and keep it there all the way through",
                "Imagine brushing the grass forward — don't try to lift the ball into the air",
                "Keep your hands ahead of the clubhead at impact — shaft lean is essential",
                "Minimize hip and leg movement; this is an arms-and-chest shot",
            ],
        },

        "putt": {
            "spine_angle_deg":             48,
            "knee_flex_deg":               165,
            "stance_width_ratio":          0.80,
            "shoulder_turn_deg":           16,
            "hip_rotation_deg":            3,
            "lead_arm_angle_deg":          168,
            "weight_transfer_pct":         52,
            "hip_to_shoulder_ratio":       0.19,
            "finish_balance":              0.98,
            "key_traits": [
                "Pure pendulum motion — shoulders rock, everything else is still",
                "Eyes directly over the ball at address for accurate aim",
                "Zero hip and lower-body movement throughout the stroke",
                "Smooth acceleration through the ball — never decelerates into impact",
                "Head stays down until the ball is well past the hole",
            ],
            "coaching_cues": [
                "Drop a ball from your eye line at address — it should hit the ball on the green",
                "Rock your shoulders like a pendulum; arms just follow along for the ride",
                "Keep your head perfectly still until the ball is well past the cup",
                "Accelerate through impact — a decelerating putter causes misses",
            ],
        },
    },

    "rory_mcilroy": {
        "name":   "Rory McIlroy",
        "team":   "Nike Golf",
        "style":  "explosive speed / elite ball-striking / high launch",
        "color":  "#0057A8",   # Nike blue
        "number": 0,

        "drive": {
            "spine_angle_deg":             42,
            "knee_flex_deg":               162,
            "stance_width_ratio":          1.10,
            "shoulder_turn_deg":           100,
            "hip_rotation_deg":            52,
            "lead_arm_angle_deg":          168,
            "weight_transfer_pct":         82,
            "hip_to_shoulder_ratio":       0.52,
            "finish_balance":              0.90,
            "key_traits": [
                "Fastest hip rotation speed on Tour — primary source of elite clubhead speed",
                "Exceptional hip-shoulder separation at the top (widest X-factor on Tour)",
                "Athletic lower-body squat into transition triggers the speed chain from the ground up",
                "Full, uninhibited release through to a high, balanced finish",
                "Upright spine at address enables a fuller shoulder turn without restriction",
                "Trail heel lifts early on the backswing to allow maximum hip turn",
            ],
            "coaching_cues": [
                "Feel a slight squat as you begin the downswing — drop into your legs first",
                "Allow your trail heel to lift early on the backswing for full rotation",
                "Sense your lead hip pocket firing back and through before anything else",
                "Chase the finish — fully commit to a high, balanced finish position",
            ],
        },

        "chip": {
            "spine_angle_deg":             41,
            "knee_flex_deg":               165,
            "stance_width_ratio":          0.75,
            "shoulder_turn_deg":           38,
            "hip_rotation_deg":            10,
            "lead_arm_angle_deg":          163,
            "weight_transfer_pct":         65,
            "hip_to_shoulder_ratio":       0.26,
            "finish_balance":              0.95,
            "key_traits": [
                "Athletic, neutral setup with minimal manipulation",
                "Clean, direct strike — trusts the loft rather than manufacturing trajectory",
                "Good rhythm even on the shortest of shots",
                "Adjusts trajectory with ball position, not hand action",
            ],
            "coaching_cues": [
                "Play the ball in the center of your stance for a neutral, medium trajectory",
                "Maintain the same grip pressure throughout the shot — light and consistent",
                "Think 'sternum over the ball' to maintain solid contact through impact",
                "Match your follow-through length to your backswing length for consistent distance",
            ],
        },

        "putt": {
            "spine_angle_deg":             44,
            "knee_flex_deg":               164,
            "stance_width_ratio":          0.82,
            "shoulder_turn_deg":           14,
            "hip_rotation_deg":            2,
            "lead_arm_angle_deg":          167,
            "weight_transfer_pct":         51,
            "hip_to_shoulder_ratio":       0.14,
            "finish_balance":              0.97,
            "key_traits": [
                "Rhythmic, flowing stroke rather than mechanical or forced",
                "Consistent tempo — same pace on short and long putts alike",
                "Excellent green reading and decisive commitment to his line",
                "Slight forward press at address naturally activates the stroke",
            ],
            "coaching_cues": [
                "Think of putting as 'rolling' rather than 'hitting' the ball",
                "Use the same pre-shot routine on every single putt, regardless of length",
                "Count 'one' on the backstroke and 'two' on the through-stroke for consistent tempo",
                "Look at your intended target line, not just the hole, when setting up",
            ],
        },
    },

    "scottie_scheffler": {
        "name":   "Scottie Scheffler",
        "team":   "Titleist",
        "style":  "consistency & control / elite ball-striking / World #1",
        "color":  "#006747",   # Masters green
        "number": 0,

        "drive": {
            "spine_angle_deg":             38,
            "knee_flex_deg":               164,
            "stance_width_ratio":          1.02,
            "shoulder_turn_deg":           85,
            "hip_rotation_deg":            40,
            "lead_arm_angle_deg":          172,
            "weight_transfer_pct":         76,
            "hip_to_shoulder_ratio":       0.47,
            "finish_balance":              0.94,
            "key_traits": [
                "Most consistent ball-striking on Tour — the most repeatable swing at the top level",
                "Controlled backswing length — never over-swings, clubs stay on plane",
                "Exceptionally smooth transition — powerful but unhurried at the top",
                "Slight over-the-top loop that he has mastered into a reliable, consistent shape",
                "Rock-solid lower body creates a reliable platform for everything above it",
                "Makes hard shots look routine through superior course management and execution",
            ],
            "coaching_cues": [
                "Focus on repeating the same swing every time rather than chasing maximum distance",
                "Feel a slight pause at the top before initiating the downswing — don't rush",
                "Drive through with your legs — your lower body is your real power source",
                "Commit to your shot shape every single time — pick a path and trust it completely",
            ],
        },

        "chip": {
            "spine_angle_deg":             40,
            "knee_flex_deg":               166,
            "stance_width_ratio":          0.72,
            "shoulder_turn_deg":           40,
            "hip_rotation_deg":            9,
            "lead_arm_angle_deg":          163,
            "weight_transfer_pct":         66,
            "hip_to_shoulder_ratio":       0.23,
            "finish_balance":              0.96,
            "key_traits": [
                "Outstanding touch and feel around the greens — consistently ranks top-5 on Tour",
                "Uses multiple trajectories with the same neutral setup",
                "Exceptional distance control from all types of lies",
                "Quiet, reliable technique that holds up under the greatest pressure",
            ],
            "coaching_cues": [
                "Choke down on the grip slightly for better feel and control",
                "Soften your grip pressure — imagine holding something fragile",
                "Keep your lower body quiet: this is a finesse shot, not a power shot",
                "Visualize exactly where you want the ball to land before you take the club back",
            ],
        },

        "putt": {
            "spine_angle_deg":             46,
            "knee_flex_deg":               166,
            "stance_width_ratio":          0.85,
            "shoulder_turn_deg":           15,
            "hip_rotation_deg":            2,
            "lead_arm_angle_deg":          168,
            "weight_transfer_pct":         52,
            "hip_to_shoulder_ratio":       0.13,
            "finish_balance":              0.98,
            "key_traits": [
                "One of the best putters on Tour — elite performance across all surface speeds",
                "Consistent ball and eye position at address on every single putt",
                "Smooth, confident stroke with zero hesitation or second-guessing",
                "Exceptional green reading and full commitment to every chosen line",
            ],
            "coaching_cues": [
                "Build a consistent pre-putt routine and commit to it every single time",
                "Focus on the process, not the outcome — just roll it on your chosen line",
                "Breathe out slowly and completely just as you begin your backstroke",
                "Keep the putter head low to the ground through and past impact",
            ],
        },
    },
}


# ── Metric weights for overall scoring ───────────────────────────────────────

DRIVE_METRIC_WEIGHTS = {
    "setup":          0.15,
    "shoulder_turn":  0.25,
    "kinematic":      0.25,
    "weight_transfer": 0.20,
    "finish":         0.15,
}

CHIP_METRIC_WEIGHTS = {
    "setup":          0.20,
    "shoulder_turn":  0.20,
    "weight_transfer": 0.35,
    "finish":         0.25,
}

PUTT_METRIC_WEIGHTS = {
    "setup":          0.30,
    "stroke":         0.40,
    "stability":      0.30,
}


# ── Calibration field registry ────────────────────────────────────────────────

GOLF_CALIBRATION_FIELDS = {
    "drive": {
        "spine_angle_deg":           "Spine Angle at Address (°)",
        "knee_flex_deg":             "Knee Flex at Address (°)",
        "stance_width_ratio":        "Stance Width Ratio",
        "shoulder_turn_deg":         "Peak Shoulder Turn (°)",
        "hip_rotation_deg":          "Hip Rotation at Top (°)",
        "lead_arm_angle_deg":        "Lead Arm Angle at Top (°)",
        "weight_transfer_pct":       "Weight Forward at Impact (%)",
        "hip_to_shoulder_ratio":     "Hip/Shoulder Ratio (X-factor)",
        "finish_balance":            "Finish Balance Score (0–1)",
    },
    "chip": {
        "spine_angle_deg":           "Spine Angle (°)",
        "knee_flex_deg":             "Knee Flex (°)",
        "stance_width_ratio":        "Stance Width Ratio",
        "shoulder_turn_deg":         "Peak Shoulder Turn (°)",
        "weight_transfer_pct":       "Weight Forward (%)",
    },
    "putt": {
        "spine_angle_deg":           "Spine Angle (°)",
        "knee_flex_deg":             "Knee Flex (°)",
        "stance_width_ratio":        "Stance Width Ratio",
        "shoulder_turn_deg":         "Stroke Arc (°)",
        "finish_balance":            "Stability Score (0–1)",
    },
}
