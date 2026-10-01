"""
Biomechanical reference profiles for elite NHL players.
Based on documented coaching analyses, sports science research,
and publicly available movement studies of each player.
"""

PRO_PROFILES = {
    "connor_mcdavid": {
        "name": "Connor McDavid",
        "team": "Edmonton Oilers",
        "number": 97,
        "style": "explosive speed / edge mastery",
        "color": "#FF6600",
        "skating": {
            "stride_cadence_hz": 4.5,           # strides per second at top speed
            "knee_bend_angle_deg": 105,          # ideal knee angle at mid-stride
            "hip_extension_deg": 175,            # hip at full extension (push-off)
            "forward_lean_deg": 15,              # torso lean from vertical
            "stride_length_ratio": 1.35,         # stride length / leg length
            "ankle_plantarflexion_deg": 155,     # ankle angle at push-off
            "shin_angle_deg": 22,               # shin forward tilt from vertical
            "glute_drive_symmetry": 0.97,        # 0-1, left/right balance
            "crossover_knee_inside_edge": True,
            "key_traits": [
                "Exceptionally deep knee bend sustained through full stride cycle",
                "Rapid heel recovery close to body minimizing rotational inertia",
                "Full ankle extension (toe flick) generating ~30% extra propulsion",
                "Head stays level — minimal vertical oscillation",
                "Hip drop generates powerful edge engagement during crossovers",
            ],
            "coaching_cues": [
                "Drive your heel UP behind you quickly — don't let it drag",
                "Bend your knees until it 'burns' — McDavid skates lower than almost anyone",
                "Push through the toe completely before recovery",
                "Keep your chest over your lead knee",
            ],
        },
        "shooting": {
            "hip_rotation_deg": 85,
            "shoulder_tilt_deg": 12,
            "weight_transfer_pct": 88,
            "stick_flex_pct": 92,
            "release_point_body_ratio": 0.55,
            "wrist_snap_velocity": "elite",
            "key_traits": [
                "Deceptive release with minimal wind-up telegraphing",
                "Core rotation initiates before arm pull",
                "Elite weight transfer from back to front foot",
            ],
        },
    },

    "nathan_mackinnon": {
        "name": "Nathan MacKinnon",
        "team": "Colorado Avalanche",
        "number": 29,
        "style": "power / relentless acceleration",
        "color": "#6F263D",
        "skating": {
            "stride_cadence_hz": 4.2,
            "knee_bend_angle_deg": 100,
            "hip_extension_deg": 178,
            "forward_lean_deg": 18,
            "stride_length_ratio": 1.45,
            "ankle_plantarflexion_deg": 158,
            "shin_angle_deg": 25,               # shin forward tilt from vertical
            "glute_drive_symmetry": 0.95,
            "crossover_knee_inside_edge": True,
            "key_traits": [
                "Longest effective stride length among top NHLers relative to body size",
                "Aggressive forward lean creates powerful horizontal force vector",
                "Dominant glute and hamstring drive — visibly powerful hip extension",
                "Strong upper-body counter-rotation amplifies stride power",
                "Sustained top speed over longer distances than peers",
            ],
            "coaching_cues": [
                "Push BACK not down — drive the ice away behind you",
                "Lean forward aggressively — feel your shin against your boot tongue",
                "Let your arms swing with more force to power your strides",
                "Full extension on every single push — no short strides",
            ],
        },
        "shooting": {
            "hip_rotation_deg": 90,
            "shoulder_tilt_deg": 15,
            "weight_transfer_pct": 92,
            "stick_flex_pct": 95,
            "release_point_body_ratio": 0.52,
            "wrist_snap_velocity": "elite",
            "key_traits": [
                "Massive power generation from legs and core",
                "One of the hardest shots in the league — elite stick flex utilization",
                "Explosive hip-to-shoulder kinetic chain",
            ],
        },
    },

    "quinn_hughes": {
        "name": "Quinn Hughes",
        "team": "Vancouver Canucks",
        "number": 43,
        "style": "agility / edge transitions / gap control",
        "color": "#00843D",
        "skating": {
            "stride_cadence_hz": 4.3,
            "knee_bend_angle_deg": 108,
            "hip_extension_deg": 170,
            "forward_lean_deg": 12,
            "stride_length_ratio": 1.25,
            "ankle_plantarflexion_deg": 150,
            "shin_angle_deg": 18,               # shin forward tilt from vertical
            "glute_drive_symmetry": 0.99,
            "crossover_knee_inside_edge": True,
            "key_traits": [
                "Near-perfect bilateral symmetry — equally dangerous on both edges",
                "Effortless-looking transitions from forward to backward skating",
                "Exceptional inside-edge control during tight turns",
                "Low center of gravity maintained through all directional changes",
                "Momentum preservation — rarely loses speed through turns",
            ],
            "coaching_cues": [
                "Focus on edge transitions — train your inside edges constantly",
                "Keep your hips LOW through every turn, don't stand up",
                "Practice tight pivots until they feel effortless in both directions",
                "Equal time training your weaker side",
            ],
        },
        "shooting": {
            "hip_rotation_deg": 78,
            "shoulder_tilt_deg": 10,
            "weight_transfer_pct": 82,
            "stick_flex_pct": 85,
            "release_point_body_ratio": 0.60,
            "wrist_snap_velocity": "high",
            "key_traits": [
                "Quick release prioritized over raw power",
                "Exceptional shooting off one foot during agile moves",
                "Elite shot selection and deception",
            ],
        },
    },

    "alex_ovechkin": {
        "name": "Alex Ovechkin",
        "team": "Washington Capitals",
        "number": 8,
        "style": "one-timer / left circle / pure goal scoring",
        "color": "#C8102E",
        "skating": {
            "stride_cadence_hz": 3.9,
            "knee_bend_angle_deg": 108,
            "hip_extension_deg": 174,
            "forward_lean_deg": 16,
            "stride_length_ratio": 1.44,
            "ankle_plantarflexion_deg": 155,
            "shin_angle_deg": 20,               # shin forward tilt from vertical
            "glute_drive_symmetry": 0.93,
            "crossover_knee_inside_edge": True,
            "key_traits": [
                "Massive stride length driven by elite lower-body power and size",
                "Uses body mass as a weapon — accelerates through contact rather than around it",
                "Exceptional top-end speed for his size and playing style",
                "Dominant along the boards — wide base and powerful hip drive",
                "Positions himself using skating efficiency, not necessarily top speed",
            ],
            "coaching_cues": [
                "Use your size — drive through defenders, don't shy away from contact",
                "Long powerful strides — Ovechkin doesn't take short choppy steps",
                "Get to your spots: skating with purpose beats skating with speed",
                "Drive your outside edge hard on the wing to build speed before the circle",
            ],
        },
        "shooting": {
            "hip_rotation_deg": 100,
            "shoulder_tilt_deg": 22,
            "weight_transfer_pct": 95,
            "stick_flex_pct": 102,
            "release_point_body_ratio": 0.48,
            "wrist_snap_velocity": "elite",
            "key_traits": [
                "Greatest one-timer in NHL history — generates explosive power from the left circle",
                "Extreme shoulder tilt drops his top hand low, creating massive stick flex",
                "Full body weight committed through the shot — no holding back",
                "Releases the puck at the lowest point of his swing for maximum flex and velocity",
                "Shooting motion is a full-body rotation — legs, hips, core, and arms in sequence",
            ],
            "coaching_cues": [
                "Drop your top shoulder aggressively — Ovechkin's secret is extreme shoulder tilt",
                "Commit ALL your weight forward — plant your front foot hard and drive through",
                "Load early: get your stick back before the pass arrives so you can release instantly",
                "Practice one-timers from the left circle until the motion is completely automatic",
            ],
        },
    },

    "sidney_crosby": {
        "name": "Sidney Crosby",
        "team": "Pittsburgh Penguins",
        "number": 87,
        "style": "power / compete level / complete two-way game",
        "color": "#FCB514",
        "skating": {
            "stride_cadence_hz": 4.1,
            "knee_bend_angle_deg": 102,
            "hip_extension_deg": 176,
            "forward_lean_deg": 17,
            "stride_length_ratio": 1.42,
            "ankle_plantarflexion_deg": 157,
            "shin_angle_deg": 23,               # shin forward tilt from vertical
            "glute_drive_symmetry": 0.96,
            "crossover_knee_inside_edge": True,
            "key_traits": [
                "Exceptionally low center of gravity — impossible to knock off the puck",
                "Powerful quad and glute drive generates elite stride force",
                "Aggressive forward lean transfers maximum energy into horizontal propulsion",
                "Elite cornering — maintains speed through sharp turns better than almost any player",
                "Uses body width and hip strength to protect puck while accelerating",
            ],
            "coaching_cues": [
                "Get LOW — Crosby's hips are below most players' knees in his crouch",
                "Drive forward with your chest over your lead knee at all times",
                "Push through the full stride — no lazy recoveries",
                "Work tight-turn speed: Crosby is fastest when others slow down",
            ],
        },
        "shooting": {
            "hip_rotation_deg": 93,
            "shoulder_tilt_deg": 16,
            "weight_transfer_pct": 91,
            "stick_flex_pct": 97,
            "release_point_body_ratio": 0.54,
            "wrist_snap_velocity": "elite",
            "key_traits": [
                "Backhand as dangerous as forehand — defenses cannot cheat to one side",
                "Elite wrist shot accuracy to all four corners with consistent mechanics",
                "Generates shot power from an extremely low and wide stance",
                "Exceptional ability to shoot while absorbing contact or off-balance",
                "Deliberate shot placement — targets specific spots, rarely shoots to shoot",
            ],
            "coaching_cues": [
                "Train your backhand to the same standard as your forehand",
                "Stay low when you shoot — Crosby fires from his crouch, not standing up",
                "Pick a corner before you receive the puck — decide early, execute fast",
                "Practice shooting while being pushed or held — build stability through contact",
            ],
        },
    },

    "auston_matthews": {
        "name": "Auston Matthews",
        "team": "Toronto Maple Leafs",
        "number": 34,
        "style": "power shot / deceptive one-timer / elite scoring",
        "color": "#003E7E",
        "skating": {
            "stride_cadence_hz": 4.0,
            "knee_bend_angle_deg": 107,
            "hip_extension_deg": 173,
            "forward_lean_deg": 13,
            "stride_length_ratio": 1.38,
            "ankle_plantarflexion_deg": 154,
            "shin_angle_deg": 20,               # shin forward tilt from vertical
            "glute_drive_symmetry": 0.94,
            "crossover_knee_inside_edge": True,
            "key_traits": [
                "Powerful stride driven by elite lower-body strength",
                "Deceptive change-of-pace — can explode from a seemingly slow tempo",
                "Strong on-puck protection using body positioning and wide base",
                "Controlled aggression — physical without sacrificing skating efficiency",
                "Excellent backward-to-forward transition reads",
            ],
            "coaching_cues": [
                "Use your size — skate with a wide base and own your space",
                "Work explosive change-of-pace drills: slow-slow-BURST",
                "Drive through contact — don't avoid it, absorb it and keep moving",
                "Full hip extension on every stride — Matthews' power comes from his glutes",
            ],
        },
        "shooting": {
            "hip_rotation_deg": 95,
            "shoulder_tilt_deg": 20,
            "weight_transfer_pct": 93,
            "stick_flex_pct": 100,
            "release_point_body_ratio": 0.53,
            "wrist_snap_velocity": "elite",
            "key_traits": [
                "Two-position release — disguises shot as pass until the last instant",
                "Maximum stick flex utilization — loads the shaft for explosive energy release",
                "Elite hip-to-shoulder kinetic chain — one of the hardest wrist shots in the league",
                "Shoots with equal accuracy to both corners from any angle",
                "Rarely telegraphs shot direction through shoulder or eye movement",
            ],
            "coaching_cues": [
                "Load the shaft — push the blade into the ice hard before releasing",
                "Keep your top hand high and away from your body to maximize lever arm",
                "Drive your back hip through the shot — Matthews' power is hip-first",
                "Practice the two-position shot: set up to pass, then convert at the last moment",
            ],
        },
    },

    "connor_bedard": {
        "name": "Connor Bedard",
        "team": "Chicago Blackhawks",
        "number": 98,
        "style": "elite shot / deceptive release / playmaking",
        "color": "#CF0A2C",
        "skating": {
            "stride_cadence_hz": 4.1,
            "knee_bend_angle_deg": 106,
            "hip_extension_deg": 172,
            "forward_lean_deg": 14,
            "stride_length_ratio": 1.28,
            "ankle_plantarflexion_deg": 152,
            "shin_angle_deg": 19,               # shin forward tilt from vertical
            "glute_drive_symmetry": 0.95,
            "crossover_knee_inside_edge": True,
            "key_traits": [
                "Exceptional core stability maintains balance through tight contact",
                "Low center of gravity despite upright upper body",
                "Quick-twitch acceleration bursts in tight spaces",
                "Strong edge engagement on inside cuts",
                "Efficient stride mechanics conserve energy for shooting",
            ],
            "coaching_cues": [
                "Stay low through contact — Bedard absorbs checks without losing edges",
                "Work tight turns — train agility in small spaces",
                "Quick feet — short explosive strides in traffic",
                "Keep your head up through transitions",
            ],
        },
        "shooting": {
            "hip_rotation_deg": 88,
            "shoulder_tilt_deg": 18,
            "weight_transfer_pct": 86,
            "stick_flex_pct": 94,
            "release_point_body_ratio": 0.58,
            "wrist_snap_velocity": "elite",
            "key_traits": [
                "One of the fastest and most deceptive releases in the NHL",
                "Exceptional shoulder tilt generates extreme stick flex and unpredictable shot angles",
                "Shoots accurately off balance and in motion — doesn't need to set feet",
                "Elite wrist snap velocity — puck leaves stick with minimal visible wind-up",
            ],
            "coaching_cues": [
                "Tilt your top shoulder down aggressively through the shot — creates more flex and deception",
                "Practice one-touch releases — catch and shoot in a single motion",
                "Shoot off your back foot occasionally — Bedard generates power from any position",
                "Keep your hands away from your body for a longer power arc",
            ],
        },
    },

    "roope_hintz": {
        "name": "Roope Hintz",
        "team": "Dallas Stars",
        "number": 24,
        "style": "power stride / elite mechanics / two-way dominance",
        "color": "#006847",
        "skating": {
            "stride_cadence_hz": 4.1,
            "knee_bend_angle_deg": 104,           # deep, sustained knee bend through full stride
            "hip_extension_deg": 177,             # exceptional hip drive to near-full extension
            "forward_lean_deg": 16,               # aggressive forward lean from hips
            "stride_length_ratio": 1.43,          # long, powerful strides relative to leg length
            "ankle_plantarflexion_deg": 157,      # strong ankle extension / toe push
            "shin_angle_deg": 24,               # shin forward tilt from vertical
            "glute_drive_symmetry": 0.96,         # balanced bilateral power
            "crossover_knee_inside_edge": True,
            "key_traits": [
                "Elite hip drive generating exceptional push-off power",
                "Deep, consistent knee bend maintained through the full stride cycle",
                "Aggressive forward lean transfers maximum force into horizontal propulsion",
                "Complete ankle extension at toe-off — maximizes every stride's propulsive phase",
                "Smooth, efficient crossover transitions that preserve momentum",
                "Exceptional size-to-speed ratio — uses body mass without sacrificing agility",
            ],
            "coaching_cues": [
                "Drive your hips through — Hintz generates elite power by fully extending at the hip",
                "Stay low and lean forward aggressively — your shin should press into your boot tongue",
                "Complete every push with a full toe flick before heel recovery",
                "Keep your knee bend consistent — don't stand up between strides",
            ],
        },
        "shooting": {
            "hip_rotation_deg": 88,
            "shoulder_tilt_deg": 14,
            "weight_transfer_pct": 90,
            "stick_flex_pct": 95,
            "release_point_body_ratio": 0.53,
            "wrist_snap_velocity": "elite",
            "key_traits": [
                "Powerful wrist shot generated from elite lower-body stability",
                "Strong core rotation drives kinetic chain from skates to stick",
                "Excellent shooting mechanics in stride — doesn't need to set feet",
            ],
            "coaching_cues": [
                "Generate power from your legs up — Hintz uses his skating base for every shot",
                "Keep your core engaged through the shooting motion",
                "Drive your bottom hand through the puck on release",
                "Practice shooting in full stride — build the habit of not stopping to shoot",
            ],
        },
    },

    "cale_makar": {
        "name": "Cale Makar",
        "team": "Colorado Avalanche",
        "number": 8,
        "style": "explosive acceleration / lethal shot / 200-ft game",
        "color": "#236192",
        "skating": {
            "stride_cadence_hz": 4.4,
            "knee_bend_angle_deg": 103,
            "hip_extension_deg": 176,
            "forward_lean_deg": 16,
            "stride_length_ratio": 1.40,
            "ankle_plantarflexion_deg": 156,
            "shin_angle_deg": 21,               # shin forward tilt from vertical
            "glute_drive_symmetry": 0.96,
            "crossover_knee_inside_edge": True,
            "key_traits": [
                "Explosive first-step acceleration from standing start",
                "Combines McDavid-like speed with MacKinnon-like power",
                "Elite crossover technique in both offensive and defensive zone",
                "Efficient stride mechanics minimize energy expenditure",
                "Reads play two steps ahead — positioning minimizes need for top speed",
            ],
            "coaching_cues": [
                "Work your crossovers — Makar's crossovers generate as much speed as straight strides",
                "Train explosive first-step drills — 3-cone, edge bursts",
                "Stay disciplined in your positioning to use speed intelligently",
                "Hip drive on crossovers — don't just step, PUSH",
            ],
        },
        "shooting": {
            "hip_rotation_deg": 92,
            "shoulder_tilt_deg": 14,
            "weight_transfer_pct": 90,
            "stick_flex_pct": 98,
            "release_point_body_ratio": 0.50,
            "wrist_snap_velocity": "elite",
            "key_traits": [
                "Hardest shot among defensemen — elite stick flex utilization",
                "Exceptional ability to shoot off rushing stride",
                "Hip rotation generates power before arms engage",
            ],
        },
    },
}

# Metric weights for overall scoring
SKATING_METRIC_WEIGHTS = {
    "knee_bend":          0.22,
    "forward_lean":       0.18,
    "stride_length":      0.18,
    "hip_extension":      0.18,
    "symmetry":           0.14,
    "ankle_dorsiflexion": 0.10,
}

SHOOTING_METRIC_WEIGHTS = {
    "hip_rotation": 0.30,
    "weight_transfer": 0.25,
    "shoulder_alignment": 0.20,
    "release_point": 0.25,
}
