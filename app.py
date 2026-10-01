"""
Hockey Movement Analyzer — Flask backend.
Supports single video upload per analysis.
"""

import os
import re
import json
import uuid
import smtplib
import concurrent.futures
from datetime import datetime, timedelta
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from functools import wraps
from pathlib import Path
from flask import Flask, request, jsonify, render_template, send_from_directory, session, redirect, url_for
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from analyzer import analyze_video, calibrate_video, load_calibration_overrides, CALIBRATION_FILE
from golf_analyzer import analyze_golf_video

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")
except ImportError:
    pass

EMAIL_RE    = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')
ADMIN_EMAIL = "portoine@gmail.com"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 250 * 1024 * 1024  # 3 × 75 MB + headroom
app.secret_key = os.environ.get("SECRET_KEY", "hockey-analyzer-secret-2024")

SMTP_HOST    = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT    = int(os.environ.get("SMTP_PORT", 587))
SMTP_USER    = os.environ.get("SMTP_USER", "")
SMTP_PASS    = os.environ.get("SMTP_PASS", "")
APP_BASE_URL = os.environ.get("APP_BASE_URL", "http://127.0.0.1:5050").rstrip("/")

UPLOAD_DIR    = Path("uploads")
PROCESSED_DIR = Path("processed")
USERS_FILE    = Path("users.json")
UPLOAD_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

ALLOWED_EXT       = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_VIDEO_SIZE_MB  = 75
MAX_VIDEOS         = 1
MIN_PASSWORD_LEN   = 6

# In-memory reset tokens: {token: {"email": str, "expires": datetime}}
_reset_tokens: dict = {}


# ── User store (JSON file) ────────────────────────────────────────────────────

def load_users() -> dict:
    if USERS_FILE.exists():
        users = json.loads(USERS_FILE.read_text())
        # Migrate: existing accounts without status get "approved"
        changed = False
        for email, u in users.items():
            if "status" not in u:
                u["status"] = "approved"
                changed = True
        if changed:
            USERS_FILE.write_text(json.dumps(users, indent=2))
        return users
    return {}


def save_users(users: dict):
    USERS_FILE.write_text(json.dumps(users, indent=2))


def get_user(username: str) -> dict | None:
    return load_users().get(username.lower())


def create_user(username: str, password: str, name: str = "", phone: str = ""):
    users = load_users()
    email = username.lower()
    status = "approved" if email == ADMIN_EMAIL else "pending"
    users[email] = {
        "display": name.strip() if name.strip() else email.split("@")[0],
        "name":    name.strip(),
        "phone":   phone.strip(),
        "email":   email,
        "password_hash": generate_password_hash(password),
        "status":  status,
        "created": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_users(users)
    return status


def verify_user(username: str, password: str) -> bool:
    user = get_user(username)
    if not user:
        return False
    return check_password_hash(user["password_hash"], password)


# ── Email helpers ─────────────────────────────────────────────────────────────

def send_reset_email(to_email: str, reset_url: str) -> bool:
    """Send a password-reset email. Returns True on success, False on failure."""
    if not SMTP_USER or not SMTP_PASS or SMTP_PASS == "your-app-password-here":
        return False

    subject = "Hockey Analyzer — Reset your password"

    html_body = f"""\
<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
</head>
<body style="margin:0;padding:0;background:#0a1628;font-family:'Segoe UI',system-ui,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0a1628;padding:40px 20px;">
    <tr><td align="center">
      <table width="520" cellpadding="0" cellspacing="0"
             style="background:#111e2e;border:1px solid #1e3048;border-radius:16px;overflow:hidden;max-width:520px;">

        <!-- Header -->
        <tr>
          <td style="background:linear-gradient(135deg,#060f1c,#082030);
                     border-bottom:2px solid #00c8ff;
                     padding:24px 32px;text-align:center;">
            <p style="margin:0;font-size:22px;font-weight:800;color:#ffffff;letter-spacing:-0.3px;">
              🏒 Hockey Movement Analyzer
            </p>
            <p style="margin:4px 0 0;font-size:13px;color:#8899aa;">
              AI-powered biomechanical analysis
            </p>
          </td>
        </tr>

        <!-- Body -->
        <tr>
          <td style="padding:36px 32px;">
            <p style="margin:0 0 8px;font-size:22px;font-weight:700;color:#e8f4fd;">
              Reset your password
            </p>
            <p style="margin:0 0 28px;font-size:14px;color:#8899aa;line-height:1.6;">
              We received a request to reset the password for
              <strong style="color:#e8f4fd;">{to_email}</strong>.
              Click the button below — this link expires in <strong style="color:#00c8ff;">30 minutes</strong>.
            </p>

            <!-- CTA button -->
            <table cellpadding="0" cellspacing="0" width="100%">
              <tr><td align="center" style="padding:4px 0 32px;">
                <a href="{reset_url}"
                   style="display:inline-block;background:#00c8ff;color:#0a1628;
                          font-size:15px;font-weight:800;letter-spacing:0.03em;
                          text-decoration:none;padding:14px 36px;border-radius:10px;">
                  Set New Password &rarr;
                </a>
              </td></tr>
            </table>

            <p style="margin:0 0 8px;font-size:12px;color:#8899aa;line-height:1.6;">
              If the button doesn't work, copy and paste this link into your browser:
            </p>
            <p style="margin:0 0 28px;font-size:11px;color:#00c8ff;word-break:break-all;">
              {reset_url}
            </p>

            <hr style="border:none;border-top:1px solid #1e3048;margin:0 0 24px;"/>

            <p style="margin:0;font-size:12px;color:#3a5070;line-height:1.6;">
              If you didn't request a password reset you can safely ignore this email —
              your password will not be changed.
            </p>
          </td>
        </tr>

        <!-- Footer -->
        <tr>
          <td style="background:#0d1a2a;border-top:1px solid #1e3048;
                     padding:16px 32px;text-align:center;">
            <p style="margin:0;font-size:11px;color:#2a4060;">
              Hockey Movement Analyzer &mdash; All video processing happens locally on your machine.
            </p>
          </td>
        </tr>

      </table>
    </td></tr>
  </table>
</body>
</html>"""

    plain_body = (
        f"Hockey Movement Analyzer — Reset your password\n\n"
        f"Click the link below to set a new password (expires in 30 minutes):\n\n"
        f"{reset_url}\n\n"
        f"If you didn't request this, you can safely ignore this email."
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"Hockey Analyzer <{SMTP_USER}>"
    msg["To"]      = to_email
    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body,  "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as srv:
            srv.ehlo()
            srv.starttls()
            srv.login(SMTP_USER, SMTP_PASS)
            srv.sendmail(SMTP_USER, to_email, msg.as_string())
        return True
    except Exception as e:
        app.logger.error("Failed to send reset email: %s", e)
        return False


def send_registration_notification(name: str, email: str) -> bool:
    """Email the admin when a new user registers."""
    if not SMTP_USER or not SMTP_PASS or SMTP_PASS == "your-app-password-here":
        return False

    admin_url = f"{APP_BASE_URL}/admin"
    subject   = f"New Registration Request — {name or email}"

    html_body = f"""\
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"/></head>
<body style="margin:0;padding:0;background:#0a1628;font-family:'Segoe UI',system-ui,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0a1628;padding:40px 20px;">
    <tr><td align="center">
      <table width="520" cellpadding="0" cellspacing="0"
             style="background:#111e2e;border:1px solid #1e3048;border-radius:16px;overflow:hidden;max-width:520px;">
        <tr>
          <td style="background:linear-gradient(135deg,#060f1c,#082030);border-bottom:2px solid #00c8ff;padding:24px 32px;text-align:center;">
            <p style="margin:0;font-size:22px;font-weight:800;color:#fff;">🏒 Hockey Movement Analyzer</p>
            <p style="margin:4px 0 0;font-size:13px;color:#8899aa;">New account request</p>
          </td>
        </tr>
        <tr>
          <td style="padding:36px 32px;">
            <p style="margin:0 0 8px;font-size:20px;font-weight:700;color:#e8f4fd;">New Registration</p>
            <p style="margin:0 0 24px;font-size:14px;color:#8899aa;line-height:1.6;">
              A new user is requesting access to the Hockey Movement Analyzer.
            </p>
            <table style="width:100%;border-collapse:collapse;margin-bottom:28px;">
              <tr><td style="padding:8px 0;font-size:13px;color:#8899aa;width:100px;">Name</td>
                  <td style="padding:8px 0;font-size:13px;color:#e8f4fd;font-weight:600;">{name or '—'}</td></tr>
              <tr><td style="padding:8px 0;font-size:13px;color:#8899aa;">Email</td>
                  <td style="padding:8px 0;font-size:13px;color:#00c8ff;">{email}</td></tr>
            </table>
            <table cellpadding="0" cellspacing="0" width="100%">
              <tr><td align="center" style="padding:4px 0 8px;">
                <a href="{admin_url}"
                   style="display:inline-block;background:#00c8ff;color:#0a1628;
                          font-size:15px;font-weight:800;text-decoration:none;padding:14px 36px;border-radius:10px;">
                  Review in Admin Panel &rarr;
                </a>
              </td></tr>
            </table>
          </td>
        </tr>
        <tr>
          <td style="background:#0d1a2a;border-top:1px solid #1e3048;padding:16px 32px;text-align:center;">
            <p style="margin:0;font-size:11px;color:#2a4060;">Hockey Movement Analyzer — Admin notification</p>
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body>
</html>"""

    plain_body = (
        f"New registration request\n\n"
        f"Name:  {name or '—'}\n"
        f"Email: {email}\n\n"
        f"Review and approve at: {admin_url}"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"Hockey Analyzer <{SMTP_USER}>"
    msg["To"]      = ADMIN_EMAIL
    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body,  "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as srv:
            srv.ehlo(); srv.starttls(); srv.login(SMTP_USER, SMTP_PASS)
            srv.sendmail(SMTP_USER, ADMIN_EMAIL, msg.as_string())
        return True
    except Exception as e:
        app.logger.error("Failed to send registration notification: %s", e)
        return False


# ── Session store ─────────────────────────────────────────────────────────────

def compute_session_summary(results: list, mode: str) -> dict:
    """Aggregate scores across all successful videos for one analysis session."""
    valid = [r for r in results
             if not r.get("error") and r.get("total_frames_analyzed", 0) > 0]
    if not valid:
        return {}

    pro_score_lists: dict = {}
    all_breakdowns: list  = []

    for r in valid:
        comps = r.get("comparisons") or {}
        for pro_key, comp in comps.items():
            pro_score_lists.setdefault(pro_key, []).append(comp.get("overall_score", 0))
        best = r.get("best_match")
        if best and comps.get(best):
            bd = comps[best].get("score_breakdown", {})
            if bd:
                all_breakdowns.append(bd)

    if not pro_score_lists:
        return {}

    avg_scores  = {k: sum(v) / len(v) for k, v in pro_score_lists.items()}
    best_match  = max(avg_scores, key=avg_scores.get)
    avg_score   = round(avg_scores[best_match])

    avg_breakdown: dict = {}
    if all_breakdowns:
        for k in all_breakdowns[0]:
            vals = [b[k] for b in all_breakdowns if k in b]
            avg_breakdown[k] = round(sum(vals) / len(vals)) if vals else 0

    return {
        "avg_score":    avg_score,
        "best_match":   best_match,
        "avg_breakdown": avg_breakdown,
        "scores_by_pro": {k: round(v) for k, v in avg_scores.items()},
    }


def save_session(email: str, record: dict):
    users = load_users()
    if email not in users:
        return
    users[email].setdefault("sessions", []).append(record)
    save_users(users)


def get_sessions(email: str) -> list:
    user = get_user(email)
    if not user:
        return []
    return sorted(user.get("sessions", []), key=lambda s: s["timestamp"], reverse=True)


def get_archived_sessions(email: str) -> list:
    user = get_user(email)
    if not user:
        return []
    return sorted(user.get("archived_sessions", []), key=lambda s: s["timestamp"], reverse=True)


def archive_all_sessions(email: str) -> int:
    """Move all active sessions to archived_sessions. Returns count archived."""
    users = load_users()
    user  = users.get(email)
    if not user:
        return 0
    active   = user.get("sessions", [])
    archived = user.get("archived_sessions", [])
    if not active:
        return 0
    archived.extend(active)
    user["archived_sessions"] = archived
    user["sessions"]          = []
    save_users(users)
    return len(active)


def restore_session(email: str, session_id: str) -> bool:
    """Move a single archived session back to active sessions."""
    users = load_users()
    user  = users.get(email)
    if not user:
        return False
    archived = user.get("archived_sessions", [])
    target   = next((s for s in archived if s["id"] == session_id), None)
    if not target:
        return False
    user["archived_sessions"] = [s for s in archived if s["id"] != session_id]
    user.setdefault("sessions", []).append(target)
    save_users(users)
    return True


# ── Auth helpers ──────────────────────────────────────────────────────────────

def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in") or session.get("email") != ADMIN_EMAIL:
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# ── File helpers ──────────────────────────────────────────────────────────────

def allowed_file(filename):
    return Path(filename).suffix.lower() in ALLOWED_EXT


def process_one(file_obj, label, mode):
    uid  = uuid.uuid4().hex[:8]
    ext  = Path(secure_filename(file_obj.filename)).suffix.lower()
    path = UPLOAD_DIR / f"{uid}{ext}"
    file_obj.save(path)
    try:
        result = analyze_video(str(path), mode=mode, output_dir=str(PROCESSED_DIR))
        result["label"] = label
        result["filename"] = file_obj.filename
    except Exception as e:
        result = {"error": str(e), "label": label, "filename": file_obj.filename}
    finally:
        path.unlink(missing_ok=True)
    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("landing"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if verify_user(username, password):
            user = get_user(username)
            if user.get("status", "approved") == "pending":
                error = "Your account is awaiting admin approval. You'll receive an email once it's approved."
            else:
                session["logged_in"] = True
                session["username"]  = user["display"]
                session["email"]     = username.lower()
                return redirect(url_for("landing"))
        else:
            error = "Invalid username or password."
    return render_template("login.html", error=error)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("logged_in"):
        return redirect(url_for("landing"))
    error   = None
    pending = False
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm", "")
        name     = request.form.get("name", "").strip()
        phone    = request.form.get("phone", "").strip()

        if not name:
            error = "Please enter your full name."
        elif not username or not EMAIL_RE.match(username):
            error = "Please enter a valid email address."
        elif len(password) < MIN_PASSWORD_LEN:
            error = f"Password must be at least {MIN_PASSWORD_LEN} characters."
        elif password != confirm:
            error = "Passwords do not match."
        elif get_user(username):
            error = "An account with that email already exists."
        else:
            status = create_user(username, password, name=name, phone=phone)
            if status == "approved":
                session["logged_in"] = True
                session["username"]  = name or username.split("@")[0]
                session["email"]     = username.lower()
                return redirect(url_for("landing"))
            else:
                # Notify admin
                send_registration_notification(name, username.lower())
                pending = True

    return render_template("signup.html", error=error, pending=pending)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))


@app.route("/")
@login_required
def landing():
    email = session.get("email", "")
    return render_template("landing.html",
                           username=session.get("username", ""),
                           is_admin=(email == ADMIN_EMAIL))


@app.route("/hockey")
@login_required
def hockey():
    from flask import make_response
    resp = make_response(render_template("index.html", username=session.get("username", "")))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/processed/<filename>")
@login_required
def processed_file(filename):
    return send_from_directory(PROCESSED_DIR, filename)


@app.route("/analyze", methods=["POST"])
@login_required
def analyze():
    mode  = request.form.get("mode", "skating")
    files = request.files.getlist("videos")

    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No video files uploaded"}), 400

    files = [f for f in files if f.filename != ""]

    if len(files) > MAX_VIDEOS:
        return jsonify({"error": f"Maximum {MAX_VIDEOS} videos per submission."}), 400

    for f in files:
        if not allowed_file(f.filename):
            return jsonify({"error": f"'{f.filename}' is not a supported format. Use MP4, MOV, AVI, MKV, or WebM."}), 400
        f.seek(0, 2)
        size_mb = f.tell() / (1024 * 1024)
        f.seek(0)
        if size_mb > MAX_VIDEO_SIZE_MB:
            return jsonify({"error": f"'{f.filename}' is {size_mb:.0f} MB — maximum is {MAX_VIDEO_SIZE_MB} MB per video."}), 400

    labels = [f"Video {i+1}" for i in range(len(files))]
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_VIDEOS) as pool:
        futures = [pool.submit(process_one, f, lbl, mode) for f, lbl in zip(files, labels)]
        results = [fut.result() for fut in futures]

    # Auto-save session for the logged-in user
    email = session.get("email", "")
    if email:
        summary = compute_session_summary(results, mode)
        if summary:
            valid_count = len([r for r in results
                                if not r.get("error") and r.get("total_frames_analyzed", 0) > 0])
            record = {
                "id":          uuid.uuid4().hex[:12],
                "timestamp":   datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                "mode":        mode,
                "video_count": valid_count,
            }
            record.update(summary)
            save_session(email, record)

    return jsonify({"results": results, "mode": mode, "count": len(results)})


@app.route("/api/sessions")
@login_required
def api_sessions():
    email = session.get("email", "")
    data  = get_sessions(email) if email else []
    return jsonify({"sessions": data})


@app.route("/api/sessions/<session_id>", methods=["DELETE"])
@login_required
def delete_session_route(session_id):
    email = session.get("email", "")
    if not email:
        return jsonify({"error": "Not logged in"}), 401
    users = load_users()
    user  = users.get(email)
    if not user:
        return jsonify({"error": "User not found"}), 404
    user["sessions"] = [s for s in user.get("sessions", []) if s["id"] != session_id]
    save_users(users)
    return jsonify({"ok": True})


@app.route("/api/sessions/archive-all", methods=["POST"])
@login_required
def archive_all_sessions_route():
    email = session.get("email", "")
    if not email:
        return jsonify({"error": "Not logged in"}), 401
    count = archive_all_sessions(email)
    return jsonify({"ok": True, "archived": count})


@app.route("/api/sessions/archived")
@login_required
def get_archived_sessions_route():
    email = session.get("email", "")
    data  = get_archived_sessions(email) if email else []
    return jsonify({"sessions": data})


@app.route("/api/sessions/<session_id>/restore", methods=["POST"])
@login_required
def restore_session_route(session_id):
    email = session.get("email", "")
    if not email:
        return jsonify({"error": "Not logged in"}), 401
    ok = restore_session(email, session_id)
    if not ok:
        return jsonify({"error": "Session not found in archive"}), 404
    return jsonify({"ok": True})


@app.route("/api/sessions/archived/<session_id>", methods=["DELETE"])
@login_required
def delete_archived_session_route(session_id):
    email = session.get("email", "")
    if not email:
        return jsonify({"error": "Not logged in"}), 401
    users = load_users()
    user  = users.get(email)
    if not user:
        return jsonify({"error": "User not found"}), 404
    user["archived_sessions"] = [
        s for s in user.get("archived_sessions", []) if s["id"] != session_id
    ]
    save_users(users)
    return jsonify({"ok": True})


# ── Calibration helpers ───────────────────────────────────────────────────────

def save_calibration_data(data: dict):
    CALIBRATION_FILE.write_text(json.dumps(data, indent=2))


# ── Calibration routes ────────────────────────────────────────────────────────

@app.route("/calibrate")
@login_required
def calibrate_page():
    from pro_profiles import PRO_PROFILES
    cal = load_calibration_overrides()
    status = {
        key: {
            "name":               p["name"],
            "color":              p["color"],
            "style":              p["style"],
            "skating_calibrated": bool(cal.get(key, {}).get("skating")),
            "shooting_calibrated": bool(cal.get(key, {}).get("shooting")),
        }
        for key, p in PRO_PROFILES.items()
    }
    return render_template("calibrate.html", status=status)


@app.route("/calibrate/run", methods=["POST"])
@login_required
def calibrate_run():
    from pro_profiles import PRO_PROFILES
    from analyzer import CALIBRATION_FIELDS

    pro_key = request.form.get("pro_key", "").strip()
    mode    = request.form.get("mode", "skating").strip()

    if pro_key not in PRO_PROFILES:
        return jsonify({"error": "Unknown pro player."}), 400
    if mode not in ("skating", "shooting"):
        return jsonify({"error": "Mode must be 'skating' or 'shooting'."}), 400

    f = request.files.get("video")
    if not f or f.filename == "":
        return jsonify({"error": "No video file provided."}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Unsupported format. Use MP4, MOV, AVI, MKV, or WebM."}), 400

    f.seek(0, 2); size_mb = f.tell() / (1024 * 1024); f.seek(0)
    if size_mb > MAX_VIDEO_SIZE_MB:
        return jsonify({"error": f"File is {size_mb:.0f} MB — max is {MAX_VIDEO_SIZE_MB} MB."}), 400

    uid  = uuid.uuid4().hex[:8]
    ext  = Path(secure_filename(f.filename)).suffix.lower()
    path = UPLOAD_DIR / f"{uid}{ext}"
    f.save(path)

    try:
        result = calibrate_video(str(path), mode)
    except Exception as e:
        result = {"error": str(e)}
    finally:
        path.unlink(missing_ok=True)

    if "error" in result:
        return jsonify(result), 400

    profile  = PRO_PROFILES[pro_key]
    mode_ref = profile["skating"] if mode == "skating" else profile["shooting"]
    cal      = load_calibration_overrides()
    saved    = cal.get(pro_key, {}).get(mode, {})

    result["pro_key"]          = pro_key
    result["pro_name"]         = profile["name"]
    result["hardcoded"]        = {k: mode_ref.get(k) for k in CALIBRATION_FIELDS[mode]}
    result["previously_saved"] = saved
    result["field_labels"]     = CALIBRATION_FIELDS[mode]

    return jsonify(result)


@app.route("/calibrate/save", methods=["POST"])
@login_required
def calibrate_save():
    from pro_profiles import PRO_PROFILES
    body    = request.get_json(force=True)
    pro_key = body.get("pro_key", "")
    mode    = body.get("mode", "")
    values  = body.get("values", {})

    if pro_key not in PRO_PROFILES:
        return jsonify({"error": "Unknown pro player."}), 400
    if mode not in ("skating", "shooting"):
        return jsonify({"error": "Invalid mode."}), 400

    # Coerce all values to float
    clean = {}
    for k, v in values.items():
        try:
            clean[k] = round(float(v), 4)
        except (TypeError, ValueError):
            pass

    cal = load_calibration_overrides()
    cal.setdefault(pro_key, {})[mode] = clean
    save_calibration_data(cal)

    return jsonify({"ok": True, "saved": len(clean), "pro": PRO_PROFILES[pro_key]["name"]})


@app.route("/calibrate/reset", methods=["POST"])
@login_required
def calibrate_reset():
    from pro_profiles import PRO_PROFILES
    body    = request.get_json(force=True)
    pro_key = body.get("pro_key", "")
    mode    = body.get("mode", "")

    cal = load_calibration_overrides()
    if pro_key in cal and mode in cal[pro_key]:
        del cal[pro_key][mode]
        if not cal[pro_key]:
            del cal[pro_key]
        save_calibration_data(cal)

    return jsonify({"ok": True})


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if session.get("logged_in"):
        return redirect(url_for("landing"))
    message = None
    error   = None
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        if not email or not EMAIL_RE.match(email):
            error = "Please enter a valid email address."
        else:
            if get_user(email):
                token = uuid.uuid4().hex
                _reset_tokens[token] = {
                    "email":   email,
                    "expires": datetime.utcnow() + timedelta(minutes=30),
                }
                reset_url = f"{APP_BASE_URL}/reset-password/{token}"
                sent = send_reset_email(email, reset_url)
                if not sent:
                    # SMTP not configured — fall back to direct redirect (dev mode)
                    return redirect(url_for("reset_password", token=token))
            # Always show the same message so we don't leak which emails exist
            message = "If that email is registered you'll receive a reset link shortly. Check your inbox (and spam folder)."
    return render_template("forgot_password.html", error=error, message=message)


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    if session.get("logged_in"):
        return redirect(url_for("landing"))

    entry = _reset_tokens.get(token)
    if not entry or datetime.utcnow() > entry["expires"]:
        return render_template("forgot_password.html",
                               error="This reset link is invalid or has expired. Please try again.",
                               message=None)

    error = None
    if request.method == "POST":
        password = request.form.get("password", "")
        confirm  = request.form.get("confirm", "")
        if len(password) < MIN_PASSWORD_LEN:
            error = f"Password must be at least {MIN_PASSWORD_LEN} characters."
        elif password != confirm:
            error = "Passwords do not match."
        else:
            users = load_users()
            email = entry["email"]
            if email in users:
                users[email]["password_hash"] = generate_password_hash(password)
                save_users(users)
            del _reset_tokens[token]
            return render_template("forgot_password.html",
                                   message="Password updated successfully. You can now sign in.",
                                   error=None)

    return render_template("reset_password.html", token=token, error=error,
                           email=entry["email"])


@app.route("/golf")
@login_required
def golf():
    from flask import make_response
    resp = make_response(render_template("golf.html", username=session.get("username", "")))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    resp.headers["Expires"] = "0"
    return resp


@app.route("/analyze_golf", methods=["POST"])
@login_required
def analyze_golf():
    shot_type = request.form.get("shot_type", "auto").strip()
    f = request.files.get("video")

    if not f or f.filename == "":
        return jsonify({"error": "No video file uploaded"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": f"'{f.filename}' is not a supported format. Use MP4, MOV, AVI, MKV, or WebM."}), 400

    f.seek(0, 2)
    size_mb = f.tell() / (1024 * 1024)
    f.seek(0)
    if size_mb > MAX_VIDEO_SIZE_MB:
        return jsonify({"error": f"'{f.filename}' is {size_mb:.0f} MB — maximum is {MAX_VIDEO_SIZE_MB} MB per video."}), 400

    uid  = uuid.uuid4().hex[:8]
    ext  = Path(secure_filename(f.filename)).suffix.lower()
    path = UPLOAD_DIR / f"{uid}{ext}"
    f.save(path)

    try:
        result = analyze_golf_video(str(path), shot_type=shot_type, output_dir=str(PROCESSED_DIR))
        result["filename"] = f.filename
    except Exception as e:
        result = {"error": str(e), "filename": f.filename}
    finally:
        path.unlink(missing_ok=True)

    # Save golf session
    email = session.get("email", "")
    if email and not result.get("error"):
        comps = result.get("comparisons") or {}
        best  = result.get("best_match")
        if best and comps.get(best):
            record = {
                "id":          uuid.uuid4().hex[:12],
                "timestamp":   datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S"),
                "mode":        f"golf_{result.get('shot_type', 'drive')}",
                "video_count": 1,
                "avg_score":   comps[best].get("overall_score", 0),
                "best_match":  best,
                "avg_breakdown": comps[best].get("score_breakdown", {}),
                "scores_by_pro": {k: v.get("overall_score", 0) for k, v in comps.items()},
            }
            save_session(email, record)

    return jsonify(result)


# ── Admin routes ──────────────────────────────────────────────────────────────

@app.route("/admin")
@admin_required
def admin_panel():
    users = load_users()
    pending  = []
    approved = []
    for email, u in users.items():
        row = {
            "email":   email,
            "name":    u.get("name") or u.get("display", ""),
            "phone":   u.get("phone", ""),
            "display": u.get("display", ""),
            "status":  u.get("status", "approved"),
            "created": u.get("created", ""),
            "sessions": len(u.get("sessions", [])),
        }
        if u.get("status", "approved") == "pending":
            pending.append(row)
        else:
            approved.append(row)
    # Sort approved by name
    approved.sort(key=lambda r: (r["name"] or r["email"]).lower())
    pending.sort(key=lambda r: r["created"])
    return render_template("admin.html",
                           username=session.get("username", ""),
                           pending=pending,
                           approved=approved)


@app.route("/api/admin/approve/<path:email>", methods=["POST"])
@admin_required
def admin_approve(email):
    email = email.lower()
    users = load_users()
    if email not in users:
        return jsonify({"error": "User not found"}), 404
    users[email]["status"] = "approved"
    save_users(users)
    # Optionally notify the user
    _send_approval_email(email, users[email].get("name", ""))
    return jsonify({"ok": True})


@app.route("/api/admin/reject/<path:email>", methods=["POST"])
@admin_required
def admin_reject(email):
    email = email.lower()
    users = load_users()
    if email not in users:
        return jsonify({"error": "User not found"}), 404
    del users[email]
    save_users(users)
    return jsonify({"ok": True})


def _send_approval_email(to_email: str, name: str) -> bool:
    """Notify a user that their account has been approved."""
    if not SMTP_USER or not SMTP_PASS or SMTP_PASS == "your-app-password-here":
        return False
    subject = "Your Hockey Analyzer account has been approved!"
    login_url = f"{APP_BASE_URL}/login"
    html_body = f"""\
<!DOCTYPE html><html><head><meta charset="UTF-8"/></head>
<body style="margin:0;padding:0;background:#0a1628;font-family:'Segoe UI',system-ui,sans-serif;">
  <table width="100%" cellpadding="0" cellspacing="0" style="background:#0a1628;padding:40px 20px;">
    <tr><td align="center">
      <table width="520" cellpadding="0" cellspacing="0"
             style="background:#111e2e;border:1px solid #1e3048;border-radius:16px;overflow:hidden;max-width:520px;">
        <tr>
          <td style="background:linear-gradient(135deg,#060f1c,#082030);border-bottom:2px solid #27ae60;padding:24px 32px;text-align:center;">
            <p style="margin:0;font-size:22px;font-weight:800;color:#fff;">🏒 Hockey Movement Analyzer</p>
          </td>
        </tr>
        <tr>
          <td style="padding:36px 32px;">
            <p style="margin:0 0 16px;font-size:20px;font-weight:700;color:#e8f4fd;">You're approved, {name or 'Skater'}!</p>
            <p style="margin:0 0 28px;font-size:14px;color:#8899aa;line-height:1.6;">
              Your account has been approved. You can now sign in and start analyzing your skating.
            </p>
            <table cellpadding="0" cellspacing="0" width="100%">
              <tr><td align="center">
                <a href="{login_url}"
                   style="display:inline-block;background:#27ae60;color:#fff;
                          font-size:15px;font-weight:800;text-decoration:none;padding:14px 36px;border-radius:10px;">
                  Sign In Now &rarr;
                </a>
              </td></tr>
            </table>
          </td>
        </tr>
      </table>
    </td></tr>
  </table>
</body></html>"""
    plain_body = f"Hi {name or 'Skater'},\n\nYour Hockey Analyzer account has been approved. Sign in at: {login_url}"
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"Hockey Analyzer <{SMTP_USER}>"
    msg["To"]      = to_email
    msg.attach(MIMEText(plain_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))
    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as srv:
            srv.ehlo(); srv.starttls(); srv.login(SMTP_USER, SMTP_PASS)
            srv.sendmail(SMTP_USER, to_email, msg.as_string())
        return True
    except Exception as e:
        app.logger.error("Failed to send approval email: %s", e)
        return False


if __name__ == "__main__":
    app.run(debug=True, port=5050)
