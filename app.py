"""Smart Nutrition & Diet Planner - Flask application.   Run:  python app.py"""
import os
import re
import json
import secrets
from datetime import date, timedelta, datetime
from functools import wraps

from flask import (Flask, render_template, request, redirect, url_for, session, flash, jsonify, send_file, g, abort)
from werkzeug.security import check_password_hash

from utils import database as db
from utils import calorie_calculator as cc
from utils import diet_recommender as dr
from utils import ml_predictor
from utils.pdf_report import build_report

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)


def _secret_key():
    if os.environ.get("SECRET_KEY"):
        return os.environ["SECRET_KEY"]
    path = os.path.join(BASE_DIR, ".secret_key")
    if not os.path.exists(path):
        with open(path, "w") as fh:
            fh.write(secrets.token_hex(32))
    return open(path).read().strip()


app.config.update(SECRET_KEY=_secret_key(), SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
                  PERMANENT_SESSION_LIFETIME=timedelta(hours=8), MAX_CONTENT_LENGTH=1024 * 1024)
db.init_db()
ml_predictor.ensure_model()
FOODS = {f["food_name"]: f for f in dr.load_foods()}


# --------------------------------------------------------------------------- security helpers
def csrf_token():
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_hex(16)
    return session["_csrf"]


app.jinja_env.globals.update(csrf_token=csrf_token, ACTIVITY=cc.ACTIVITY, GOALS=cc.GOALS, CHALLENGES=db.CHALLENGES)


@app.before_request
def load_user_and_check_csrf():
    g.user = db.get_user(session["user_id"]) if session.get("user_id") else None
    if request.method == "POST":
        sent = request.form.get("csrf_token") or request.headers.get("X-CSRFToken")
        if not sent or sent != session.get("_csrf"):
            if request.is_json or request.path.startswith("/api/"):
                return jsonify(error="Session expired. Refresh the page and try again."), 400
            flash("Your session expired. Please try again.", "warning")
            return redirect(request.referrer or url_for("login"))


@app.after_request
def headers(resp):
    resp.headers["X-Content-Type-Options"] = "nosniff"
    resp.headers["X-Frame-Options"] = "SAMEORIGIN"
    return resp


def login_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        if not g.user:
            flash("Please log in to continue.", "info")
            return redirect(url_for("login", next=request.path))
        return fn(*a, **kw)
    return wrapper


def profile_required(fn):
    @wraps(fn)
    def wrapper(*a, **kw):
        g.profile = db.get_profile(g.user["id"])
        if not g.profile:
            flash("Complete your nutrition assessment first.", "info")
            return redirect(url_for("profile"))
        return fn(*a, **kw)
    return wrapper


def validate_password(pw):
    errs = []
    if len(pw) < 8: errs.append("at least 8 characters")
    if not re.search(r"[A-Z]", pw): errs.append("an uppercase letter")
    if not re.search(r"[a-z]", pw): errs.append("a lowercase letter")
    if not re.search(r"\d", pw): errs.append("a number")
    if not re.search(r"[^A-Za-z0-9]", pw): errs.append("a special character")
    return errs


def num(form, key, lo, hi, label, cast=float):
    try:
        v = cast(form.get(key, "").strip())
    except ValueError:
        raise ValueError(f"{label} must be a number.")
    if not lo <= v <= hi:
        raise ValueError(f"{label} must be between {lo} and {hi}.")
    return v


# --------------------------------------------------------------------------- auth
@app.route("/")
def index():
    return redirect(url_for("dashboard" if g.user else "login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if g.user:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        f = request.form
        username, email, pw = f.get("username", "").strip(), f.get("email", "").strip().lower(), f.get("password", "")
        errors = []
        if not re.fullmatch(r"[A-Za-z0-9_]{3,30}", username):
            errors.append("Username must be 3-30 letters, numbers or underscores.")
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", email):
            errors.append("Enter a valid email address.")
        miss = validate_password(pw)
        if miss:
            errors.append("Password needs " + ", ".join(miss) + ".")
        if pw != f.get("confirm", ""):
            errors.append("Passwords do not match.")
        try:
            age = num(f, "age", 14, 90, "Age", int)
        except ValueError as e:
            errors.append(str(e)); age = None
        gender = f.get("gender")
        if gender not in ("Male", "Female", "Other"):
            errors.append("Select a gender.")
        if not errors and db.user_exists(username, email):
            errors.append("That username or email is already registered.")
        if errors:
            for e in errors: flash(e, "danger")
            return render_template("register.html", form=f), 400
        uid = db.create_user(username, email, pw, age, gender)
        session.clear(); session.permanent = True; session["user_id"] = uid
        flash("Welcome! Tell us about yourself to get your personalised plan.", "success")
        return redirect(url_for("profile"))
    return render_template("register.html", form={})


@app.route("/login", methods=["GET", "POST"])
def login():
    if g.user:
        return redirect(url_for("dashboard"))
    if request.method == "POST":
        user = db.find_user(request.form.get("identifier", "").strip())
        if user and check_password_hash(user["password_hash"], request.form.get("password", "")):
            nxt = request.args.get("next", "")
            session.clear(); session.permanent = True; session["user_id"] = user["id"]
            return redirect(nxt if nxt.startswith("/") and not nxt.startswith("//") else url_for("dashboard"))
        flash("Incorrect username/email or password.", "danger")
        return render_template("login.html"), 401
    return render_template("login.html")


@app.route("/logout", methods=["POST", "GET"])
def logout():
    session.clear()
    flash("You have been logged out.", "info")
    return redirect(url_for("login"))


# --------------------------------------------------------------------------- core computation
def compute(profile):
    """Rule-based assessment and ML prediction are computed separately."""
    return cc.assess(profile), ml_predictor.predict(profile)


def new_plan(uid, profile, assess, ml):
    plan = dr.generate_plan(profile, assess, ml["focus"], seed=secrets.randbits(32))
    plan["ml"] = {"focus_label": ml["focus_label"], "focus_text": ml["focus_text"], "range": [ml["calories_low"], ml["calories_high"]]}
    return db.save_plan(uid, plan, assess["target_calories"], profile["goal"], profile["food_pref"], ml["focus"])


# --------------------------------------------------------------------------- profile
@app.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    uid = g.user["id"]
    existing = db.get_profile(uid)
    if request.method == "POST":
        f = request.form
        try:
            data = {
                "full_name": f.get("full_name", "").strip()[:80] or g.user["username"],
                "age": num(f, "age", 14, 90, "Age", int), "gender": f.get("gender"),
                "height_cm": num(f, "height_cm", 120, 230, "Height"), "weight_kg": num(f, "weight_kg", 30, 250, "Weight"),
                "activity_level": f.get("activity_level"), "food_pref": f.get("food_pref"), "goal": f.get("goal"),
                "allergies": f.get("allergies", "").strip()[:200], "restrictions": f.get("restrictions", "").strip()[:200],
                "exercise_min": num(f, "exercise_min", 0, 300, "Exercise duration", int),
                "meals_per_day": num(f, "meals_per_day", 3, 5, "Meals per day", int)}
            if data["gender"] not in ("Male", "Female", "Other"): raise ValueError("Select a gender.")
            if data["activity_level"] not in cc.ACTIVITY: raise ValueError("Select an activity level.")
            if data["food_pref"] not in ("Vegetarian", "Non-Vegetarian", "Vegan"): raise ValueError("Select a food preference.")
            if data["goal"] not in cc.GOALS: raise ValueError("Select a fitness goal.")
        except ValueError as e:
            flash(str(e), "danger")
            return render_template("profile.html", p=f, existing=existing), 400
        db.save_profile(uid, data)
        db.add_weight(uid, data["weight_kg"])
        assess, ml = compute(data)
        db.add_assessment(uid, data, assess, ml)
        new_plan(uid, data, assess, ml)
        flash("Profile saved. Your calculations and a fresh 7-day diet plan are ready.", "success")
        return redirect(url_for("calculator"))
    return render_template("profile.html", p=existing or {"full_name": "", "age": g.user["age"], "gender": g.user["gender"],
                                                          "meals_per_day": 3, "exercise_min": 30}, existing=existing)


# --------------------------------------------------------------------------- dashboard
@app.route("/dashboard")
@login_required
def dashboard():
    uid = g.user["id"]
    prof = db.get_profile(uid)
    if not prof:
        return render_template("dashboard.html", profile=None)
    assess = cc.assess(prof)
    today = db.today()
    eaten = db.totals_for_date(uid, today)
    series = db.daily_series(uid, 7)
    week = db.challenge_week(uid, 7)
    stats = db.challenge_stats(uid)
    done = db.done_on(uid, today)
    weights = db.weight_history(uid, 12)
    data = {"macros": [assess["protein_g"], assess["carbs_g"], assess["fat_g"]],
            "cal_labels": [datetime.fromisoformat(r["date"]).strftime("%a") for r in series],
            "cal_values": [r["calories"] for r in series], "target": assess["target_calories"],
            "w_labels": [datetime.fromisoformat(w["log_date"]).strftime("%d %b") for w in weights],
            "w_values": [w["weight_kg"] for w in weights],
            "ch_labels": [datetime.fromisoformat(r["date"]).strftime("%a") for r in week], "ch_values": [r["pct"] for r in week]}
    return render_template("dashboard.html", profile=prof, a=assess, eaten=eaten, stats=stats, done=len(done),
                           pct=round(len(done) * 100 / len(db.CHALLENGES)), data=data,
                           tips=cc.suggestions(assess["goal_code"])[:3], warnings=assess["warnings"])


# --------------------------------------------------------------------------- calorie calculator (rule-based + ML shown separately)
@app.route("/calorie-calculator")
@login_required
@profile_required
def calculator():
    assess, ml = compute(g.profile)
    return render_template("calorie_calculator.html", p=g.profile, a=assess, ml=ml,
                           tips=cc.suggestions(assess["goal_code"]),
                           restr=dr.filter_foods(g.profile)[1], data={"macros": [assess["protein_g"], assess["carbs_g"], assess["fat_g"]]})


# --------------------------------------------------------------------------- diet plan
@app.route("/diet-plan")
@login_required
@profile_required
def diet_plan():
    uid = g.user["id"]
    pid = request.args.get("plan_id", type=int)
    plan = db.get_plan(uid, pid)
    if not plan:
        assess, ml = compute(g.profile)
        new_plan(uid, g.profile, assess, ml)
        plan = db.get_plan(uid)
    return render_template("diet_plan.html", plan=plan, p=g.profile, a=cc.assess(g.profile), latest=db.get_plan(uid)["id"])


@app.route("/diet-plan/regenerate", methods=["POST"])
@login_required
@profile_required
def regenerate_plan():
    assess, ml = compute(g.profile)
    new_plan(g.user["id"], g.profile, assess, ml)
    flash("A new 7-day diet plan was generated.", "success")
    return redirect(url_for("diet_plan"))


@app.route("/api/plan/swap", methods=["POST"])
@login_required
def swap():
    d = request.get_json(silent=True) or {}
    try:
        row = db.get_plan(g.user["id"], int(d["plan_id"]))
        if not row: abort(404)
        plan = dr.swap_item(row["plan"], int(d["day"]), int(d["meal"]), int(d["item"]), int(d["alt"]))
    except (KeyError, ValueError, IndexError, TypeError):
        return jsonify(error="Invalid swap request."), 400
    db.update_plan(g.user["id"], row["id"], plan)
    return jsonify(ok=True)


# --------------------------------------------------------------------------- food tracker
@app.route("/food-tracker", methods=["GET", "POST"])
@login_required
@profile_required
def food_tracker():
    uid = g.user["id"]
    if request.method == "POST":
        f = request.form
        try:
            d = datetime.strptime(f.get("log_date", ""), "%Y-%m-%d").date()
            if d > date.today(): raise ValueError("You cannot log food for a future date.")
            servings = num(f, "servings", 0.1, 20, "Quantity")
            name = f.get("food_name", "").strip()[:100]
            if not name: raise ValueError("Enter a food name.")
            meal = f.get("meal") if f.get("meal") in ("Breakfast", "Morning Snack", "Lunch", "Evening Snack", "Dinner") else "Snack"
            food = FOODS.get(name)
            if food and not f.get("custom"):
                row = {"servings": servings, "quantity_text": f"{dr.fmt(round(servings * food['unit_n'], 2))} {food['unit_text']}",
                       **{k: round(food[k] * servings, 1) for k in ("calories", "protein", "carbs", "fat")}}
            else:
                row = {"servings": servings, "quantity_text": f"{servings:g} serving(s)",
                       "calories": num(f, "calories", 0, 5000, "Calories"), "protein": num(f, "protein", 0, 500, "Protein"),
                       "carbs": num(f, "carbs", 0, 800, "Carbs"), "fat": num(f, "fat", 0, 500, "Fat")}
            db.add_food(uid, {**row, "log_date": d.isoformat(), "meal": meal, "food_name": name})
            flash(f"Added {name}.", "success")
            return redirect(url_for("food_tracker", date=d.isoformat()))
        except ValueError as e:
            flash(str(e), "danger")
            return redirect(url_for("food_tracker", date=f.get("log_date", "")))
    try:
        d = datetime.strptime(request.args.get("date", ""), "%Y-%m-%d").date()
    except ValueError:
        d = date.today()
    a = cc.assess(g.profile)
    tot = db.totals_for_date(uid, d.isoformat())
    logs = db.foods_for_date(uid, d.isoformat())
    foods = [{"name": n, "desc": f["serving_desc"], **{k: f[k] for k in ("calories", "protein", "carbs", "fat")}} for n, f in FOODS.items()]
    return render_template("food_tracker.html", d=d.isoformat(), today=date.today().isoformat(), a=a, tot=tot, logs=logs,
                           prev=(d - timedelta(days=1)).isoformat(), next=(d + timedelta(days=1)).isoformat(),
                           data={"foods": foods, "consumed": [tot["protein"], tot["carbs"], tot["fat"]]})


@app.route("/food-tracker/delete/<int:fid>", methods=["POST"])
@login_required
def delete_food(fid):
    db.delete_food(g.user["id"], fid)
    flash("Entry removed.", "info")
    return redirect(request.referrer or url_for("food_tracker"))


# --------------------------------------------------------------------------- challenges
@app.route("/challenges")
@login_required
def challenges():
    uid = g.user["id"]
    done = db.done_on(uid, db.today())
    stats = db.challenge_stats(uid)
    week = db.challenge_week(uid, 7)
    return render_template("challenges.html", done=done, stats=stats, week=week, badges=db.badges(stats),
                           history=db.challenge_history(uid, 14), today_pts=sum(db.CH_MAP[c]["points"] for c in done),
                           pct=round(len(done) * 100 / len(db.CHALLENGES)), max_pts=db.MAX_DAILY_POINTS,
                           week_pts=sum(r["points"] for r in week), week_pct=round(sum(r["done"] for r in week) * 100 / (7 * len(db.CHALLENGES))),
                           data={"labels": [datetime.fromisoformat(r["date"]).strftime("%a") for r in week], "values": [r["pct"] for r in week]})


@app.route("/api/challenge/toggle", methods=["POST"])
@login_required
def toggle_challenge():
    cid = (request.get_json(silent=True) or {}).get("challenge_id")
    try:
        completed = db.toggle_challenge(g.user["id"], cid)
    except ValueError:
        return jsonify(error="Unknown challenge."), 400
    done = db.done_on(g.user["id"], db.today())
    return jsonify(completed=completed, points=db.CH_MAP[cid]["points"], done=len(done),
                   pct=round(len(done) * 100 / len(db.CHALLENGES)))


# --------------------------------------------------------------------------- history & reports
@app.route("/history", methods=["GET", "POST"])
@login_required
def history():
    uid = g.user["id"]
    if request.method == "POST":
        try:
            w = num(request.form, "weight_kg", 30, 250, "Weight")
            d = datetime.strptime(request.form.get("log_date", db.today()), "%Y-%m-%d").date()
            if d > date.today(): raise ValueError("Date cannot be in the future.")
            db.add_weight(uid, w, d.isoformat())
            prof = db.get_profile(uid)
            if prof and d == date.today():
                db.save_profile(uid, {**prof, "weight_kg": w})
            flash("Weight saved.", "success")
        except ValueError as e:
            flash(str(e), "danger")
        return redirect(url_for("history", tab="weight"))
    weights = db.weight_history(uid, 60)
    series = db.daily_series(uid, 14)
    prof = db.get_profile(uid)
    target = cc.assess(prof)["target_calories"] if prof else None
    return render_template("history.html", assessments=db.list_assessments(uid), plans=db.list_plans(uid), series=series,
                           target=target, ch=db.challenge_history(uid, 14), weights=list(reversed(weights)), today=db.today(),
                           tab=request.args.get("tab", "assessments"),
                           data={"w_labels": [w["log_date"] for w in weights], "w_values": [w["weight_kg"] for w in weights],
                                 "c_labels": [r["date"][5:] for r in series], "c_values": [r["calories"] for r in series], "target": target})


@app.route("/report.pdf")
@login_required
@profile_required
def report():
    uid = g.user["id"]
    assess, ml = compute(g.profile)
    plan = db.get_plan(uid)
    buf = build_report(g.user, g.profile, assess, ml, plan, db.daily_series(uid, 7), db.weight_history(uid, 30),
                       db.challenge_history(uid, 7), db.challenge_stats(uid), cc.DISCLAIMER)
    return send_file(buf, mimetype="application/pdf", as_attachment=True,
                     download_name=f"nutrition_report_{date.today().isoformat()}.pdf")


@app.errorhandler(404)
def not_found(_):
    return render_template("error.html", code=404, msg="That page could not be found."), 404


@app.errorhandler(500)
def server_error(_):
    return render_template("error.html", code=500, msg="Something went wrong on our side. Please try again."), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=os.environ.get("FLASK_DEBUG") == "1")
