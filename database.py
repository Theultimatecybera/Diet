"""SQLite data layer. The database (database.db) and all tables are created automatically."""
import os
import json
import random
import sqlite3
from datetime import date, datetime, timedelta
from werkzeug.security import generate_password_hash

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "database.db")

CHALLENGES = [
    {"id": "water", "title": "Drink 8 glasses of water", "icon": "💧", "points": 10},
    {"id": "steps", "title": "Walk 8,000 steps", "icon": "👟", "points": 15},
    {"id": "exercise", "title": "Exercise for 30 minutes", "icon": "🏃", "points": 20},
    {"id": "fruits", "title": "Eat 2 servings of fruits", "icon": "🍎", "points": 10},
    {"id": "junk", "title": "Avoid junk food for one day", "icon": "🥗", "points": 15},
    {"id": "veggies", "title": "Eat vegetables with lunch and dinner", "icon": "🥦", "points": 10},
    {"id": "sleep", "title": "Sleep for 7-8 hours", "icon": "😴", "points": 10},
]
CH_MAP = {c["id"]: c for c in CHALLENGES}
MAX_DAILY_POINTS = sum(c["points"] for c in CHALLENGES)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users(
  id INTEGER PRIMARY KEY AUTOINCREMENT, username TEXT UNIQUE NOT NULL, email TEXT UNIQUE NOT NULL,
  password_hash TEXT NOT NULL, age INTEGER, gender TEXT, created_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS profiles(
  user_id INTEGER PRIMARY KEY REFERENCES users(id) ON DELETE CASCADE, full_name TEXT, age INTEGER, gender TEXT,
  height_cm REAL, weight_kg REAL, activity_level TEXT, food_pref TEXT, goal TEXT, allergies TEXT,
  restrictions TEXT, exercise_min INTEGER, meals_per_day INTEGER, updated_at TEXT);
CREATE TABLE IF NOT EXISTS assessments(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at TEXT, weight_kg REAL, height_cm REAL, bmi REAL, bmr INTEGER, tdee INTEGER, calories INTEGER,
  protein_g INTEGER, carbs_g INTEGER, fat_g INTEGER, goal TEXT, ml_low INTEGER, ml_high INTEGER, ml_focus TEXT);
CREATE TABLE IF NOT EXISTS weight_logs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  log_date TEXT NOT NULL, weight_kg REAL NOT NULL, UNIQUE(user_id, log_date));
CREATE TABLE IF NOT EXISTS diet_plans(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  created_at TEXT, calories INTEGER, goal TEXT, food_pref TEXT, focus TEXT, plan_json TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS food_logs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  log_date TEXT NOT NULL, meal TEXT, food_name TEXT NOT NULL, quantity_text TEXT, servings REAL,
  calories REAL, protein REAL, carbs REAL, fat REAL);
CREATE TABLE IF NOT EXISTS challenge_logs(
  id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  log_date TEXT NOT NULL, challenge_id TEXT NOT NULL, points INTEGER NOT NULL,
  UNIQUE(user_id, log_date, challenge_id));
CREATE INDEX IF NOT EXISTS idx_food ON food_logs(user_id, log_date);
CREATE INDEX IF NOT EXISTS idx_ch ON challenge_logs(user_id, log_date);
"""


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    return c


def q(sql, args=(), one=False):
    with _conn() as c:
        rows = c.execute(sql, args).fetchall()
    rows = [dict(r) for r in rows]
    return (rows[0] if rows else None) if one else rows


def ex(sql, args=()):
    c = _conn()
    try:
        with c:
            return c.execute(sql, args).lastrowid
    finally:
        c.close()


def today():
    return date.today().isoformat()


def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def init_db():
    c = _conn()
    c.executescript(SCHEMA)
    c.close()
    if not q("SELECT id FROM users WHERE username='demo'", one=True):
        seed_demo()


# ------------------------------------------------------------------ users
def create_user(username, email, password, age, gender):
    return ex("INSERT INTO users(username,email,password_hash,age,gender,created_at) VALUES(?,?,?,?,?,?)",
              (username, email.lower(), generate_password_hash(password), age, gender, now()))


def find_user(identifier):
    return q("SELECT * FROM users WHERE lower(username)=lower(?) OR email=lower(?)", (identifier, identifier), one=True)


def get_user(uid):
    return q("SELECT id,username,email,age,gender,created_at FROM users WHERE id=?", (uid,), one=True)


def user_exists(username, email):
    return q("SELECT id FROM users WHERE lower(username)=lower(?) OR email=lower(?)", (username, email), one=True)


# ------------------------------------------------------------------ profile / assessments / weight
def get_profile(uid):
    return q("SELECT * FROM profiles WHERE user_id=?", (uid,), one=True)


def save_profile(uid, d):
    ex("""INSERT INTO profiles(user_id,full_name,age,gender,height_cm,weight_kg,activity_level,food_pref,goal,
          allergies,restrictions,exercise_min,meals_per_day,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)
          ON CONFLICT(user_id) DO UPDATE SET full_name=excluded.full_name, age=excluded.age, gender=excluded.gender,
          height_cm=excluded.height_cm, weight_kg=excluded.weight_kg, activity_level=excluded.activity_level,
          food_pref=excluded.food_pref, goal=excluded.goal, allergies=excluded.allergies,
          restrictions=excluded.restrictions, exercise_min=excluded.exercise_min,
          meals_per_day=excluded.meals_per_day, updated_at=excluded.updated_at""",
       (uid, d["full_name"], d["age"], d["gender"], d["height_cm"], d["weight_kg"], d["activity_level"],
        d["food_pref"], d["goal"], d["allergies"], d["restrictions"], d["exercise_min"], d["meals_per_day"], now()))


def add_assessment(uid, p, a, ml):
    ex("""INSERT INTO assessments(user_id,created_at,weight_kg,height_cm,bmi,bmr,tdee,calories,protein_g,carbs_g,fat_g,
          goal,ml_low,ml_high,ml_focus) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
       (uid, now(), p["weight_kg"], p["height_cm"], a["bmi"], a["bmr"], a["tdee"], a["target_calories"],
        a["protein_g"], a["carbs_g"], a["fat_g"], p["goal"], ml["calories_low"], ml["calories_high"], ml["focus_label"]))


def list_assessments(uid, limit=20):
    return q("SELECT * FROM assessments WHERE user_id=? ORDER BY id DESC LIMIT ?", (uid, limit))


def add_weight(uid, weight, log_date=None):
    ex("""INSERT INTO weight_logs(user_id,log_date,weight_kg) VALUES(?,?,?)
          ON CONFLICT(user_id,log_date) DO UPDATE SET weight_kg=excluded.weight_kg""",
       (uid, log_date or today(), weight))


def weight_history(uid, limit=60):
    rows = q("SELECT log_date, weight_kg FROM weight_logs WHERE user_id=? ORDER BY log_date DESC LIMIT ?", (uid, limit))
    return list(reversed(rows))


# ------------------------------------------------------------------ diet plans
def save_plan(uid, plan, calories, goal, food_pref, focus):
    return ex("INSERT INTO diet_plans(user_id,created_at,calories,goal,food_pref,focus,plan_json) VALUES(?,?,?,?,?,?,?)",
              (uid, now(), calories, goal, food_pref, focus, json.dumps(plan)))


def get_plan(uid, plan_id=None):
    row = (q("SELECT * FROM diet_plans WHERE id=? AND user_id=?", (plan_id, uid), one=True) if plan_id else
           q("SELECT * FROM diet_plans WHERE user_id=? ORDER BY id DESC LIMIT 1", (uid,), one=True))
    if row:
        row["plan"] = json.loads(row.pop("plan_json"))
    return row


def update_plan(uid, plan_id, plan):
    ex("UPDATE diet_plans SET plan_json=? WHERE id=? AND user_id=?", (json.dumps(plan), plan_id, uid))


def list_plans(uid, limit=20):
    return q("SELECT id,created_at,calories,goal,food_pref,focus FROM diet_plans WHERE user_id=? ORDER BY id DESC LIMIT ?",
             (uid, limit))


# ------------------------------------------------------------------ food tracker
def add_food(uid, d):
    return ex("""INSERT INTO food_logs(user_id,log_date,meal,food_name,quantity_text,servings,calories,protein,carbs,fat)
                 VALUES(?,?,?,?,?,?,?,?,?,?)""",
              (uid, d["log_date"], d["meal"], d["food_name"], d["quantity_text"], d["servings"],
               d["calories"], d["protein"], d["carbs"], d["fat"]))


def delete_food(uid, fid):
    ex("DELETE FROM food_logs WHERE id=? AND user_id=?", (fid, uid))


def foods_for_date(uid, d):
    return q("SELECT * FROM food_logs WHERE user_id=? AND log_date=? ORDER BY id", (uid, d))


def totals_for_date(uid, d):
    r = q("""SELECT COALESCE(SUM(calories),0) calories, COALESCE(SUM(protein),0) protein,
             COALESCE(SUM(carbs),0) carbs, COALESCE(SUM(fat),0) fat FROM food_logs WHERE user_id=? AND log_date=?""",
          (uid, d), one=True)
    return {k: round(v, 1) for k, v in r.items()}


def daily_series(uid, days=7):
    start = date.today() - timedelta(days=days - 1)
    rows = q("""SELECT log_date, SUM(calories) c, SUM(protein) p, SUM(carbs) cb, SUM(fat) f FROM food_logs
                WHERE user_id=? AND log_date>=? GROUP BY log_date""", (uid, start.isoformat()))
    m = {r["log_date"]: r for r in rows}
    out = []
    for i in range(days):
        d = (start + timedelta(days=i)).isoformat()
        r = m.get(d)
        out.append({"date": d, "calories": round(r["c"]) if r else 0, "protein": round(r["p"]) if r else 0,
                    "carbs": round(r["cb"]) if r else 0, "fat": round(r["f"]) if r else 0})
    return out


# ------------------------------------------------------------------ challenges
def toggle_challenge(uid, cid, d=None):
    d = d or today()
    if cid not in CH_MAP:
        raise ValueError("Unknown challenge")
    if q("SELECT id FROM challenge_logs WHERE user_id=? AND log_date=? AND challenge_id=?", (uid, d, cid), one=True):
        ex("DELETE FROM challenge_logs WHERE user_id=? AND log_date=? AND challenge_id=?", (uid, d, cid))
        return False
    ex("INSERT INTO challenge_logs(user_id,log_date,challenge_id,points) VALUES(?,?,?,?)",
       (uid, d, cid, CH_MAP[cid]["points"]))
    return True


def done_on(uid, d):
    return [r["challenge_id"] for r in q("SELECT challenge_id FROM challenge_logs WHERE user_id=? AND log_date=?", (uid, d))]


def challenge_week(uid, days=7):
    start = date.today() - timedelta(days=days - 1)
    rows = q("""SELECT log_date, COUNT(*) n, SUM(points) pts FROM challenge_logs WHERE user_id=? AND log_date>=?
                GROUP BY log_date""", (uid, start.isoformat()))
    m = {r["log_date"]: r for r in rows}
    out = []
    for i in range(days):
        d = (start + timedelta(days=i)).isoformat()
        n = m[d]["n"] if d in m else 0
        out.append({"date": d, "done": n, "points": m[d]["pts"] if d in m else 0,
                    "pct": round(n * 100 / len(CHALLENGES))})
    return out


def challenge_history(uid, days=14):
    return list(reversed(challenge_week(uid, days)))


def challenge_stats(uid):
    rows = q("SELECT log_date, challenge_id, points FROM challenge_logs WHERE user_id=?", (uid,))
    total_pts = sum(r["points"] for r in rows)
    per_day, per_ch = {}, {}
    for r in rows:
        per_day[r["log_date"]] = per_day.get(r["log_date"], 0) + 1
        per_ch[r["challenge_id"]] = per_ch.get(r["challenge_id"], 0) + 1
    streak, d = 0, date.today()
    if d.isoformat() not in per_day:
        d -= timedelta(days=1)
    while d.isoformat() in per_day:
        streak += 1
        d -= timedelta(days=1)
    perfect = sum(1 for n in per_day.values() if n == len(CHALLENGES))
    return {"points": total_pts, "streak": streak, "perfect_days": perfect, "completions": len(rows),
            "per_challenge": per_ch, "active_days": len(per_day)}


def badges(stats):
    pc = stats["per_challenge"]
    defs = [
        ("First Step", "Complete your first challenge", "🌱", stats["completions"], 1),
        ("Hydration Hero", "Drink 8 glasses of water on 7 days", "💧", pc.get("water", 0), 7),
        ("Step Master", "Hit 8,000 steps on 7 days", "👟", pc.get("steps", 0), 7),
        ("Fruit Lover", "Eat 2 fruit servings on 7 days", "🍎", pc.get("fruits", 0), 7),
        ("Fitness Fanatic", "Exercise for 30 minutes on 7 days", "💪", pc.get("exercise", 0), 7),
        ("Perfect Day", "Complete every challenge in one day", "🌟", stats["perfect_days"], 1),
        ("Streak Starter", "Stay active 3 days in a row", "🔥", stats["streak"], 3),
        ("Week Warrior", "Stay active 7 days in a row", "🏅", stats["streak"], 7),
        ("Point Collector", "Earn 250 points", "🥉", stats["points"], 250),
        ("Nutrition Champion", "Earn 1,000 points", "🏆", stats["points"], 1000),
    ]
    return [{"name": n, "desc": d, "icon": i, "current": min(c, t), "target": t, "earned": c >= t}
            for n, d, i, c, t in defs]


# ------------------------------------------------------------------ demo data
def seed_demo():
    uid = create_user("demo", "demo@example.com", "Demo@1234", 28, "Male")
    save_profile(uid, {"full_name": "Aarav Sharma", "age": 28, "gender": "Male", "height_cm": 175, "weight_kg": 78,
                       "activity_level": "moderate", "food_pref": "Vegetarian", "goal": "Weight Loss",
                       "allergies": "peanut", "restrictions": "none", "exercise_min": 30, "meals_per_day": 5})
    rng = random.Random(7)
    for w in range(8):  # weekly weights, oldest first
        add_weight(uid, round(82.0 - w * 0.55 + rng.uniform(-0.2, 0.2), 1), (date.today() - timedelta(days=7 * (7 - w))).isoformat())
    add_weight(uid, 78.0, today())
    meals = [("Breakfast", "Oats porridge (with water)", 150, 5, 27, 3), ("Breakfast", "Banana", 105, 1.3, 27, 0.4),
             ("Lunch", "Roti (phulka) x2", 200, 6, 36, 4), ("Lunch", "Dal tadka", 160, 8, 22, 4),
             ("Evening Snack", "Roasted chana", 110, 6, 18, 2), ("Dinner", "Paneer bhurji", 220, 14, 5, 16),
             ("Dinner", "Mixed vegetable sabzi", 110, 3, 12, 6)]
    for i in range(1, 7):
        d = (date.today() - timedelta(days=i)).isoformat()
        for m, n, c, p, cb, f in meals[: rng.randint(5, 7)]:
            k = rng.uniform(0.9, 1.25)
            add_food(uid, {"log_date": d, "meal": m, "food_name": n, "quantity_text": "1 serving", "servings": 1,
                           "calories": round(c * k), "protein": round(p * k, 1), "carbs": round(cb * k, 1), "fat": round(f * k, 1)})
        for ch in rng.sample([c["id"] for c in CHALLENGES], rng.randint(3, 7)):
            toggle_challenge(uid, ch, d)
