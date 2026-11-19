"""Diet plan generator: filters the food table by preference/allergies and builds a 7-day plan.

The calorie target comes from the rule-based calculator. The ML model contributes only the
"food focus" (which food categories/nutrient profile to favour). Portions are then fitted to
each meal's calorie budget with a small deterministic optimiser.
"""
import os
import re
import random
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOOD_CSV = os.path.join(BASE_DIR, "dataset", "nutrition_dataset.csv")
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

MEAL_SHARES = {
    3: [("Breakfast", .30), ("Lunch", .40), ("Dinner", .30)],
    4: [("Breakfast", .25), ("Lunch", .35), ("Evening Snack", .10), ("Dinner", .30)],
    5: [("Breakfast", .25), ("Morning Snack", .10), ("Lunch", .30), ("Evening Snack", .10), ("Dinner", .25)],
}
TEMPLATES = {
    "Breakfast": [["breakfast"], ["protein", "dairy", "pulse"], ["fruit"]],
    "Morning Snack": [["fruit"], ["nuts", "snack"]],
    "Lunch": [["grain"], ["pulse", "protein"], ["vegetable"], ["dairy", "vegetable"]],
    "Evening Snack": [["snack", "dairy"], ["nuts", "fruit", "snack"]],
    "Dinner": [["grain"], ["pulse", "protein"], ["vegetable"], ["dairy", "vegetable"]],
}
MAX_SERV = {"grain": 4, "breakfast": 3, "protein": 3, "pulse": 2.5, "vegetable": 2, "dairy": 2, "fruit": 2,
            "nuts": 2, "snack": 3}
ALLERGEN_WORDS = {
    "milk": ["milk", "dairy", "lactose", "paneer", "curd", "yogurt", "yoghurt", "dahi", "ghee", "cheese"],
    "gluten": ["gluten", "wheat", "celiac", "coeliac", "atta"],
    "egg": ["egg"], "fish": ["fish", "seafood", "shellfish", "prawn", "prawns"],
    "nuts": ["nut", "almond", "walnut", "cashew", "pistachio", "treenut"],
    "peanut": ["peanut", "groundnut"], "soy": ["soy", "soya", "tofu"], "sesame": ["sesame", "til"],
}
STOP = {"none", "nil", "not", "any", "and", "avoid", "allergic", "allergy", "allergies", "with", "restriction",
        "restrictions", "free", "diet", "food", "foods", "low", "high", "intake", "less", "reduce", "have", "has",
        "dont", "eat", "the", "for", "too", "much", "also", "mild", "severe", "very", "some", "etc"}
_foods = None


def load_foods():
    global _foods
    if _foods is None:
        df = pd.read_csv(FOOD_CSV).fillna("")
        rows = []
        for r in df.to_dict("records"):
            m = re.match(r"^(\d+(?:\.\d+)?)\s*(.*)$", str(r["serving_desc"]))
            r["unit_n"], r["unit_text"] = (float(m.group(1)), m.group(2)) if m else (1.0, str(r["serving_desc"]))
            r["meal_types"] = set(str(r["meal_types"]).split("|"))
            r["allergens"] = set(a for a in str(r["allergens"]).split(";") if a)
            r["tags"] = set(t for t in str(r["tags"]).split(";") if t)
            rows.append(r)
        _foods = rows
    return _foods


def parse_restrictions(*texts):
    text = " ".join(t or "" for t in texts).lower()
    allergens, name_ex, avoid_tags, notes = set(), set(), set(), []
    for tok in re.findall(r"[a-z]+", text):
        stem = tok[:-1] if tok.endswith("s") and len(tok) > 4 else tok
        hit = [a for a, words in ALLERGEN_WORDS.items() if tok in words or stem in words]
        if hit:
            allergens.update(hit)
        elif stem.startswith("diabet") or tok in ("sugar", "pcos", "prediabetes"):
            avoid_tags.add("sweet")
            notes.append("Sugary foods (e.g. mango, banana, dates) were avoided because of your sugar/diabetes note. "
                         "Please follow your doctor's advice for blood-sugar management.")
        elif tok in ("cholesterol", "heart", "cardiac"):
            avoid_tags.add("fried")
            notes.append("Fried foods were avoided because of your cholesterol/heart note.")
        elif tok in ("bp", "hypertension", "pressure", "sodium", "salt", "thyroid", "kidney", "pregnant", "pregnancy"):
            notes.append(f"'{tok}' noted: this planner cannot tailor to medical conditions. Please follow your doctor's guidance.")
        elif len(tok) >= 4 and tok not in STOP and stem not in STOP:
            name_ex.add(stem)
    return allergens, name_ex, avoid_tags, sorted(set(notes))


def filter_foods(profile):
    pref = (profile.get("food_pref") or "Vegetarian").lower()
    allowed = {"vegan": {"vegan"}, "vegetarian": {"vegan", "vegetarian"}}.get(pref, {"vegan", "vegetarian", "non-vegetarian"})
    allergens, name_ex, avoid_tags, notes = parse_restrictions(profile.get("allergies"), profile.get("restrictions"))
    out, warnings = [], []
    for f in load_foods():
        if f["diet_type"] not in allowed or f["allergens"] & allergens or f["tags"] & avoid_tags:
            continue
        if any(w in f["food_name"].lower() for w in name_ex):
            continue
        out.append(f)
    if allergens:
        warnings.append("Allergy filter active: foods containing " + ", ".join(sorted(allergens)) +
                        " were excluded. Always check food labels and ingredients yourself - this tool cannot guarantee allergen safety.")
    if name_ex:
        warnings.append("Also excluded foods matching: " + ", ".join(sorted(name_ex)) + ".")
    return out, warnings + notes


def fmt(x):
    return f"{x:g}"


def make_item(f, servings):
    s = round(servings, 2)
    return {"food": f["food_name"], "category": f["category"], "servings": s,
            "qty_text": f"{fmt(round(s * f['unit_n'], 2))} {f['unit_text']}", "grams": round(s * f["serving_g"]),
            "calories": round(s * f["calories"]), "protein": round(s * f["protein"], 1),
            "carbs": round(s * f["carbs"], 1), "fat": round(s * f["fat"], 1)}


def focus_weight(f, focus):
    cal = max(f["calories"], 1)
    pshare, cshare = 4 * f["protein"] / cal, 4 * f["carbs"] / cal
    w = {"high_protein_lean": 0.3 + 6 * pshare - cal / 500, "calorie_dense_protein": 0.5 + 3 * pshare + cal / 250,
         "high_carb_energy": 0.4 + 2.5 * cshare}.get(focus, 1.0)
    return max(w, 0.1)


def pick(cands, focus, rng):
    return rng.choices(cands, weights=[focus_weight(f, focus) for f in cands], k=1)[0]


def fit_servings(foods, target):
    """Choose servings per food so the meal's calories are as close to `target` as possible."""
    base = sum(f["calories"] for f in foods) or 1
    s = [min(max(round(target / base / f["step"]) * f["step"], f["step"]), MAX_SERV[f["category"]]) for f in foods]
    for _ in range(40):
        cur = abs(sum(a * f["calories"] for a, f in zip(s, foods)) - target)
        best = None
        for i, f in enumerate(foods):
            for d in (f["step"], -f["step"]):
                n = s[i] + d
                if n < f["step"] - 1e-9 or n > MAX_SERV[f["category"]] + 1e-9:
                    continue
                t = s[:i] + [n] + s[i + 1:]
                err = abs(sum(a * g["calories"] for a, g in zip(t, foods)) - target)
                if err < cur - 1e-9 and (best is None or err < best[0]):
                    best = (err, i, n)
        if not best:
            break
        s[best[1]] = best[2]
    return s


def totals(items):
    return {k: round(sum(i[k] for i in items), 1) for k in ("calories", "protein", "carbs", "fat")}


def build_meal(name, target, foods, focus, rng, recent):
    key = "snack" if "Snack" in name else name.lower()
    pool = [f for f in foods if key in f["meal_types"]]
    chosen = []
    for n, slot in enumerate(TEMPLATES[name]):
        if "Snack" in name and n == 1 and target < 140:
            continue
        cands = [f for f in pool if f["category"] in slot and f not in chosen]
        fresh = [f for f in cands if f["food_name"] not in recent]
        cands = fresh or cands
        if cands:
            chosen.append(pick(cands, focus, rng))
    if not chosen:
        return {"name": name, "target": round(target), "items": [], "totals": totals([])}
    serv = fit_servings(chosen, target)
    items = []
    for f, s in zip(chosen, serv):
        it = make_item(f, s)
        alts = [f2 for f2 in pool if f2["category"] == f["category"] and f2["food_name"] != f["food_name"]
                and f2 not in chosen]
        rng.shuffle(alts)
        it["alternatives"] = []
        for a in alts[:2]:
            sv = min(max(round(it["calories"] / a["calories"] / a["step"]) * a["step"], a["step"]), MAX_SERV[a["category"]])
            it["alternatives"].append(make_item(a, sv))
        items.append(it)
    recent.extend(f["food_name"] for f in chosen)
    del recent[:-8]
    return {"name": name, "target": round(target), "items": items, "totals": totals(items)}


def generate_plan(profile, assessment, focus, seed=None):
    rng = random.Random(seed)
    foods, warnings = filter_foods(profile)
    target = assessment["target_calories"]
    structure = MEAL_SHARES.get(int(profile.get("meals_per_day") or 3), MEAL_SHARES[3])
    recent = {name: [] for name, _ in structure}
    days = []
    for day in DAYS:
        meals = [build_meal(n, target * share, foods, focus, rng, recent[n]) for n, share in structure]
        days.append({"day": day, "meals": meals, "totals": totals([i for m in meals for i in m["items"]])})
    avg = round(sum(d["totals"]["calories"] for d in days) / 7)
    return {"target_calories": target, "avg_calories": avg, "focus": focus, "warnings": warnings, "days": days,
            "foods_available": len(foods)}


def swap_item(plan, day, meal, item, alt):
    it = plan["days"][day]["meals"][meal]["items"][item]
    new = dict(it["alternatives"][alt])
    others = [a for k, a in enumerate(it["alternatives"]) if k != alt]
    old = {k: v for k, v in it.items() if k != "alternatives"}
    new["alternatives"] = [old] + others
    plan["days"][day]["meals"][meal]["items"][item] = new
    m = plan["days"][day]["meals"][meal]
    m["totals"] = totals(m["items"])
    plan["days"][day]["totals"] = totals([i for mm in plan["days"][day]["meals"] for i in mm["items"]])
    plan["avg_calories"] = round(sum(d["totals"]["calories"] for d in plan["days"]) / 7)
    return plan
