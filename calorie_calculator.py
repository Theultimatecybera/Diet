"""Rule-based nutrition calculations (NOT machine learning).

Uses published, widely accepted formulas:
  * BMI  = weight(kg) / height(m)^2                  (WHO categories)
  * BMR  = Mifflin-St Jeor equation (1990)
  * TDEE = BMR x activity factor
Estimates are for general guidance only and are not a medical diagnosis.
"""

ACTIVITY = {
    "sedentary": (1.2, "Sedentary (little or no exercise)"),
    "light": (1.375, "Lightly active (1-3 days/week)"),
    "moderate": (1.55, "Moderately active (3-5 days/week)"),
    "active": (1.725, "Very active (6-7 days/week)"),
    "extra": (1.9, "Extra active (hard training / physical job)"),
}
GOALS = ["Weight Loss", "Weight Gain", "Maintain Weight"]
DISCLAIMER = ("These values are estimates based on standard equations and are not a medical diagnosis "
              "or treatment advice. Consult a doctor or registered dietitian before making major diet "
              "changes, especially if you are pregnant, have a medical condition, or take medication.")


def goal_code(goal):
    g = (goal or "").lower()
    return "loss" if "loss" in g else "gain" if "gain" in g else "maintain"


def activity_factor(level):
    return ACTIVITY.get(level, ACTIVITY["sedentary"])[0]


def bmi(weight_kg, height_cm):
    return weight_kg / ((height_cm / 100.0) ** 2)


def bmi_category(value):
    if value < 18.5:
        return "Underweight"
    if value < 25:
        return "Normal weight"
    if value < 30:
        return "Overweight"
    return "Obese"


def healthy_weight_range(height_cm):
    m2 = (height_cm / 100.0) ** 2
    return round(18.5 * m2, 1), round(24.9 * m2, 1)


def bmr_mifflin(weight_kg, height_cm, age, gender):
    base = 10 * weight_kg + 6.25 * height_cm - 5 * age
    g = (gender or "").lower()
    return base + (5 if g == "male" else -161 if g == "female" else -78)


def min_safe_calories(gender):
    g = (gender or "").lower()
    return 1500 if g == "male" else 1200 if g == "female" else 1350


def macro_split(kcal, weight_kg, goal_c):
    per_kg = 1.2 if goal_c == "maintain" else 1.6
    protein = min(per_kg * weight_kg, 0.30 * kcal / 4)
    fat = (0.28 if goal_c == "maintain" else 0.25) * kcal / 9
    carbs = max((kcal - protein * 4 - fat * 9) / 4, 0)
    return round(protein), round(carbs), round(fat)


def water_intake(weight_kg, exercise_min):
    litres = (35 * weight_kg + 350 * (exercise_min / 30.0)) / 1000.0
    litres = round(min(max(litres, 1.5), 5.0), 1)
    return litres, round(litres * 1000 / 250)


def assess(p):
    """Full rule-based assessment from a profile dict."""
    w, h, age = float(p["weight_kg"]), float(p["height_cm"]), int(p["age"])
    gender, goal = p["gender"], p["goal"]
    b = bmi(w, h)
    bmr = bmr_mifflin(w, h, age, gender)
    tdee = bmr * activity_factor(p["activity_level"])
    code = goal_code(goal)
    notes, warnings = [], []
    floor = max(min_safe_calories(gender), bmr)
    if code == "loss":
        if b < 18.5:
            target, code = tdee, "maintain"
            warnings.append("Your BMI is in the underweight range, so a weight-loss calorie target was NOT applied. "
                            "Your plan is set to maintenance. Please speak with a doctor or dietitian.")
        else:
            deficit = min(500, 0.20 * tdee)
            target = max(tdee - deficit, min(floor, tdee))
            notes.append(f"Moderate deficit of about {round(tdee - target)} kcal/day "
                         "(about 0.4 kg loss per week). Extreme deficits are avoided on purpose.")
    elif code == "gain":
        target = tdee + 400
        notes.append("Moderate surplus of about 400 kcal/day for gradual, mostly lean weight gain.")
        if b >= 30:
            warnings.append("Your BMI is in the obese range. Please consult a doctor before aiming for weight gain.")
    else:
        target = tdee
    target = round(target / 10) * 10
    protein, carbs, fat = macro_split(target, w, code)
    litres, glasses = water_intake(w, float(p.get("exercise_min") or 0))
    lo, hi = healthy_weight_range(h)
    return {
        "bmi": round(b, 1), "bmi_category": bmi_category(b), "healthy_range": [lo, hi],
        "bmr": round(bmr), "tdee": round(tdee), "target_calories": int(target),
        "protein_g": protein, "carbs_g": carbs, "fat_g": fat,
        "fiber_g": round(14 * target / 1000),
        "water_l": litres, "water_glasses": glasses,
        "goal_code": code, "notes": notes, "warnings": warnings,
        "formula": "Mifflin-St Jeor", "disclaimer": DISCLAIMER,
    }


SUGGESTIONS = {
    "loss": ["Build meals around vegetables, dal, lean protein and whole grains to stay full on fewer calories.",
             "Use portion control: a plate that is half vegetables, a quarter protein, a quarter grains.",
             "Aim for 150+ minutes of moderate exercise a week and include 2 days of strength training.",
             "Target a loss of 0.25-0.5 kg per week. Faster loss is not recommended without medical supervision.",
             "Limit sugary drinks, deep-fried snacks and sweets; choose fruit, roasted chana or nuts instead."],
    "gain": ["Add calorie-dense, nutritious foods: nuts, milk, paneer, bananas and whole grains.",
             "Eat protein with every meal (dal, eggs, paneer, soya, curd) to support muscle growth.",
             "Eat 4-5 smaller meals if big meals are hard to finish; add a snack before bed.",
             "Pair the calorie surplus with 3+ strength-training sessions per week.",
             "Gain gradually (0.25-0.5 kg per week) and avoid relying on junk food for extra calories."],
    "maintain": ["Keep meals balanced: whole grains, protein, vegetables, fruit and healthy fats.",
                 "Stay active most days and keep your routine consistent.",
                 "Weigh yourself weekly and adjust portions if your weight drifts by more than 2 kg.",
                 "Drink enough water and limit processed, high-sugar foods."],
}


def suggestions(goal_c):
    return SUGGESTIONS.get(goal_c, SUGGESTIONS["maintain"]) + [
        "Aim for 7-8 hours of sleep and 5+ servings of vegetables and fruit a day."]
