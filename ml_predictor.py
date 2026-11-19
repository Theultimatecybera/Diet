"""Machine-learning helpers: feature engineering + loading the Joblib model.

The model (trained in train_model.py with scikit-learn RandomForests) predicts:
  1. a suitable daily calorie RANGE (regression; spread of the forest's trees)
  2. the best-fitting FOOD FOCUS category (classification) used to bias food selection

This is deliberately separate from utils/calorie_calculator.py, which holds the
rule-based formulas (BMI/BMR/TDEE).
"""
import os
import numpy as np
import pandas as pd
import joblib

from utils.calorie_calculator import activity_factor, goal_code, min_safe_calories

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
FEATURES = ["age", "gender_code", "height_cm", "weight_kg", "bmi", "activity_factor",
            "exercise_min", "meals_per_day", "goal_loss", "goal_gain"]

FOCUS_INFO = {
    "high_protein_lean": {
        "label": "High-protein, lean & high-fibre",
        "categories": ["pulse", "protein", "vegetable", "fruit"],
        "text": "Prioritise lean protein, dals, sprouts, vegetables and fruit; keep refined carbs and fried foods low."},
    "balanced": {
        "label": "Balanced macros",
        "categories": ["grain", "pulse", "vegetable", "dairy", "fruit"],
        "text": "A balanced mix of whole grains, pulses, vegetables, dairy and fruit across the day."},
    "calorie_dense_protein": {
        "label": "Calorie-dense with high protein",
        "categories": ["nuts", "dairy", "protein", "grain"],
        "text": "Add nutrient-dense foods such as nuts, milk, paneer/eggs and whole grains to reach your calories."},
    "high_carb_energy": {
        "label": "Carb-rich for active energy",
        "categories": ["grain", "fruit", "breakfast", "snack"],
        "text": "Lean on whole grains, fruit and energy-rich snacks to fuel regular training."},
}

_cache = {}


def encode_dataframe(df):
    """Raw columns -> numeric model features. Used for BOTH training and prediction."""
    out = pd.DataFrame(index=df.index)
    out["age"] = df["age"].astype(float)
    out["gender_code"] = df["gender"].str.lower().map({"male": 1.0, "female": 0.0}).fillna(0.5)
    out["height_cm"] = df["height_cm"].astype(float)
    out["weight_kg"] = df["weight_kg"].astype(float)
    out["bmi"] = out["weight_kg"] / (out["height_cm"] / 100.0) ** 2
    out["activity_factor"] = df["activity_level"].map(lambda x: activity_factor(x))
    out["exercise_min"] = df["exercise_min"].astype(float)
    out["meals_per_day"] = df["meals_per_day"].astype(float)
    codes = df["goal"].map(goal_code)
    out["goal_loss"] = (codes == "loss").astype(float)
    out["goal_gain"] = (codes == "gain").astype(float)
    return out[FEATURES]


def profile_to_frame(p):
    return pd.DataFrame([{
        "age": p["age"], "gender": p["gender"], "height_cm": p["height_cm"], "weight_kg": p["weight_kg"],
        "activity_level": p["activity_level"], "exercise_min": p.get("exercise_min") or 0,
        "meals_per_day": p.get("meals_per_day") or 3, "goal": p["goal"]}])


def load_model(force=False):
    if force or "bundle" not in _cache:
        _cache["bundle"] = joblib.load(MODEL_PATH)
    return _cache["bundle"]


def ensure_model():
    """Load model.pkl; (re)train automatically if it is missing or unreadable."""
    try:
        return load_model()
    except Exception as exc:  # missing file or incompatible scikit-learn version
        print(f"[ML] model.pkl unavailable ({exc}). Training a new model...")
        import train_model
        train_model.main(quiet=True)
        return load_model(force=True)


def predict(profile):
    """Predict calorie range + food focus for one user profile (real model inference)."""
    bundle = ensure_model()
    X = encode_dataframe(profile_to_frame(profile)).values
    reg, clf = bundle["calorie_model"], bundle["focus_model"]
    tree_preds = np.array([t.predict(X)[0] for t in reg.estimators_])
    mean = float(reg.predict(X)[0])
    floor = min_safe_calories(profile["gender"])
    lo = max(float(np.percentile(tree_preds, 10)), floor)
    hi = max(float(np.percentile(tree_preds, 90)), lo + 50)
    proba = clf.predict_proba(X)[0]
    classes = list(clf.classes_)
    best = classes[int(np.argmax(proba))]
    info = FOCUS_INFO[best]
    return {
        "calories_pred": int(round(max(mean, floor) / 10) * 10),
        "calories_low": int(round(lo / 10) * 10), "calories_high": int(round(hi / 10) * 10),
        "focus": best, "focus_label": info["label"], "focus_text": info["text"],
        "focus_categories": info["categories"], "confidence": round(float(max(proba)) * 100, 1),
        "probabilities": {FOCUS_INFO[c]["label"]: round(float(p) * 100, 1) for c, p in zip(classes, proba)},
        "metrics": bundle.get("metrics", {}), "trained_at": bundle.get("trained_at", ""),
    }
