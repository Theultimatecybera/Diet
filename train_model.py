"""Train the Machine Learning models for the Smart Nutrition & Diet Planner.

Usage:   python train_model.py            (trains, evaluates, writes model.pkl + model_report.*)
         python train_model.py --regenerate   (also rebuilds dataset/user_profiles.csv)

DATA
  * dataset/user_profiles.csv   - training data: one row per person (age, gender, height, weight,
                                  activity, exercise, meals, goal) with two labels:
                                  target_calories (regression) and diet_focus (classification).
  * dataset/nutrition_dataset.csv - food composition table used by the diet recommender.
    It is validated/cleaned here as well.

IMPORTANT HONESTY NOTE
  No public dataset maps these inputs to dietitian-assigned targets, so user_profiles.csv is
  SYNTHETIC: labels are simulated "dietitian-style" decisions generated from the revised
  Harris-Benedict equation (a DIFFERENT formula from the Mifflin-St Jeor used by the rule-based
  calculator), goal/BMI dependent adjustments, individual metabolic variation and noise.
  The pipeline is real ML (train/test split, cross-validation, metrics, Joblib), and you can
  replace the CSV with real data that has the same columns and re-run this script.
"""
import os
import sys
import json
import datetime as dt
import numpy as np
import pandas as pd
import joblib
import sklearn
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
                             f1_score, classification_report, confusion_matrix)

from utils.ml_predictor import encode_dataframe, FEATURES, MODEL_PATH, BASE_DIR
from utils.calorie_calculator import ACTIVITY, goal_code

DATA_DIR = os.path.join(BASE_DIR, "dataset")
PROFILE_CSV = os.path.join(DATA_DIR, "user_profiles.csv")
FOOD_CSV = os.path.join(DATA_DIR, "nutrition_dataset.csv")
SEED = 42


# ----------------------------------------------------------------------------- data generation
def generate_user_dataset(n=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    gender = rng.choice(["Male", "Female"], n, p=[0.5, 0.5])
    age = np.clip(rng.normal(32, 11, n), 15, 75).round().astype(int)
    height = np.where(gender == "Male", rng.normal(173, 7, n), rng.normal(160, 6.5, n)).round(1)
    bmi_draw = np.clip(rng.normal(24.5, 4.5, n), 16.5, 42)
    weight = (bmi_draw * (height / 100) ** 2).round(1)
    activity = rng.choice(list(ACTIVITY), n, p=[0.28, 0.27, 0.25, 0.14, 0.06])
    exercise = np.clip(rng.normal(35, 25, n), 0, 180).round(-1).astype(int)
    exercise = np.where(np.isin(activity, ["sedentary"]), np.minimum(exercise, 20), exercise)
    exercise = np.where(np.isin(activity, ["active", "extra"]), np.maximum(exercise, 40), exercise)
    meals = rng.choice([3, 4, 5], n, p=[0.4, 0.3, 0.3])
    bmi = weight / (height / 100) ** 2
    goal = np.where(bmi >= 26, rng.choice(["Weight Loss", "Maintain Weight", "Weight Gain"], n, p=[0.7, 0.28, 0.02]),
           np.where(bmi < 20, rng.choice(["Weight Gain", "Maintain Weight", "Weight Loss"], n, p=[0.6, 0.38, 0.02]),
                    rng.choice(["Maintain Weight", "Weight Loss", "Weight Gain"], n, p=[0.5, 0.25, 0.25])))

    # simulated dietitian decision: revised Harris-Benedict BMR x activity x individual variation
    hb = np.where(gender == "Male",
                  88.362 + 13.397 * weight + 4.799 * height - 5.677 * age,
                  447.593 + 9.247 * weight + 3.098 * height - 4.330 * age)
    af = np.array([ACTIVITY[a][0] for a in activity])
    maint = hb * af * rng.normal(1.0, 0.04, n) + 0.8 * exercise
    adj = np.ones(n)
    loss, gain = goal == "Weight Loss", goal == "Weight Gain"
    adj[loss] = 1 - np.clip(0.10 + 0.012 * (bmi[loss] - 22), 0.10, 0.22) + rng.normal(0, 0.015, loss.sum())
    adj[gain] = 1 + rng.uniform(0.08, 0.16, gain.sum())
    adj[~loss & ~gain] = rng.normal(1.0, 0.02, (~loss & ~gain).sum())
    floor = np.where(gender == "Male", 1500, 1200)
    kcal = np.maximum(maint * adj + rng.normal(0, 60, n), floor).round(-1)

    # simulated dietitian "food focus" label
    focus = np.full(n, "balanced", dtype=object)
    focus[loss] = np.where(rng.random(loss.sum()) < 0.85, "high_protein_lean", "balanced")
    focus[gain] = np.where((exercise[gain] >= 60) & (rng.random(gain.sum()) < 0.5), "high_carb_energy",
                           np.where(rng.random(gain.sum()) < 0.9, "calorie_dense_protein", "balanced"))
    mt = ~loss & ~gain
    focus[mt & (exercise >= 60) & (af >= 1.725)] = "high_carb_energy"
    focus[mt & (bmi >= 27)] = "high_protein_lean"
    flip = rng.random(n) < 0.06
    focus[flip] = rng.choice(["balanced", "high_protein_lean", "calorie_dense_protein", "high_carb_energy"], flip.sum())

    return pd.DataFrame({"age": age, "gender": gender, "height_cm": height, "weight_kg": weight,
                         "activity_level": activity, "exercise_min": exercise, "meals_per_day": meals,
                         "goal": goal, "target_calories": kcal.astype(int), "diet_focus": focus})


# ----------------------------------------------------------------------------- cleaning
def clean_profiles(df):
    report = {"rows_raw": int(len(df))}
    df = df.drop_duplicates().dropna()
    df = df[df["gender"].isin(["Male", "Female"]) & df["activity_level"].isin(ACTIVITY)]
    df = df[df["age"].between(14, 90) & df["height_cm"].between(120, 230) & df["weight_kg"].between(30, 250)
            & df["exercise_min"].between(0, 300) & df["meals_per_day"].between(3, 5)
            & df["target_calories"].between(1000, 5000)]
    report["rows_clean"] = int(len(df))
    report["rows_removed"] = report["rows_raw"] - report["rows_clean"]
    return df.reset_index(drop=True), report


def validate_foods():
    foods = pd.read_csv(FOOD_CSV).drop_duplicates("food_name").dropna(subset=["calories", "protein", "carbs", "fat"])
    est = foods["protein"] * 4 + foods["carbs"] * 4 + foods["fat"] * 9
    ok = (est - foods["calories"]).abs() / foods["calories"] < 0.30
    return {"foods": int(len(foods)), "calorie_consistent_pct": round(float(ok.mean() * 100), 1),
            "categories": foods["category"].value_counts().to_dict()}


# ----------------------------------------------------------------------------- training
def main(quiet=False, regenerate=False):
    log = (lambda *a: None) if quiet else print
    if regenerate or not os.path.exists(PROFILE_CSV):
        generate_user_dataset().to_csv(PROFILE_CSV, index=False)
        log("Generated", PROFILE_CSV)
    raw = pd.read_csv(PROFILE_CSV)
    df, clean_rep = clean_profiles(raw)
    food_rep = validate_foods()
    X = encode_dataframe(df)
    y_cal, y_focus = df["target_calories"].values, df["diet_focus"].values
    Xtr, Xte, ytr, yte, ftr, fte = train_test_split(X.values, y_cal, y_focus, test_size=0.2,
                                                    random_state=SEED, stratify=y_focus)
    metrics = {"dataset": {**clean_rep, "train_rows": len(Xtr), "test_rows": len(Xte), "features": FEATURES,
                           "foods": food_rep}}

    # ---- regression: calories
    reg_models = {
        "Baseline (mean)": DummyRegressor(),
        "Decision Tree": DecisionTreeRegressor(max_depth=12, min_samples_leaf=5, random_state=SEED),
        "Random Forest": RandomForestRegressor(n_estimators=150, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
    }
    reg_res = {}
    for name, m in reg_models.items():
        m.fit(Xtr, ytr)
        p = m.predict(Xte)
        reg_res[name] = {"MAE_kcal": round(mean_absolute_error(yte, p), 1),
                         "RMSE_kcal": round(float(np.sqrt(mean_squared_error(yte, p))), 1),
                         "R2": round(r2_score(yte, p), 4)}
    reg = reg_models["Random Forest"]
    reg_res["Random Forest"]["CV_R2_5fold"] = round(float(cross_val_score(
        RandomForestRegressor(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
        X.values, y_cal, cv=5, scoring="r2").mean()), 4)
    # how often is the observed value inside the 10-90 percentile band of the trees?
    trees = np.array([t.predict(Xte) for t in reg.estimators_])
    lo, hi = np.percentile(trees, 10, axis=0), np.percentile(trees, 90, axis=0)
    reg_res["Random Forest"]["range_coverage_10_90_pct"] = round(float(((yte >= lo) & (yte <= hi)).mean() * 100), 1)
    metrics["calorie_regression"] = reg_res

    # ---- classification: food focus
    clf_models = {
        "Baseline (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Decision Tree": DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=SEED),
        "Random Forest": RandomForestClassifier(n_estimators=150, min_samples_leaf=5, n_jobs=-1, random_state=SEED),
    }
    clf_res = {}
    for name, m in clf_models.items():
        m.fit(Xtr, ftr)
        p = m.predict(Xte)
        clf_res[name] = {"accuracy": round(accuracy_score(fte, p), 4),
                         "macro_F1": round(f1_score(fte, p, average="macro"), 4)}
    clf = clf_models["Random Forest"]
    clf_res["Random Forest"]["CV_accuracy_5fold"] = round(float(cross_val_score(
        RandomForestClassifier(n_estimators=150, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
        X.values, y_focus, cv=5, scoring="accuracy").mean()), 4)
    pred = clf.predict(Xte)
    labels = list(clf.classes_)
    clf_res["Random Forest"]["confusion_matrix"] = {"labels": labels,
                                                    "matrix": confusion_matrix(fte, pred, labels=labels).tolist()}
    metrics["focus_classification"] = clf_res
    metrics["feature_importance"] = {
        "calorie_model": dict(sorted(zip(FEATURES, map(lambda v: round(float(v), 4), reg.feature_importances_)),
                                     key=lambda kv: -kv[1])),
        "focus_model": dict(sorted(zip(FEATURES, map(lambda v: round(float(v), 4), clf.feature_importances_)),
                                   key=lambda kv: -kv[1]))}

    # ---- refit final models on ALL clean data, then save with Joblib
    reg.fit(X.values, y_cal)
    clf.fit(X.values, y_focus)
    bundle = {"calorie_model": reg, "focus_model": clf, "features": FEATURES, "sklearn_version": sklearn.__version__,
              "trained_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M"),
              "metrics": {"calorie_mae": reg_res["Random Forest"]["MAE_kcal"],
                          "calorie_r2": reg_res["Random Forest"]["R2"],
                          "focus_accuracy": clf_res["Random Forest"]["accuracy"],
                          "focus_f1": clf_res["Random Forest"]["macro_F1"],
                          "train_rows": len(df)}}
    joblib.dump(bundle, MODEL_PATH, compress=3)
    write_report(metrics, classification_report(fte, pred, zero_division=0), bundle["trained_at"])
    log(open(os.path.join(BASE_DIR, "model_report.txt")).read())
    log(f"Saved model to {MODEL_PATH}")
    return metrics


def write_report(m, clf_text, stamp):
    L = ["MODEL EVALUATION REPORT - Smart Nutrition & Diet Planner", "=" * 60, f"Generated: {stamp}", ""]
    d = m["dataset"]
    L += ["1. DATASET", f"   Raw rows: {d['rows_raw']}   After cleaning: {d['rows_clean']}   Removed: {d['rows_removed']}",
          f"   Train/test split: {d['train_rows']} / {d['test_rows']} (80/20, stratified on diet_focus)",
          f"   Features ({len(d['features'])}): {', '.join(d['features'])}",
          f"   Food table: {d['foods']['foods']} foods, {d['foods']['calorie_consistent_pct']}% have calories "
          "consistent with 4/4/9 kcal per g macros (+-30%)", "",
          "   NOTE: user_profiles.csv is SYNTHETIC (see train_model.py docstring). Scores show how well the",
          "   models learn the simulated dietitian policy, not real-world clinical accuracy.", "",
          "2. CALORIE REGRESSION (target: daily calories)"]
    for k, v in m["calorie_regression"].items():
        L.append(f"   {k:<26}" + "  ".join(f"{a}={b}" for a, b in v.items()))
    L += ["", "3. FOOD-FOCUS CLASSIFICATION (4 classes)"]
    for k, v in m["focus_classification"].items():
        L.append(f"   {k:<26}" + "  ".join(f"{a}={b}" for a, b in v.items() if a != "confusion_matrix"))
    cm = m["focus_classification"]["Random Forest"]["confusion_matrix"]
    L += ["", "   Confusion matrix (rows=true, cols=predicted), labels: " + ", ".join(cm["labels"])]
    L += ["   " + str(r) for r in cm["matrix"]]
    L += ["", "   Per-class report (Random Forest, test set):"] + ["   " + l for l in clf_text.splitlines()]
    L += ["", "4. FEATURE IMPORTANCE (Random Forest)"]
    for name, imp in m["feature_importance"].items():
        L.append(f"   {name}: " + ", ".join(f"{k}={v}" for k, v in list(imp.items())[:6]))
    L += ["", "5. HOW THE MODEL IS USED", "   - Calorie model -> shows a predicted calorie RANGE (10th-90th percentile of tree outputs).",
          "   - Focus model   -> picks food categories to emphasise when building the 7-day plan.",
          "   - Safe calorie targets (BMI/BMR/TDEE/macros) remain rule-based and are NOT produced by ML."]
    base = os.path.dirname(os.path.abspath(__file__))
    open(os.path.join(base, "model_report.txt"), "w").write("\n".join(L) + "\n")
    json.dump(m, open(os.path.join(base, "model_report.json"), "w"), indent=2)


if __name__ == "__main__":
    main(regenerate="--regenerate" in sys.argv)
