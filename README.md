# Smart Nutrition & Diet Planner (Flask + Machine Learning)

A college-level web application that calculates BMI/BMR/TDEE and macro targets, uses a **real scikit-learn
Random Forest model** to suggest a calorie range and food focus, builds a personalised **7-day Indian diet plan**,
tracks daily food intake, gamifies healthy habits with **fitness challenges**, shows **Chart.js dashboards**
and exports a **PDF report**. Responsive (desktop + mobile) with dark/light mode.

> **Disclaimer:** all values are estimates from standard formulas and a statistical model. They are not a medical
> diagnosis or treatment advice. Consult a doctor or registered dietitian before major diet changes.

## 1. Quick start

```bash
cd nutrition_diet_planner
python -m venv venv
source venv/bin/activate            # Windows: venv\Scripts\activate
pip install -r requirements.txt
python train_model.py               # optional: re-trains the model, writes model.pkl + model_report.txt/json
python app.py                       # open http://127.0.0.1:5000
```

* Python 3.9+ is required. An internet connection is needed in the browser for Bootstrap, Chart.js and Google Fonts (CDN).
* `database.db` (SQLite) is **created automatically** on first run, including the demo account.
* If `model.pkl` is missing or was saved by an incompatible scikit-learn version, the app retrains it automatically.

**Demo login:** username `demo` · password `Demo@1234` (comes with a profile, weight history, food logs and challenge history).
Or register a new account (password: 8+ chars, upper + lower case, number, special character).

## 2. Features

| Module | What it does |
|---|---|
| Auth | Register/login/logout, hashed passwords (Werkzeug), server-side validation, session cookies, CSRF tokens on every POST |
| Assessment | Name, age, gender, height, weight, activity, diet preference, goal, allergies, restrictions, exercise, meals/day |
| Calculator (rule-based) | BMI (WHO), BMR (Mifflin-St Jeor), TDEE, calorie target, protein/carbs/fat, fibre, water. Safety limits: max 500 kcal (20 %) deficit, never below 1200/1500 kcal or BMR, no weight-loss target if underweight |
| ML prediction | Calorie **range** (Random Forest Regressor) + **food focus** category (Random Forest Classifier), shown separately from the rules |
| Diet plan | 7 days × up to 5 meals; food, quantity, kcal, protein, carbs, fat; alternatives with one-click swap; regenerate; previous plans |
| Food tracker | Log foods from the dataset (or enter manually), per-day totals vs target, remaining calories, delete entries, browse past dates |
| Challenges | 7 daily challenges, points, streak, weekly progress, 10 badges, 14-day history |
| Dashboard | Profile, calories, BMI/BMR/TDEE, macros, water, challenges; pie, bar, line and progress charts (Chart.js) |
| History & reports | Assessments, saved plans, intake, challenges, weight log; downloadable PDF (ReportLab) |

## 3. Machine learning (what is real and what is rule-based)

| Part | Type | File |
|---|---|---|
| BMI, BMR, TDEE, macros, water, safety limits | **Rule-based** formulas | `utils/calorie_calculator.py` |
| Calorie range prediction | **ML** – `RandomForestRegressor` (range = 10th–90th percentile of tree predictions) | `train_model.py`, `utils/ml_predictor.py` |
| Food focus (high-protein lean / balanced / calorie-dense protein / carb-rich energy) | **ML** – `RandomForestClassifier` | same |
| Portioning, allergy filtering, meal assembly | Algorithmic (not ML) | `utils/diet_recommender.py` |

The pipeline in `train_model.py`: load data → clean/validate → feature engineering (`encode_dataframe`) →
80/20 stratified split → compare baseline, Decision Tree and Random Forest → 5-fold cross-validation →
refit on all data → save with Joblib to `model.pkl` → write `model_report.txt` / `model_report.json`.
Flask loads `model.pkl` and runs inference on each user's real profile (no random numbers).

**Results on the held-out test set** (see `model_report.txt`): calorie MAE ≈ 107 kcal, R² ≈ 0.92;
food-focus accuracy ≈ 87 %, macro-F1 ≈ 0.81 (baselines: R² ≈ 0, accuracy 39 %).

### Honest note about the data
No public dataset maps these user inputs to dietitian-assigned targets, so `dataset/user_profiles.csv` is
**synthetic**: labels were simulated from the *revised Harris-Benedict* equation (a different formula from the
Mifflin-St Jeor used by the rule-based calculator), goal/BMI-dependent adjustments, individual variation and noise.
The scores therefore show the model learning that simulated policy, **not** clinical accuracy. To use real data,
replace the CSV (same columns) and run `python train_model.py`. Regenerate the synthetic file with
`python train_model.py --regenerate`.
`dataset/nutrition_dataset.csv` holds 72 common Indian foods with typical per-serving values
(approximate averages, e.g. from IFCT/USDA-style tables; real recipes vary).

## 4. Project structure

```
nutrition_diet_planner/
├── app.py                      Flask routes, auth, CSRF, API endpoints
├── train_model.py              dataset generation, cleaning, training, evaluation, Joblib export
├── model.pkl                   trained models (Joblib)
├── model_report.txt / .json    evaluation report
├── requirements.txt
├── database.db                 created automatically on first run
├── dataset/
│   ├── nutrition_dataset.csv   food composition table
│   └── user_profiles.csv       ML training data (synthetic)
├── templates/                  login, register, dashboard, profile, calorie_calculator, diet_plan,
│                               food_tracker, challenges, history (+ base.html, error.html)
├── static/{css/style.css, js/script.js, images/}
└── utils/
    ├── calorie_calculator.py   rule-based formulas
    ├── ml_predictor.py         features + model loading + prediction
    ├── diet_recommender.py     filters, meal builder, portion optimiser, swaps
    ├── database.py             SQLite schema + queries + demo seed
    └── pdf_report.py           ReportLab report
```

## 5. Database tables
`users`, `profiles`, `assessments` (history of every calculation), `weight_logs`, `diet_plans` (JSON),
`food_logs`, `challenge_logs`.

## 6. Notes & limitations
* Allergy/restriction filtering is keyword-based on the food table; always read food labels.
* Eggs are classed as non-vegetarian; Vegan excludes dairy and eggs; ghee/dairy tags are approximate.
* Set `SECRET_KEY` in the environment for production, and run behind a real WSGI server (e.g. `gunicorn app:app`).
  `python app.py` uses Flask's development server. `FLASK_DEBUG=1` enables debug mode.
* Possible extensions: real food-composition datasets (IFCT), more foods, barcode scanning, email password reset.
