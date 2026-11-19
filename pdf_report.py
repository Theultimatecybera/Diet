"""Downloadable PDF nutrition report (ReportLab)."""
from io import BytesIO
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

GREEN, LIGHT, ORANGE = colors.HexColor("#1f6f4a"), colors.HexColor("#e8f5ee"), colors.HexColor("#f28c28")


def _table(data, widths=None, header=True):
    t = Table(data, colWidths=widths, repeatRows=1 if header else 0)
    style = [("FONTSIZE", (0, 0), (-1, -1), 8.5), ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b9d8c6")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style += [("BACKGROUND", (0, 0), (-1, 0), GREEN), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                  ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold")]
    t.setStyle(TableStyle(style))
    return t


def build_report(user, profile, assess, ml, plan, food_series, weights, ch_week, ch_stats, disclaimer):
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm, topMargin=1.5 * cm,
                            bottomMargin=1.5 * cm, title="Nutrition Report", author="Smart Nutrition & Diet Planner")
    ss = getSampleStyleSheet()
    h1 = ParagraphStyle("h1", parent=ss["Title"], textColor=GREEN, fontSize=22, alignment=0)
    h2 = ParagraphStyle("h2", parent=ss["Heading2"], textColor=GREEN, spaceBefore=10, spaceAfter=4)
    body = ParagraphStyle("b", parent=ss["BodyText"], fontSize=9, leading=12)
    small = ParagraphStyle("s", parent=body, fontSize=7.5, textColor=colors.HexColor("#555555"))
    S = []
    S += [Paragraph("Smart Nutrition &amp; Diet Planner", h1),
          Paragraph(f"Personal nutrition report for <b>{profile['full_name']}</b> ({user['username']}) &nbsp;|&nbsp; "
                    f"Generated {datetime.now():%d %b %Y, %H:%M}", body), Spacer(1, 6)]
    S += [Paragraph("1. Profile", h2), _table([
        ["Age", "Gender", "Height", "Weight", "Activity", "Food preference", "Goal"],
        [profile["age"], profile["gender"], f"{profile['height_cm']:g} cm", f"{profile['weight_kg']:g} kg",
         profile["activity_level"].title(), profile["food_pref"], profile["goal"]]])]
    S += [Paragraph(f"Allergies: {profile['allergies'] or 'None'} &nbsp;|&nbsp; Restrictions: {profile['restrictions'] or 'None'} "
                    f"&nbsp;|&nbsp; Exercise: {profile['exercise_min']} min/day &nbsp;|&nbsp; Meals/day: {profile['meals_per_day']}", body)]
    a = assess
    S += [Paragraph("2. Rule-based results (Mifflin-St Jeor)", h2), _table([
        ["BMI", "BMR", "TDEE", "Daily calories", "Protein", "Carbs", "Fat", "Water"],
        [f"{a['bmi']} ({a['bmi_category']})", f"{a['bmr']} kcal", f"{a['tdee']} kcal", f"{a['target_calories']} kcal",
         f"{a['protein_g']} g", f"{a['carbs_g']} g", f"{a['fat_g']} g", f"{a['water_l']} L"]])]
    for w in a["warnings"] + a["notes"]:
        S.append(Paragraph("&bull; " + w, body))
    S += [Paragraph("3. Machine-learning prediction (Random Forest)", h2),
          Paragraph(f"Predicted calorie range: <b>{ml['calories_low']} - {ml['calories_high']} kcal</b> "
                    f"(central estimate {ml['calories_pred']}). Recommended food focus: <b>{ml['focus_label']}</b> "
                    f"(confidence {ml['confidence']}%). {ml['focus_text']}", body)]
    if plan:
        S += [Paragraph("4. Active 7-day diet plan", h2)]
        for w in plan["plan"]["warnings"]:
            S.append(Paragraph("&bull; " + w, small))
        rows = [["Day", "Calories", "Protein (g)", "Carbs (g)", "Fat (g)"]]
        for d in plan["plan"]["days"]:
            t = d["totals"]
            rows.append([d["day"], t["calories"], t["protein"], t["carbs"], t["fat"]])
        S.append(_table(rows))
        S.append(PageBreak())
        for d in plan["plan"]["days"]:
            S.append(Paragraph(f"{d['day']} - {d['totals']['calories']:.0f} kcal", h2))
            rows = [["Meal", "Food", "Quantity", "kcal", "P", "C", "F"]]
            for m in d["meals"]:
                for i, it in enumerate(m["items"]):
                    rows.append([m["name"] if i == 0 else "", it["food"], it["qty_text"], it["calories"], it["protein"],
                                 it["carbs"], it["fat"]])
            S.append(_table(rows, [2.6 * cm, 5 * cm, 5.4 * cm, 1.3 * cm, 1.2 * cm, 1.2 * cm, 1.2 * cm]))
    S += [PageBreak(), Paragraph("5. Calorie intake - last 7 days", h2),
          _table([["Date", "Calories", "Target", "Protein", "Carbs", "Fat"]] +
                 [[r["date"], r["calories"], a["target_calories"], r["protein"], r["carbs"], r["fat"]] for r in food_series])]
    S.append(Paragraph("6. Weight progress", h2))
    S.append(_table([["Date", "Weight (kg)"]] + [[w["log_date"], w["weight_kg"]] for w in weights[-12:]]) if weights
             else Paragraph("No weight entries yet.", body))
    S += [Paragraph("7. Fitness challenges", h2),
          Paragraph(f"Total points: <b>{ch_stats['points']}</b> &nbsp;|&nbsp; Current streak: <b>{ch_stats['streak']} days</b> "
                    f"&nbsp;|&nbsp; Perfect days: <b>{ch_stats['perfect_days']}</b>", body),
          _table([["Date", "Completed", "Points", "Completion"]] +
                 [[r["date"], f"{r['done']} / 7", r["points"], f"{r['pct']}%"] for r in ch_week]),
          Spacer(1, 10), Paragraph("<b>Important:</b> " + disclaimer, small)]
    doc.build(S)
    buf.seek(0)
    return buf
