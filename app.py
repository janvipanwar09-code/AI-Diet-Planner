from flask import Flask, render_template, request
import numpy as np
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)


# -----------------------------
# AI CALORIE PREDICTION MODEL
# -----------------------------

# Sample training data for demonstration
# Features: age, height(cm), weight(kg), activity level
training_data = np.array([
    [20, 155, 50, 1],
    [22, 160, 55, 2],
    [25, 165, 60, 2],
    [28, 170, 70, 3],
    [30, 175, 75, 3],
    [35, 160, 65, 2],
    [40, 165, 80, 1],
    [24, 168, 62, 4],
    [27, 172, 68, 4],
    [32, 158, 72, 2],
    [21, 162, 58, 3],
    [29, 180, 85, 4]
])

# Approximate daily calorie targets for demonstration
calorie_targets = np.array([
    1800, 1950, 2100, 2400,
    2500, 2000, 1900, 2500,
    2300, 2000, 2200, 2700
])

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(training_data, calorie_targets)


# -----------------------------
# DIET RECOMMENDATION FUNCTION
# -----------------------------

def generate_diet(goal, preference):

    if preference == "veg":
        meals = {
            "breakfast": "Oats with milk and fruit",
            "lunch": "Dal, roti, mixed vegetables and curd",
            "snack": "Fruit with a handful of nuts",
            "dinner": "Paneer/tofu with vegetables and roti"
        }
    else:
        meals = {
            "breakfast": "Eggs with whole-grain toast and fruit",
            "lunch": "Chicken, rice/roti and mixed vegetables",
            "snack": "Fruit with yogurt",
            "dinner": "Grilled chicken/paneer with vegetables"
        }

    if goal == "weight_loss":
        note = "Focus on balanced portions and regular physical activity."
    elif goal == "weight_gain":
        note = "Include adequate calories and protein across meals."
    else:
        note = "Maintain balanced meals and consistent eating habits."

    return meals, note


# -----------------------------
# HOME PAGE
# -----------------------------

@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    if request.method == "POST":

        try:
            age = float(request.form["age"])
            height = float(request.form["height"])
            weight = float(request.form["weight"])
            activity = int(request.form["activity"])
            goal = request.form["goal"]
            preference = request.form["preference"]

            # Basic validation
            if age <= 0 or height <= 0 or weight <= 0:
                raise ValueError("Invalid values")

            # Predict calorie requirement
            features = np.array([
                [age, height, weight, activity]
            ])

            predicted_calories = int(model.predict(features)[0])

            # Adjust target according to goal
            if goal == "weight_loss":
                target_calories = predicted_calories - 300
            elif goal == "weight_gain":
                target_calories = predicted_calories + 300
            else:
                target_calories = predicted_calories

            meals, note = generate_diet(
                goal,
                preference
            )

            result = {
                "calories": max(target_calories, 1200),
                "goal": goal.replace("_", " ").title(),
                "preference": preference.title(),
                "meals": meals,
                "note": note
            }

        except (ValueError, KeyError):
            result = {
                "error": "Please enter valid information."
            }

    return render_template(
        "index.html",
        result=result
    )


# -----------------------------
# RUN APPLICATION
# -----------------------------

if __name__ == "__main__":
    app.run(debug=True)
