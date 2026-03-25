from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import pickle
import io
import matplotlib.pyplot as plt
import datetime

# Load model and preprocessing 
with open("model.pkl", "rb") as f:
    model_bundle = pickle.load(f)

model = model_bundle["model"]
scaler = model_bundle["scaler"]
features = model_bundle["features"]

exclude_from_scaling = ['Age', 'Income', 'Number of Children']

# categorical mappings to avoid LabelEncoder issues
label_maps = {
    "Marital Status": {"Single": 0, "Married": 1, "Divorced": 2, "Widowed": 3},
    "Education Level": {"High School": 0, "Bachelor's Degree": 1, "Master's Degree": 2, "Doctorate": 3},
    "Smoking Status": {"Smoker": 1, "Non-smoker": 0},
    "Physical Activity Level": {"Sedentary": 0, "Moderate": 1, "Active": 2},
    "Employment Status": {"Unemployed": 0, "Employed": 1},
    "Alcohol Consumption": {"Low": 0, "Moderate": 1, "High": 2},
    "Dietary Habits": {"Healthy": 0, "Moderate": 1, "Unhealthy": 2},
    "Sleep Patterns": {"Good": 0, "Fair": 1, "Poor": 2},
    "History of Mental Illness": {"No": 0, "Yes": 1},
    "History of Substance Abuse": {"No": 0, "Yes": 1},
    "Family History of Depression": {"No": 0, "Yes": 1},
}

app = Flask(__name__)

mood_log = []

@app.route('/')
def home():
    return render_template("landingpage.html")

@app.route('/predict', methods=['POST'])
def predict():
    form_data = request.form.to_dict()
    input_df = pd.DataFrame([form_data])

    input_df["Age"] = input_df["Age"].astype(int)
    input_df["Number of Children"] = input_df["Number of Children"].astype(int)
    input_df["Income"] = input_df["Income"].astype(float)

    for col, mapping in label_maps.items():
        if col in input_df:
            input_df[col] = input_df[col].map(mapping)

    for col in input_df.columns:
        if col not in exclude_from_scaling and col in scaler.feature_names_in_:
            input_df[col] = scaler.transform(input_df[[col]])

    input_df = input_df[features]
    prediction = model.predict(input_df)[0]
    result = "Bipolar Disorder" if prediction == "Yes" else "No Bipolar Disorder"
    return f"<h3>Prediction: {result}</h3>"

@app.route('/risk_score', methods=['POST'])
def risk_score():
    data = request.json
    score = 0
    risk_factors = [
        ("Smoking Status", "Smoker"),
        ("Alcohol Consumption", "High"),
        ("Dietary Habits", "Unhealthy"),
        ("Sleep Patterns", "Poor"),
        ("History of Mental Illness", "Yes"),
        ("History of Substance Abuse", "Yes"),
        ("Family History of Depression", "Yes")
    ]
    for field, risky_value in risk_factors:
        if data.get(field) == risky_value:
            score += 1
    return jsonify({'risk_score': score})

@app.route('/genetic_risk', methods=['POST'])
def genetic_risk():
    data = request.json
    mental_illness = data.get("History of Mental Illness") == "Yes"
    family_history = data.get("Family History of Depression") == "Yes"
    risk = 0
    if mental_illness: risk += 0.5
    if family_history: risk += 0.5
    return jsonify({'genetic_risk_score': risk})

@app.route('/mood_checkin', methods=['POST'])
def mood_checkin():
    mood = request.json.get("mood")
    date = datetime.date.today()
    mood_log.append((date, mood))
    return jsonify({'message': 'Mood recorded'})

@app.route('/mood_progress')
def mood_progress():
    dates = [entry[0] for entry in mood_log]
    moods = [entry[1] for entry in mood_log]
    fig, ax = plt.subplots()
    ax.plot(dates, moods, marker='o')
    ax.set_title("Mood Progress Over Time")
    ax.set_ylabel("Mood (1-10)")
    ax.set_xlabel("Date")
    plt.xticks(rotation=45)
    fig.tight_layout()
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    return send_file(img, mimetype='image/png')

@app.route('/tips', methods=['POST'])
def tips():
    risk_level = request.json.get("risk_level", "Low")
    tips_dict = {
        "Low": ["Keep a regular sleep schedule", "Stay active", "Maintain social contact"],
        "Medium": ["Reduce stress", "Eat healthy", "Seek therapy if needed"],
        "High": ["Consult a mental health professional", "Avoid alcohol and drugs", "Develop a strong support system"]
    }
    return jsonify({'tips': tips_dict.get(risk_level, [])})

if __name__ == '__main__':
    app.run(debug=True)
