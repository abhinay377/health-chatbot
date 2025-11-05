from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import csv, os, warnings
from flask_cors import CORS

warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)
CORS(app)

DATA_DIR = "Data"
MASTER_DIR = "MasterData"

def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)

training = safe_read_csv(os.path.join(DATA_DIR, "Training.csv"))
testing = safe_read_csv(os.path.join(DATA_DIR, "Testing.csv"))

symptom_columns = training.columns[:-1].tolist()
target_col = training.columns[-1]

X = training[symptom_columns]
y_raw = training[target_col].astype(str)

le = preprocessing.LabelEncoder()
le.fit(y_raw)
y = le.transform(y_raw)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)

# optional: cross-val score
try:
    scores = cross_val_score(clf, x_test, y_test, cv=3)
    print("DecisionTree CV mean:", scores.mean())
except Exception:
    pass

svc = SVC()
svc.fit(x_train, y_train)
try:
    print("SVM score:", svc.score(x_test, y_test))
except Exception:
    pass

reduced_data = training.groupby(training[target_col]).max()

severity_dict = {}
description_dict = {}
precaution_dict = {}

def load_master_csvs():
    sev_path = os.path.join(MASTER_DIR, "symptom_severity.csv")
    desc_path = os.path.join(MASTER_DIR, "symptom_Description.csv")
    prec_path = os.path.join(MASTER_DIR, "symptom_precaution.csv")

    if os.path.exists(sev_path):
        with open(sev_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0].strip():
                    try:
                        severity_dict[row[0].strip()] = int(row[1])
                    except Exception:
                        severity_dict[row[0].strip()] = 1

    if os.path.exists(desc_path):
        with open(desc_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0].strip():
                    description_dict[row[0].strip()] = row[1].strip()

    if os.path.exists(prec_path):
        with open(prec_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 5 and row[0].strip():
                    precaution_dict[row[0].strip()] = [r.strip() for r in row[1:5]]

load_master_csvs()

def sec_predict(symptoms_exp):
    df = training.copy()
    X2 = df.iloc[:, :-1]
    y2_raw = df[target_col]
    y2_enc = le.transform(y2_raw)
    Xtr, Xte, ytr, yte = train_test_split(X2, y2_enc, test_size=0.3, random_state=20)
    rf = DecisionTreeClassifier()
    rf.fit(Xtr, ytr)
    # build input vector according to training columns
    input_vector = np.zeros(len(symptom_columns), dtype=int)
    for s in symptoms_exp:
        if s in symptom_columns:
            input_vector[symptom_columns.index(s)] = 1
    pred_enc = rf.predict([input_vector])[0]
    return le.inverse_transform([pred_enc])[0]

def calc_condition(symptoms_exp, days):
    total = 0
    for item in symptoms_exp:
        total += severity_dict.get(item, 1)
    denom = max(len(symptoms_exp) + 1, 1)
    score = (total * days) / denom
    if score > 13:
        return "You should consult a doctor."
    else:
        return "It might not be severe but take precautions."

@app.route("/diseases", methods=["GET"])
def get_diseases():
    diseases = sorted(training[target_col].unique().tolist())
    return jsonify(diseases)

@app.route("/symptoms/<disease>", methods=["GET"])
def get_symptoms(disease):
    try:
        rows = training[training[target_col] == disease]
        if rows.empty:
            return jsonify([])
        symptoms = set()
        for _, r in rows.iterrows():
            for col in symptom_columns:
                try:
                    if int(r[col]) == 1:
                        symptoms.add(col)
                except Exception:
                    pass
        return jsonify(sorted(list(symptoms)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()
        # payload: { symptoms: [...], days: int, disease(optional): "..." }
        symptoms = payload.get("symptoms", [])
        days = int(payload.get("days", 0))
        if not isinstance(symptoms, list):
            return jsonify({"success": False, "message": "symptoms must be a list"}), 400
        input_vector = np.zeros(len(symptom_columns), dtype=int)
        found = False
        for s in symptoms:
            s = s.strip()
            if s in symptom_columns:
                input_vector[symptom_columns.index(s)] = 1
                found = True
        if not found:
            return jsonify({"success": False, "message": "No matching symptoms found"}), 400
        pred_enc = clf.predict([input_vector])[0]
        disease_pred = le.inverse_transform([pred_enc])[0]
        # secondary prediction (trained on string labels) for cross-check
        try:
            sec = sec_predict(symptoms)
        except Exception:
            sec = None
        description = description_dict.get(disease_pred, "Description not available.")
        precautions = precaution_dict.get(disease_pred, ["No precautions available."])
        condition_msg = calc_condition(symptoms, days)
        return jsonify({
            "success": True,
            "disease": disease_pred,
            "secondary": sec,
            "description": description,
            "precautions": precautions,
            "condition": condition_msg
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500
    
    

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
    