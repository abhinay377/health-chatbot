from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import csv, os, warnings
from difflib import get_close_matches
from flask_cors import CORS

warnings.filterwarnings("ignore", category=DeprecationWarning)

app = Flask(__name__)
CORS(app)

DATA_DIR = "Data"
MASTER_DIR = "MasterData"

# -------------------------------
# Utility to safely read CSV
# -------------------------------
def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path)

# -------------------------------
# Load main datasets
# -------------------------------
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

# -------------------------------
# Load master data dictionaries
# -------------------------------
severity_dict = {}
description_dict = {}
precaution_dict = {}
medicine_dict = {}

def load_master_csvs():
    sev_path = os.path.join(MASTER_DIR, "symptom_severity.csv")
    desc_path = os.path.join(MASTER_DIR, "symptom_Description.csv")
    prec_path = os.path.join(MASTER_DIR, "symptom_precaution.csv")
    med_path = os.path.join(MASTER_DIR, "medicine_dataset.csv")

    # Symptom severity
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

    # Disease description
    if os.path.exists(desc_path):
        with open(desc_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0].strip():
                    description_dict[row[0].strip()] = row[1].strip()

    # Precautions
    if os.path.exists(prec_path):
        with open(prec_path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 5 and row[0].strip():
                    precaution_dict[row[0].strip()] = [r.strip() for r in row[1:5]]

    # Medicines (case-insensitive + clean)
    if os.path.exists(med_path):
        med_df = pd.read_csv(med_path)
        med_df.columns = [c.strip() for c in med_df.columns]
        med_df.fillna("", inplace=True)
        for _, row in med_df.iterrows():
            disease = str(row.get("Disease", "")).strip().lower()
            if not disease:
                continue
            med_info = {
                "Medicine_Name": str(row.get("Medicine_Name", "")),
                "Category": str(row.get("Category", "")),
                "Purpose": str(row.get("Purpose", "")),
                "Dosage": str(row.get("Dosage", "")),
                "Side_Effects": str(row.get("Side_Effects", "")),
            }
            if disease not in medicine_dict:
                medicine_dict[disease] = []
            medicine_dict[disease].append(med_info)
        print(f"✅ Loaded medicines for {len(medicine_dict)} diseases.")
    else:
        print("⚠️ Medicine dataset not found at:", med_path)

load_master_csvs()

# -------------------------------
# Helper functions
# -------------------------------
def sec_predict(symptoms_exp):
    df = training.copy()
    X2 = df.iloc[:, :-1]
    y2_raw = df[target_col]
    y2_enc = le.transform(y2_raw)
    Xtr, Xte, ytr, yte = train_test_split(X2, y2_enc, test_size=0.3, random_state=20)
    rf = DecisionTreeClassifier()
    rf.fit(Xtr, ytr)
    input_vector = np.zeros(len(symptom_columns), dtype=int)
    for s in symptoms_exp:
        if s in symptom_columns:
            input_vector[symptom_columns.index(s)] = 1
    pred_enc = rf.predict([input_vector])[0]
    return le.inverse_transform([pred_enc])[0]

def calc_condition(symptoms_exp, days):
    total = sum(severity_dict.get(s, 1) for s in symptoms_exp)
    denom = max(len(symptoms_exp) + 1, 1)
    score = (total * days) / denom
    if score > 13:
        return "You should consult a doctor."
    else:
        return "It might not be severe but take precautions."

# -------------------------------
# Routes
# -------------------------------
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
                if int(r[col]) == 1:
                    symptoms.add(col)
        return jsonify(sorted(list(symptoms)))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json()
        symptoms = payload.get("symptoms", [])
        days = int(payload.get("days", 0))
        if not isinstance(symptoms, list):
            return jsonify({"success": False, "message": "symptoms must be a list"}), 400

        input_vector = np.zeros(len(symptom_columns), dtype=int)
        for s in symptoms:
            s = s.strip()
            if s in symptom_columns:
                input_vector[symptom_columns.index(s)] = 1

        pred_enc = clf.predict([input_vector])[0]
        disease_pred = le.inverse_transform([pred_enc])[0]

        # Secondary prediction
        try:
            sec = sec_predict(symptoms)
        except Exception:
            sec = None

        description = description_dict.get(disease_pred, "Description not available.")
        precautions = precaution_dict.get(disease_pred, ["No precautions available."])

        # ✅ Fixed Medicine Recommendation (case-insensitive + fuzzy match)
        disease_key = disease_pred.strip().lower()
        all_meds = {k.lower().strip(): v for k, v in medicine_dict.items()}

        medicines = []
        if disease_key in all_meds:
            medicines = all_meds[disease_key]
        else:
            match = get_close_matches(disease_key, list(all_meds.keys()), n=1, cutoff=0.5)
            if match:
                medicines = all_meds[match[0]]
                print(f"⚠️ Fuzzy match for medicine: {match[0]} (instead of {disease_key})")
            else:
                medicines = [{
                    "Medicine_Name": "No medicine found",
                    "Category": "",
                    "Purpose": "",
                    "Dosage": "",
                    "Side_Effects": ""
                }]

        if not medicines or all(not m.get("Medicine_Name") for m in medicines):
            medicines = [{
                "Medicine_Name": "No medicine found",
                "Category": "",
                "Purpose": "",
                "Dosage": "",
                "Side_Effects": ""
            }]

        print(f"💊 Predicted: {disease_pred} | Medicines found: {len(medicines)}")

        condition_msg = calc_condition(symptoms, days)

        return jsonify({
            "success": True,
            "disease": disease_pred,
            "secondary": sec,
            "description": description,
            "precautions": precautions,
            "medicines": medicines,
            "condition": condition_msg
        })
    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
