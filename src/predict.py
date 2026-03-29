
import joblib
import pandas as pd

MODEL_PATH = "models/final_model.pkl"

def load_model(model_path=MODEL_PATH):
    return joblib.load(model_path)

def predict_heart_disease(input_data, model_path=MODEL_PATH):
    model = load_model(model_path)

    if isinstance(input_data, dict):
        input_df = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        input_df = input_data.copy()
    else:
        raise ValueError("input_data must be a dictionary or a pandas DataFrame")

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    return {
        "prediction": int(prediction),
        "probability_of_heart_disease": float(probability)
    }

if __name__ == "__main__":
    sample_patient = {
        "age": 57,
        "sex": 1,
        "cp": 0,
        "trestbps": 150,
        "chol": 276,
        "fbs": 0,
        "restecg": 0,
        "thalach": 112,
        "exang": 1,
        "oldpeak": 0.6,
        "slope": 1,
        "ca": 1,
        "thal": 1
    }

    result = predict_heart_disease(sample_patient)
    print(result)
