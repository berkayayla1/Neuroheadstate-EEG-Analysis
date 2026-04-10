from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import joblib
import pandas as pd
import numpy as np
from sklearn.semi_supervised import LabelSpreading
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import io
import time
from datetime import datetime
from pydantic import BaseModel

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

model_path = 'model.pkl'
csv_path = 'train.csv'

system_memory = {
    "history": [],
    "stats": None,
    "active_model_name": None
}

model_factory = {
    "LabelSpreading": LabelSpreading(kernel='knn', n_neighbors=7),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "DecisionTree": DecisionTreeClassifier(random_state=42),
    "GaussianNB": GaussianNB()
}

class UpdateItem(BaseModel):
    item_id: int
    new_result: str
# Loading the model
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

model = load_model()

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    system_memory["history"] = []
    system_memory["stats"] = None
    system_memory["active_model_name"] = None
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict_api")
def predict_api(
    AF3: float = Form(...), F7: float = Form(...), F3: float = Form(...),
    FC5: float = Form(...), T7: float = Form(...), P7: float = Form(...),
    O1: float = Form(...), O2: float = Form(...), P8: float = Form(...),
    T8: float = Form(...), FC6: float = Form(...), F4: float = Form(...),
    F8: float = Form(...), AF4: float = Form(...)
):
    global model
    
    if system_memory["active_model_name"] is None:
        return {"status": "error", "message": "⚠️ No Active Model! Please go to 'Optimize Model' section and select a model first."}

    if model is None: 
        return {"status": "error", "message": "Model file missing!"}

    start_time = time.time()
    input_data = pd.DataFrame([[AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4]],
                              columns=['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4'])
    # Inference inside the API
    prediction = model.predict(input_data)
    result_val = int(prediction[0])
    result_text = "EYE CLOSED (1)" if result_val == 1 else "EYE OPEN (0)"
    
    end_time = time.time()
    time_taken = round(end_time - start_time, 4)

    mean_voltage = input_data.mean(axis=1)[0]
    neuro_state = ""
    state_desc = ""
    signal_warning = None

    if result_val == 1: 
        neuro_state = "RESTING STATE (Low Arousal)"
        state_desc = "High Alpha Waves (8-12 Hz) likely present. The brain is in a relaxed state."
    else: 
        neuro_state = "ACTIVE STATE (High Arousal)"
        state_desc = "High Beta Waves (>13 Hz) likely present. Alpha blocking occurred due to visual focus."

    if mean_voltage > 4900:
        signal_warning = "ABNORMAL SIGNAL: Potential Artifact or High Noise Detected."
    elif mean_voltage < 1000:
        signal_warning = "POOR SIGNAL: Sensors might be disconnected."

    timestamp = datetime.now().strftime("%H:%M:%S")
    
    # --- REFERANSA TEK EKLEME: ID ---
    unique_id = int(time.time() * 1000)
    new_record = {
        "id": unique_id,
        "time": timestamp, 
        "result": result_text, 
        "model": system_memory["active_model_name"],
        "sensors": {"AF3": AF3, "F7": F7, "F3": F3, "FC5": FC5, "T7": T7, "P7": P7, "O1": O1, "O2": O2, "P8": P8, "T8": T8, "FC6": FC6, "F4": F4, "F8": F8, "AF4": AF4}
    }
    system_memory["history"].insert(0, new_record)
    system_memory["history"] = system_memory["history"][:5]

    return {
        "status": "success", 
        "result": result_text, 
        "time_taken": f"{time_taken}s",
        "model_used": system_memory["active_model_name"],
        "history": system_memory["history"],
        "mean_voltage": f"{mean_voltage:.2f} µV",
        "neuro_state": neuro_state,
        "state_desc": state_desc,
        "signal_warning": signal_warning
    }

# --- REFERANSA TEK EKLEME: CRUD ENDPOINTS ---
@app.delete("/history_api/{item_id}")
def delete_history_item(item_id: int):
    system_memory["history"] = [item for item in system_memory["history"] if item["id"] != item_id]
    return {"status": "success"}

@app.put("/history_api")
def update_history_item(item: UpdateItem):
    for record in system_memory["history"]:
        if record["id"] == item.item_id:
            record["result"] = item.new_result
            break
    return {"status": "success"}
# --------------------------------------------

@app.post("/diagnostics_api")
def run_diagnostics():
    global model
    current = system_memory["active_model_name"] if system_memory["active_model_name"] else "Not Selected ❌"
    return {"api_status": "Online (Active)", "model_loaded": "Yes" if model else "No", "current_model": current, "dataset_found": "Yes" if os.path.exists(csv_path) else "No", "sensor_channels": 14}

@app.post("/optimize_model_api")
def optimize_model(model_name: str = Form(...)):
    global model
    if model_name not in model_factory: return {"status": "Error", "message": "Invalid Model Selected"}
    try:
        df = pd.read_csv(csv_path)
        X = df.drop('eyeDetection', axis=1)
        y = df['eyeDetection']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        # Dynamic model selection and retraining
        selected_model = model_factory[model_name]
        selected_model.fit(X_train, y_train)
        # Evaluating and saving the new model
        y_pred = selected_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='binary')
        rec = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        conf = confusion_matrix(y_test, y_pred).ravel().tolist()
        system_memory["stats"] = {"accuracy": round(acc, 4), "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "confusion": conf}
        # Updating system memory with new stats
        system_memory["active_model_name"] = model_name
        joblib.dump(selected_model, model_path)
        model = selected_model
        return {"status": "Success", "new_accuracy": f"{acc*100:.2f}%", "samples_used": len(df)}
    except Exception as e: return {"status": "Error", "message": str(e)}

@app.post("/predict_batch_api")
async def predict_batch(file: UploadFile = File(...)):
    global model
    if system_memory["active_model_name"] is None: return JSONResponse({"status": "error", "message": "No Active Model! Please select one first."}, status_code=400)
    if model is None: return JSONResponse({"status": "error", "message": "Model not found!"}, status_code=400)
    try:
        contents = await file.read()
        try: df = pd.read_csv(io.BytesIO(contents))
        except: return JSONResponse({"status": "error", "message": "Invalid CSV file."}, status_code=400)
        if df.empty: return JSONResponse({"status": "error", "message": "File is empty!"}, status_code=400)
        if 'id' in df.columns: df = df.drop('id', axis=1)
        if 'eyeDetection' in df.columns: df = df.drop('eyeDetection', axis=1)
        feature_columns = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        missing = [c for c in feature_columns if c not in df.columns]
        if missing: return JSONResponse({"status": "error", "message": f"Missing columns: {missing}"}, status_code=400)
        X_batch = df[feature_columns]
        if len(X_batch) == 0: return JSONResponse({"status": "error", "message": "No data rows found."}, status_code=400)
        predictions = model.predict(X_batch)
        result_df = X_batch.copy()
        result_df['Predicted_Eye_State'] = predictions 
        stream = io.StringIO()
        result_df.to_csv(stream, index=False)
        response = StreamingResponse(iter([stream.getvalue()]), media_type="text/csv")
        response.headers["Content-Disposition"] = "attachment; filename=batch_results.csv"
        return response
    except Exception as e: return JSONResponse({"status": "error", "message": str(e)}, status_code=500)

@app.post("/visualize_batch_api")
async def visualize_batch(file: UploadFile = File(...), start: int = Form(0), end: int = Form(50)):
    global model
    if system_memory["active_model_name"] is None: return {"status": "error", "message": "No Active Model! Please select one first."}
    if model is None: model = load_model()
    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        total_rows = len(df)
        if start < 0: start = 0
        if end > total_rows: end = total_rows
        if start >= end: return {"status": "error", "message": "Start index must be less than End index."}
        df_preview = df.iloc[start:end].copy()
        feature_columns = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
        if all(col in df_preview.columns for col in feature_columns):
            X_chunk = df_preview[feature_columns]
            predictions = model.predict(X_chunk)
            eye_states = predictions.tolist()
        else:
            eye_states = [] 
        if 'id' in df_preview.columns: df_preview = df_preview.drop('id', axis=1)
        if 'eyeDetection' in df_preview.columns: df_preview = df_preview.drop('eyeDetection', axis=1)
        signals = df_preview.to_dict(orient='list')
        labels = list(range(start, end))
        return {"status": "success", "signals": signals, "labels": labels, "eye_states": eye_states}
    except Exception as e:
        return {"status": "error", "message": str(e)}

@app.post("/model_analysis_api")
def get_model_analysis():
    if system_memory["stats"] is None: return {"status": "error", "message": "No active model! Please select and train a model first."}
    return {"status": "success", "data": system_memory["stats"], "model_name": system_memory["active_model_name"]}

@app.post("/set_model_api")
def set_model(model_name: str = Form(...)):
    if model_name in model_factory: return optimize_model(model_name)
    return {"status": "Error", "message": "Model not found"}

@app.post("/dataset_stats_api")
def get_dataset_stats():
    try:
        if not os.path.exists(csv_path):
            return {"status": "error", "message": "Train.csv not found"}
        df = pd.read_csv(csv_path)
        total_samples = len(df)
        if 'eyeDetection' in df.columns:
            closed_count = int(df['eyeDetection'].sum()) 
            open_count = total_samples - closed_count    
        else:
            closed_count = 0; open_count = 0
        return {"status": "success", "total": total_samples, "open": open_count, "closed": closed_count, "features": 14}
    except Exception as e:
        return {"status": "error", "message": str(e)}
