from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib, json

# ==============================
# Paths
# ==============================
MODEL_PATH = "dbscan_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODER_PATH = "label_encoder.pkl"
MAPPING_PATH = "risk_mapping.json"
CLUSTERED_PATH = "clustered_output.csv"

# ==============================
# Load Static Artifacts
# ==============================
try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    label_encoder = joblib.load(ENCODER_PATH)
    with open(MAPPING_PATH, "r") as f:
        risk_mapping = json.load(f)
except Exception as e:
    raise RuntimeError(f"❌ Error loading artifacts: {e}")

# convert mapping keys back to int
risk_labels = {int(k): v for k, v in risk_mapping.items()}

# ==============================
# FastAPI Setup
# ==============================
app = FastAPI(
    title="Risk Zone Prediction API",
    description="Predicts next risky zones and provides 5 random risky zone outputs",
    version="1.0"
)

# ==============================
# CORS Middleware
# ==============================
origins = [
    "http://localhost:5173",  # React dev server
    "http://127.0.0.1:5173",
    "http://localhost:3000",  # optional, if React runs on 3000
    "https://sudharshanchakrasih25.vercel.app",  # deployed frontend"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # allow your frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# Predict Random Risky Zones
# ==============================
@app.get("/predict1")
def predict_next_risks():
    try:
        # Reload CSV fresh on each request
        df = pd.read_csv(CLUSTERED_PATH)

        risky_predictions = df[df["next_risk"].isin(["High-Risk", "Restricted"])]

        if risky_predictions.empty:
            return {"error": "No risky zones found in CSV"}

        # If <= 5 risky zones exist, just return them all
        if len(risky_predictions) <= 5:
            random5 = risky_predictions[["latitude", "longitude", "next_risk"]]
        else:
            # Force fresh randomness each request
            random5 = risky_predictions[["latitude", "longitude", "next_risk"]] \
                .sample(n=5, replace=False)

        random5 = random5.rename(columns={"next_risk": "risk"})

        return random5.to_dict(orient="records")
    except Exception as e:
        return {"error": str(e)}

# ==============================
# Root Endpoint
# ==============================
@app.get("/")
def root():
    return {"message": "✅ Risk Zone Prediction API is running"}
