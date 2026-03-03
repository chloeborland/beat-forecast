from pathlib import Path
import joblib
import numpy as np
import sklearn

print("NumPy:", np.__version__)
print("scikit-learn:", sklearn.__version__)

BASE = Path(__file__).resolve().parent
IN_PATH = BASE / "hit_gb_pipeline_wade.joblib"
OUT_PATH = BASE / "hit_gb_pipeline_wade_py311.joblib"

print("IN_PATH:", IN_PATH)
print("Exists:", IN_PATH.exists())

try:
    model = joblib.load(IN_PATH.as_posix())
    joblib.dump(model, OUT_PATH.as_posix(), compress=3)
    print("Saved:", OUT_PATH)
except Exception as e:
    print("FAILED to load model:", repr(e))
    raise