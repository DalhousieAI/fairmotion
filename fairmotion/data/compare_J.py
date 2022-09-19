from fairmotion.data import amass_dip
from pathlib import Path
import numpy as np
import pickle

J = amass_dip.OFFSETS

if __name__ == "__main__":
    bmdir = Path("../../tests/body_models")
    for bm_path in bmdir.glob("*.pkl"):
        with open(bm_path, "rb") as f:
            bm = pickle.load(f, encoding='latin1')
            if np.allclose(bm['J'], J):
                print(bm_path, ": match")
            else:
                print(bm_path, ": mismatch")
