
from app import predict_difficulty, LABELS
import pandas as pd

df = pd.read_csv("final_train.csv")
medium_texts = df[df["label"] == 1]["text"].dropna().head(50).tolist()

for i, text in enumerate(medium_texts):
    pred = predict_difficulty(text)
    if pred == 1:
        print(f"FOUND A TRUE MEDIUM TEXT (Index {i}):")
        print(text)
        break

