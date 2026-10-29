
from app import predict_difficulty, LABELS

medium_text = "Jatin Lalit are a music director duo of Bollywood. The duo consists of brothers, Jatin Pandit and Lalit Pandit. They started their music career in 1991. Their first breakthrough came a year later, when they composed the music for Jo Jeeta Wohi Sikandar. Since then they have composed for music for over 50 movies. In 2006, the duo stopped working together."

print("\n--- INFERENCE TEST START ---")
idx = predict_difficulty(medium_text)
print(f"Text: {medium_text}")
print(f"\nPredicted Class ID: {idx}")
print(f"Final Label: {LABELS[idx]}")
print("--- INFERENCE TEST END ---\n")

