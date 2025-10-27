import pandas as pd

df = pd.read_csv("../all_intermediary_datasets/reannotated_goemotions.csv")

# Define mapping from 38 intermediate labels to 20 reduced categories
emotion_map_20 = {
    "Agreement": "approval",
    "Approval": "approval",
    "Amusement": "amusement",
    "Aggression": "anger",
    "Anger": "anger",
    "Defiance": "anger",
    "Annoyance": "annoyance",
    "Frustration": "annoyance",
    "Anticipation": "excitement",
    "Excitement": "excitement",
    "Anxiety": "fear",
    "Fear": "fear",
    "Apology": "remorse",
    "Caring": "caring",
    "Support": "caring",
    "Confidence": "pride",
    "Pride": "pride",
    "Sarcasm": "disapproval",
    "Contempt": "disapproval",
    "Disapproval": "disapproval",
    "Disagreement": "disapproval",
    "Confusion": "confusion",
    "Skepticism": "confusion",
    "Curiosity": "curiosity",
    "Disappointment": "disappointment",
    "Disbelief": "surprise",
    "Surprise": "surprise",
    "Disgust": "disgust",
    "Encouragement": "optimism",
    "Hope": "optimism",
    "Resilience": "optimism",
    "Gratitude": "gratitude",
    "Love": "love",
    "Joy": "joy",
    "Resignation": "sadness",
    "Sadness": "sadness",
    "Neutral": None,
}

reduced_columns = sorted(set(v for v in emotion_map_20.values() if v is not None))

for col in reduced_columns:
    df[col] = 0

# propagate 1s from all mapped labels to their reduced category using OR 
for original_label, reduced_label in emotion_map_20.items():
    if original_label in df.columns and reduced_label is not None:
        print(f"Processing: {original_label} to {reduced_label}")
        # Use OR to combine values
        df[reduced_label] = df[reduced_label] | df[original_label]

for col in reduced_columns:
    ones_count = df[col].sum()
    print(f"{col}: {ones_count} instances of 1s")

# Save final dataset
df_out = df[["text"] + reduced_columns]
df_out.to_csv("emotion_dataset_20.csv", index=False)
print(" Successfully saved dataset with 20 classes.")


