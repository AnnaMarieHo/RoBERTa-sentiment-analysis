import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch

# Load dataset
df = pd.read_csv("../all_intermediary_datasets/emotion_dataset_20.csv")
length = len(df)

# Load pre-trained model for semantic similarity: all-MiniLM-L6-v2 performed the best
model = SentenceTransformer('all-MiniLM-L6-v2')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# function to get the top emotion based on semantic similarity
def get_top_emotion(row):
    # the first column is 'text'
    emotions = row.index[1:]  
    comment_embedding = model.encode(row['text'], convert_to_tensor=True).to(device)
    
    # Create a list of (emotion, similarity) tuples
    emotion_similarities = []
    for emotion in emotions:
        # Only consider emotions that are flagged
        if row[emotion] == 1:  
            emotion_embedding = model.encode(emotion, convert_to_tensor=True).to(device)
            similarity = util.pytorch_cos_sim(comment_embedding, emotion_embedding)
            emotion_similarities.append((emotion, similarity.item()))
    
    # Sort by similarity and get the top emotion
    if emotion_similarities:
         # Get the emotion with the highest similarity
        top_emotion = max(emotion_similarities, key=lambda x: x[1]) 
        # Return only the emotion name
        return top_emotion[0]  
    return None  

output_df = df.copy()  

# Process only the first 10 rows
for index, row in df.iterrows():
    
    print(f"Processing row {index + 1} of {length}")
    # Get the top emotion
    top_emotion = get_top_emotion(row)  
    if index % 500 == 0:  
        print(row['text'], top_emotion)
    for emotion in row.index[1:]: 
        output_df.at[index, emotion] = 0  
    if top_emotion:
        output_df.at[index, top_emotion] = 1  

output_df.to_csv("emotion_dataset_with_representative.csv", index=False)
print("Successfully saved dataset with reduced emotions.")