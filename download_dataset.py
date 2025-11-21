from sklearn.datasets import fetch_20newsgroups
import os

# 1. Load the dataset
dataset = fetch_20newsgroups(subset='train')

# 2. Create data/ folder if not exists
data_dir = "data"
os.makedirs(data_dir, exist_ok=True)

# 3. Save each document as a .txt file
for i, text in enumerate(dataset.data):
    file_path = os.path.join(data_dir, f"doc_{i+1:03}.txt")
    with open(file_path, "w", encoding="utf-8", errors="ignore") as f:
        f.write(text)

print(f"Saved {len(dataset.data)} documents in '{data_dir}/'")
