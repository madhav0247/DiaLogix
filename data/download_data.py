import urllib.request
import os

url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
dest_dir = os.path.dirname(os.path.abspath(__file__))
dest_file = os.path.join(dest_dir, "diabetes.csv")

print(f"Downloading Pima dataset from {url} to {dest_file}...")
urllib.request.urlretrieve(url, dest_file)

with open(dest_file, "r") as f:
    lines = f.readlines()

print(f"Successfully downloaded {len(lines)} lines (including header).")
print(f"Header: {lines[0].strip()}")
print(f"First row: {lines[1].strip()}")
