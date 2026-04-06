import os
import urllib.request
import argparse

def download_etth1(output_dir="datasets/etth1"):
    os.makedirs(output_dir, exist_ok=True)
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    dest_path = os.path.join(output_dir, "ETTh1.csv")
    
    if os.path.exists(dest_path):
        print(f"[*] ETTh1 dataset already exists at {dest_path}")
        return
        
    print(f"[*] Downloading ETTh1 dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print(f"[+] Download complete! Saved to {dest_path}")
    except Exception as e:
        print(f"[Error] Failed to download ETTh1: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default=os.path.join(os.path.dirname(__file__), "etth1"))
    args = parser.parse_args()
    
    download_etth1(args.output_dir)
