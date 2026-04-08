import os
import urllib.request

def prepare_squad2():
    base_dir = os.path.join("datasets", "squad2")
    os.makedirs(base_dir, exist_ok=True)
    
    url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
    target_path = os.path.join(base_dir, "val.json")
    
    print(f"[*] Downloading SQuAD 2.0 (dev) to {target_path}...")
    try:
        urllib.request.urlretrieve(url, target_path)
        print("[+] SQuAD 2.0 Download complete!\n")
    except Exception as e:
        print(f"[!] Failed to download SQuAD 2.0: {e}\n")

if __name__ == "__main__":
    prepare_squad2()
