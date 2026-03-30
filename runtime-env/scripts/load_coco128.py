import os
import urllib.request
import zipfile

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    datasets_dir = os.path.join(project_root, "datasets")
    os.makedirs(datasets_dir, exist_ok=True)
    
    url = "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip"
    zip_path = os.path.join(datasets_dir, "coco128.zip")
    
    print(f"[*] Downloading COCO128 from {url} to {datasets_dir}...")
    
    def reporthook(block_num, block_size, total_size):
        fetched = block_num * block_size
        if total_size > 0:
            percent = min(100, fetched * 100 // total_size)
            print(f"\rDownloading... {percent}% ({fetched/(1024*1024):.1f}MB / {total_size/(1024*1024):.1f}MB)", end="")

    try:
        urllib.request.urlretrieve(url, zip_path, reporthook)
        print("\n[+] Download complete!")
    except Exception as e:
        print(f"[!] 다운로드 실패: {e}")
        return
    
    print("[*] Extracting zip file...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # ultralytics coco128.zip 에는 자기 자신의 루트 폴더('coco128')가 내장되어 있습니다.
            zip_ref.extractall(datasets_dir)
    except Exception as e:
        print(f"[!] 압축 해제 실패: {e}")
        return
        
    print(f"[+] COCO128 dataset is ready at {os.path.join(datasets_dir, 'coco128')}")

if __name__ == "__main__":
    main()
