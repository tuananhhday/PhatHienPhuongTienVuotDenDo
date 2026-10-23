import urllib.request
from pathlib import Path

# Danh sach cac video vi pham vuot den do tu bo du lieu Red Light Violation System
VIDEOS = [f"red{i}.mp4" for i in range(1, 11)]
BASE_URL = "https://raw.githubusercontent.com/diogomaymone/Red_Light_Violation_Detection_System/main/Red_Light_Violation_Detection_System/video_data/Violations/"

def download_videos(target_dir="dataset_videos"):
    save_path = Path(target_dir)
    save_path.mkdir(exist_ok=True, parents=True)
    
    print(f"=== BAT DAU TAI BỘ VIDEO VUOT DEN DO ({len(VIDEOS)} videos) ===")
    for v in VIDEOS:
        dest = save_path / v
        if dest.exists():
            print(f"[-] {v} da ton tai, bo qua.")
            continue
        url = BASE_URL + v
        print(f"[+] Dang tai {v}...")
        try:
            urllib.request.urlretrieve(url, str(dest))
            print(f"    -> Thanh cong: {v} ({dest.stat().st_size / 1024 / 1024:.2f} MB)")
        except Exception as e:
            print(f"    [!] Loi khi tai {v}: {e}")

    print("\n[HOAN TAT] Tat ca video da duoc luu vao thu muc:", save_path.resolve())

if __name__ == "__main__":
    download_videos()
