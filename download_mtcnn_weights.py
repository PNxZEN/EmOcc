"""
Download MTCNN pretrained weights (pnet.pt, rnet.pt, onet.pt)
"""

import os
import requests
from tqdm import tqdm

def download_file(url, destination):
    """Download a file from URL with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    
    with open(destination, 'wb') as f:
        with tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(destination)) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    print(f"Downloaded: {destination}")

def main():
    # Base URL for MTCNN weights from timesler/facenet-pytorch repository
    base_url = "https://github.com/timesler/facenet-pytorch/raw/master/data/"
    
    # Files to download
    files = ['pnet.pt', 'rnet.pt', 'onet.pt']
    
    # Destination directory
    data_dir = 'data'
    
    print("Downloading MTCNN pretrained weights...")
    print("-" * 60)
    
    for filename in files:
        url = base_url + filename
        destination = os.path.join(data_dir, filename)
        
        if os.path.exists(destination):
            print(f"File already exists: {destination}")
        else:
            try:
                download_file(url, destination)
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
    
    print("-" * 60)
    print("Download complete!")

if __name__ == '__main__':
    main()
