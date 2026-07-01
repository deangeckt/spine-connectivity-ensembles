import urllib.request
import zipfile
import os
import sys

zenodo_url = "https://zenodo.org/records/21104426/files/data.zip?download=1" 
zip_path = "data.zip"
extract_dir = "."

def show_progress(block_num, block_size, total_size):
    """Callback function to display a progress bar during download."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        
        # Calculate visual bar (50 characters wide)
        bar_length = 50
        filled_length = int(bar_length * downloaded // total_size)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        # Convert sizes to Megabytes for readability
        downloaded_mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        
        # \r forces the cursor to the start of the line to overwrite it
        sys.stdout.write(f'\rDownloading: [{bar}] {percent:.1f}% ({downloaded_mb:.1f} MB / {total_mb:.1f} MB)')
        sys.stdout.flush()

print("Initializing download from Zenodo...")

urllib.request.urlretrieve(zenodo_url, zip_path, reporthook=show_progress)

print("\nDownload complete. Extracting files...")

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

print("Extraction complete. (You can delete the 'data.zip' file if you wish to save space.)")

print(f"Success! All data files are now available in the local '{extract_dir}' directory.")