import urllib.request
import os

os.makedirs('data', exist_ok=True)

zenodo_url = "ZENODO_FILE_DOWNLOAD_LINK" # TODO
output_path = "data/"

print("Downloading dataset from Zenodo ...")
urllib.request.urlretrieve(zenodo_url, output_path)
print(f"Data successfully downloaded to {output_path}")