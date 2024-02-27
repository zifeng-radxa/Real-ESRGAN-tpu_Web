import urllib.request
import os

def download_file(url, file_name, folder_path='model'):
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Combine folder path and file name to get the full file path
    file_path = os.path.join(folder_path, file_name)
    if not os.path.exists(file_path):
    # Download the file from the URL and save it to the specified folder
        urllib.request.urlretrieve(url, file_path)
        print(f"File downloaded and saved to: {file_path}")

# Example usage:
# url = "https://example.com/samplefile.txt"
# folder_path = "downloaded_files"
# file_name = "samplefile.txt"
#
# download_file(url, folder_path, file_name)
