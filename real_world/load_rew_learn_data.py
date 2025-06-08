import os
import wget
from zipfile import ZipFile

DATA_URL = 'https://figshare.com/ndownloader/articles/20449356/versions/2'
DATA_PATH = 'rew_learn_data/'
DATA_FILENAME = '20449356.zip'

def ensure_data_downloaded():
    """Ensure Reward Learning data is downloaded and extracted."""
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"Downloading Reward Learning data...")
        wget.download(DATA_URL, DATA_PATH)
        os.rename(os.path.join(DATA_PATH, DATA_FILENAME), os.path.join(DATA_PATH, 'rew_learn_data.zip'))
        with ZipFile(os.path.join(DATA_PATH, 'rew_learn_data.zip'), 'r') as zipObj:
            zipObj.extractall(DATA_PATH)

if __name__ == "__main__":
    ensure_data_downloaded()