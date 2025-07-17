# Before run:
# pip install kaggle
# place kaggle username na dpassword in .env ans export
from kaggle.api.kaggle_api_extended import KaggleApi
import os, json

DATASET    = 'nadayoussefamrawy/ms-asl'
RAW_DIR    = 'msasl_raw'
GLOSS_LIST = 'top106.json'
SPLITS     = ['train_list.txt', 'dev_list.txt', 'test_list.txt']

# 1. Auth
api = KaggleApi()
api.authenticate()

# 2. Load target glosses
with open(GLOSS_LIST) as f:
    target = set(json.load(f))

# 3. Download just the split files
os.makedirs(RAW_DIR, exist_ok=True)
for split in SPLITS:
    api.dataset_download_file(DATASET, file_name=split,
                              path=RAW_DIR, unzip=True)

# 4. Read split files to build a set of desired video paths
wanted = set()
for split in SPLITS:
    for ln in open(os.path.join(RAW_DIR, split)):
        gloss = ln.split('/',1)[0]
        if gloss in target:
            wanted.add(ln.strip())

# 5. List every file in the dataset and download only the ones in `wanted`
files = api.dataset_list_files(DATASET).files
for file in files:
    # file.name is e.g. 'BATH/0.mp4' or 'BATH/1.mp4'
    if file.name in wanted:
        api.dataset_download_file(DATASET,
                                  file_name=file.name,
                                  path=os.path.join(RAW_DIR, os.path.dirname(file.name)),
                                  unzip=True)

print(f"Downloaded {len(wanted)} filtered videos.")
