import sys
import pandas as pd
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset
from transformers import pipeline
from tqdm import tqdm
from pathlib import Path
from typing import List
from metadata_utils import get_metadata_entry

MODEL_NAME = 'MITLL/LADI-v2-classifier-small'

labels = [
    'trees_any',
    'water_any',
    'trees_damage',
    'debris_any',
    'roads_any',
    'flooding_any',
    'buildings_any',
    'buildings_affected_or_greater',
    'bridges_any',
    'flooding_structures',
    'roads_damage'
]

# Define valid image file extensions.
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}

class FileListDataset(Dataset):
    def __init__(self, paths: List[str]):
        # Filter paths that exist and have a valid image extension.
        self.paths = [
            Path(x)
            for x in paths
            if Path(x).exists() and Path(x).suffix.lower() in VALID_EXTENSIONS
        ]
        if not self.paths:
            raise ValueError("No valid image files found in provided paths.")

    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        file_path = self.paths[idx]
        try:
            img = Image.open(file_path)
            return img
        except UnidentifiedImageError as e:
            print(f"Skipping file {file_path} because it is not a valid image: {e}")
            return None

def postprocess_output(infer_output):
    output_dict = {}
    for response in infer_output:
        if response['label'] in labels:
            output_dict[response['label']] = response['score']
    return dict(sorted(output_dict.items()))

if __name__ == "__main__":
    # Reduce the number of worker processes to avoid slowness/freeze issues.
    pipe = pipeline(
         model=MODEL_NAME,
         task='image-classification',
         function_to_apply='sigmoid',
         num_workers=1
    )
    
    # Read file paths from the given file list.
    with open(sys.argv[1], 'r') as f:
        files = [line.strip() for line in f.readlines()]
    
    # Filter out files that do not have a valid image extension.
    image_files = [
        f for f in files 
        if Path(f).exists() and Path(f).suffix.lower() in VALID_EXTENSIONS
    ]
    
    if not image_files:
        print("No valid image files found.")
        sys.exit(1)
    
    ds = FileListDataset(image_files)
    
    outputs = []
    # Process the images in batches.
    for i, output in tqdm(enumerate(pipe(ds, batch_size=12, top_k=20)), total=len(ds)):
        classes = postprocess_output(output)
        curr_filename = image_files[i]
        img_metadata = get_metadata_entry(curr_filename)
        outputs.append({'file_path': curr_filename, **classes, **img_metadata})
    
    df = pd.DataFrame(data=outputs)
    df.to_csv('output.csv', index=False)
