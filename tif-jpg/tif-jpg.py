import os
from PIL import Image
from concurrent.futures import ProcessPoolExecutor, as_completed

input_folder = '/scratch/englisa8/QGIS/michael/mit_ml/py_clip/output/'
output_folder = '/scratch/englisa8/QGIS/michael/mit_ml/py_clip/output_jpg/'

os.makedirs(output_folder, exist_ok=True)

# Get a list of all .tif or .tiff files
tif_files = [
    f for f in os.listdir(input_folder)
    if f.lower().endswith(('.tif', '.tiff'))
]

# Function to convert a single file
def convert_tif_to_jpg(filename):
    input_path = os.path.join(input_folder, filename)
    output_filename = os.path.splitext(filename)[0] + '.jpg'
    output_path = os.path.join(output_folder, output_filename)
    
    try:
        with Image.open(input_path) as img:
            img.convert('RGB').save(output_path, 'JPEG')
        return f"Converted: {filename} to {output_filename}"
    except Exception as e:
        return f" Failed: {filename} ({e})"

# Use all available cores (you can limit if needed)
max_workers = os.cpu_count()

# Run conversions in parallel
if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(convert_tif_to_jpg, f): f for f in tif_files}
        for future in as_completed(futures):
            print(future.result())



