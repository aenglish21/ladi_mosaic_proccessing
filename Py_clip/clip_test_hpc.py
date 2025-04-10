import os

import numpy as np

import rasterio

from rasterio.mask import mask

from rasterio.windows import from_bounds, transform as window_transform

from rasterio.io import MemoryFile

import fiona

from shapely.geometry import shape

from concurrent.futures import ProcessPoolExecutor, as_completed



def process_feature(feature, raster_path, output_dir):

    feature_id = feature["id"]

    geom = feature["geometry"]

    

    # Convert the feature geometry to a shapely geometry and get its bounds

    shp_geom = shape(geom)

    bounds = shp_geom.bounds  # (minx, miny, maxx, maxy)

    

    try:

        with rasterio.open(raster_path) as src:

            # Compute a window covering the polygon's bounding box.

            # By providing height and width, we ensure the window is clipped to the raster.

            window = from_bounds(*bounds, transform=src.transform, height=src.height, width=src.width)

            # Read only the data within that window

            subset_data = src.read(window=window)

            # Compute a new transform for this window

            new_transform = window_transform(window, src.transform)

            # Copy and update metadata for this subset

            subset_meta = src.meta.copy()

            subset_meta.update({

                "height": subset_data.shape[1],

                "width": subset_data.shape[2],

                "transform": new_transform

            })

            nodata_value = src.nodata if src.nodata is not None else 0



        # Create an in-memory dataset for clipping the subset

        with MemoryFile() as memfile:

            with memfile.open(**subset_meta) as mem:

                mem.write(subset_data)

                # Clip the in-memory dataset using the polygon geometry; crop exactly to the polygon.

                clipped_data, clipped_transform = mask(mem, [geom], crop=True, nodata=nodata_value)

                # Update metadata for the clipped output

                clipped_meta = subset_meta.copy()

                clipped_meta.update({

                    "height": clipped_data.shape[1],

                    "width": clipped_data.shape[2],

                    "transform": clipped_transform,

                    "nodata": nodata_value

                })



        output_fp = os.path.join(output_dir, f"clip_{feature_id}.tif")

        with rasterio.open(output_fp, "w", **clipped_meta) as dest:

            dest.write(clipped_data)

        return output_fp

        

    except Exception as e:

        print(f"Error processing feature {feature_id}: {e}")

        return None



def main():

    raster_path = "/scratch/englisa8/QGIS/michael/mit_ml/py_clip/input_raster/20181011aC0853130w300600n.tif"

    shapefile_path = "/scratch/englisa8/QGIS/michael/mit_ml/py_clip/Gridsv3.shp"

    output_dir = "/scratch/englisa8/QGIS/michael/mit_ml/py_clip/output"

    

    if not os.path.exists(output_dir):

        os.makedirs(output_dir)

    

    # Limit the number of processes to avoid overwhelming the I/O system.

    num_workers = min(16, os.cpu_count() or 1)

    print(f"Using {num_workers} processes for parallel processing.")

    

    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:

        futures = []

        # Iterate over features one by one (lazy iteration)

        with fiona.open(shapefile_path, "r") as shp:

            for feature in shp:

                futures.append(executor.submit(process_feature, feature, raster_path, output_dir))

                

        for future in as_completed(futures):

            result = future.result()

            if result:

                print(f"Created: {result}")

                results.append(result)

    

    print("All features processed.")



if __name__ == "__main__":

    main() 