# ------------------------------------
# PIPELINE
# ------------------------------------
# 1. go through every CT image
# 2. load the image
# 3. run a segmentation pipeline
#   3.1 gaussian
#   3.2 median
#   3.3 fixed thresholding
#   3.4 otsu
#   3.5 opening
#   3.6 closing
# 4. compare results with manual mask
# 5. metrics

import os
import numpy as np
import pandas as pd
from skimage.filters import threshold_otsu
from scipy.ndimage import gaussian_filter, median_filter
from skimage import morphology as morph
from NiftyIO import readNifty, saveNifty

# ------------------------------------
# PARAMETERS
# ------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_INPUT_ROOT = os.path.join(BASE_DIR, "..", "data", "input")
DATA_OUTPUT_ROOT = os.path.join(BASE_DIR, "..", "data", "output")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")
IMAGE_DIR = os.path.join(DATA_INPUT_ROOT, "image")
MASK_DIR  = os.path.join(DATA_INPUT_ROOT, "nodule_mask")

RESULTS_CSV = os.path.join(RESULTS_DIR, "results.csv")

GAUSSIAN_SIGMAS = [0, 0.5, 1, 2, 3, 4]  # 0 = no Gaussian
MEDIAN_SIZES    = [1, 3, 5]             # 1 = no Median

THRESHOLDS = [
    ("otsu", None),
    ("fixed", -500),
    ("fixed", -400),
    ("fixed", -300)
]

OPENING_SIZES = [0, 2, 3]  # 0 = no opening
CLOSING_SIZES = [0, 2, 3]  # 0 = no closing


def dice_coef(a, b):
    a = a > 0
    b = b > 0
    inter = np.sum(a & b)
    denom = np.sum(a) + np.sum(b)
    return 2 * inter / denom if denom != 0 else 1.0


def segment_pipeline(img, sigma, med_size, thr_method, thr_value, open_size, close_size):
    img_proc = img.copy()

    #apply gaussian/median
    if sigma > 0:
        img_proc = gaussian_filter(img_proc, sigma=sigma)
    if med_size > 1:
        img_proc = median_filter(img_proc, size=med_size)
    
    #calculate threshold
    if thr_method == "otsu":
        th = threshold_otsu(img_proc)
    else:
        th = thr_value
    
    mask = img_proc > th

    # postprocessing
    if open_size > 0:
        mask = morph.binary_opening(mask, morph.cube(open_size))
    if close_size > 0:
        mask = morph.binary_closing(mask, morph.cube(close_size))

    return mask, th


def main():
    all_results = []
    masks_results = []
    test_img = ["LIDC-IDRI-0001_R_1.nii.gz", "LIDC-IDRI-0003_R_2.nii.gz", "LIDC-IDRI-0003_R_3.nii.gz", "LIDC-IDRI-0003_R_4.nii.gz", "LIDC-IDRI-0004_R_1.nii.gz"]
    
    #for img_name in os.listdir(IMAGE_DIR):
    #    if not img_name.endswith(".nii") and not img_name.endswith(".nii.gz"):
    #        continue
    for img_name in test_img:
        masks_results = []
        results = []

        img_path = os.path.join(IMAGE_DIR, img_name)
        mask_path = os.path.join(MASK_DIR, img_name)
        img, metadata = readNifty(img_path)
        gt_mask, _ = readNifty(mask_path)

        print(f"Start analysis with {img_name}")
        for sigma in GAUSSIAN_SIGMAS:
            for med_size in MEDIAN_SIZES:
                for thr_method, thr_value in THRESHOLDS:
                    for open_size in OPENING_SIZES:
                        for close_size in CLOSING_SIZES:
                            print(f"Parameters")
                            print(f"Gaussian sigma: {sigma}")
                            print(f"Median size: {med_size}")
                            print(f"Threshold type: {thr_method}")
                            print(f"Threshold value: {thr_value}")
                            print(f"Open size: {open_size}")
                            print(f"Close size: {close_size}")
                            
                            mask, th = segment_pipeline(img, sigma, med_size, thr_method, thr_value, open_size, close_size)
                            dice = dice_coef(mask, gt_mask)
                            
                            thr_val = th if thr_method == "otsu" else thr_value
                            
                            results.append({
                                "image": img_name,
                                "sigma": sigma,
                                "median_size": med_size,
                                "threshold_method": thr_method,
                                "threshold_value": thr_val,
                                "open_size": open_size,
                                "close_size": close_size,
                                "dice": dice
                            })
                            masks_results.append((mask, metadata))
        
        all_results.extend(results)

        all_dice = [r['dice'] for r in results]
        best_idx = np.argmax(all_dice)
        best_mask, metadata = masks_results[best_idx]

        output_path = os.path.join(DATA_OUTPUT_ROOT, f"best_mask_{img_name}")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        saveNifty(best_mask.astype(np.uint8), metadata, output_path)
    
    
    df = pd.DataFrame(all_results)
    os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)
    print("Pipeline finished. All opening/closing combinations tested.")

if __name__ == "__main__":
    main()