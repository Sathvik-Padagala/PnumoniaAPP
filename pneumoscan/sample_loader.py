"""
Generates and manages ready-to-test sample Chest X-rays for PneumoScan.
Includes:
1. Normal Chest X-ray (Stable)
2. Pneumonia Confident (Stable, dense consolidation)
3. Borderline Case (Uncertain, entropy > 0.50, needs expert review)
4. Corrupted / Non-Chest image (Lungs not found fallback trigger)
"""

import os
import cv2
import numpy as np

def create_synthetic_chest_xray(case_type="normal", size=(512, 512)):
    """Creates an anatomically realistic synthetic chest radiograph for testing."""
    h, w = size
    # Base dark background with slight tissue exposure
    xray = np.zeros((h, w), dtype=np.float32) + 20.0
    
    # Thoracic ribcage oval contour
    y, x = np.ogrid[:h, :w]
    thorax_mask = (((x - w / 2) / (0.42 * w)) ** 2 + ((y - h * 0.48) / (0.45 * h)) ** 2) <= 1.0
    xray[thorax_mask] += 90.0
    
    # Lung air cavities (darker regions inside thorax)
    # Left lung cavity
    left_lung = (((x - w * 0.33) / (0.13 * w)) ** 2 + ((y - h * 0.46) / (0.32 * h)) ** 2) <= 1.0
    # Right lung cavity
    right_lung = (((x - w * 0.67) / (0.13 * w)) ** 2 + ((y - h * 0.46) / (0.32 * h)) ** 2) <= 1.0
    
    # Air attenuation is lower, so lung fields are radiolucent (darker)
    xray[left_lung] -= 55.0
    xray[right_lung] -= 55.0
    
    # Mediastinum and cardiac silhouette (dense white in center, biased to left chest)
    heart_mask = (((x - w * 0.48) / (0.15 * w)) ** 2 + ((y - h * 0.58) / (0.18 * h)) ** 2) <= 1.0
    spine_mask = (np.abs(x - w / 2) < (0.04 * w)) & (y < h * 0.85)
    xray[heart_mask] += 60.0
    xray[spine_mask] += 40.0
    
    # Diaphragm domes
    left_hemidiaphragm = (y > h * 0.72) & (((x - w * 0.32) / (0.18 * w)) ** 2 <= 1.0)
    right_hemidiaphragm = (y > h * 0.70) & (((x - w * 0.68) / (0.18 * w)) ** 2 <= 1.0)
    xray[left_hemidiaphragm] += 50.0
    xray[right_hemidiaphragm] += 50.0

    # Add rib shadows
    for rib_y in range(int(h * 0.2), int(h * 0.75), int(h * 0.08)):
        rib_band = (np.abs(y - rib_y + 0.1 * np.abs(x - w / 2)) < (h * 0.02)) & thorax_mask
        xray[rib_band] += 18.0

    # Bronchovascular markings / anatomical texture
    np.random.seed(42 if case_type != "corrupted" else 99)
    noise = np.random.normal(0, 7.0, size=(h, w)).astype(np.float32)
    xray = xray + noise

    if case_type == "pneumonia_confident":
        # Dense patchy alveolar consolidation in right lower/middle lobe
        patch_r = (((x - w * 0.68) / (0.09 * w)) ** 2 + ((y - h * 0.52) / (0.12 * h)) ** 2) <= 1.0
        patch_fluffy = cv2.GaussianBlur(patch_r.astype(np.float32), (35, 35), 0)
        xray += patch_fluffy * 80.0
        
        # Second opacity patch
        patch_2 = (((x - w * 0.64) / (0.06 * w)) ** 2 + ((y - h * 0.40) / (0.08 * h)) ** 2) <= 1.0
        xray += cv2.GaussianBlur(patch_2.astype(np.float32), (25, 25), 0) * 55.0

    elif case_type == "uncertain_borderline":
        # Subtle peribronchial thickening or borderline interstitial infiltrates
        patch_subtle = (((x - w * 0.36) / (0.07 * w)) ** 2 + ((y - h * 0.50) / (0.09 * h)) ** 2) <= 1.0
        xray += cv2.GaussianBlur(patch_subtle.astype(np.float32), (45, 45), 0) * 26.0

    elif case_type == "corrupted_non_lung":
        # An image without clear lung fields (e.g. abdomen, artifact, or solid blank)
        xray_u8 = np.random.uniform(100, 150, size=(h, w)).astype(np.uint8)
        cv2.putText(xray_u8, "NO THORAX / BLANK ARTIFACT", (30, h // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230,), 2)
        return xray_u8

    # Smooth and clip to uint8
    xray = cv2.GaussianBlur(xray, (3, 3), 0)
    xray = np.clip(xray, 0, 255).astype(np.uint8)
    return xray

def ensure_sample_images(target_dir):
    """Generates a suite of sample images for quick testing in the UI."""
    os.makedirs(target_dir, exist_ok=True)
    samples = {
        "sample_1_normal_stable.jpg": "normal",
        "sample_2_pneumonia_confirmed.jpg": "pneumonia_confident",
        "sample_3_uncertain_borderline.jpg": "uncertain_borderline",
        "sample_4_corrupted_no_lungs.jpg": "corrupted_non_lung"
    }
    
    file_paths = {}
    for filename, case_type in samples.items():
        out_path = os.path.join(target_dir, filename)
        if not os.path.exists(out_path):
            img = create_synthetic_chest_xray(case_type)
            cv2.imwrite(out_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        file_paths[filename] = out_path
        
    return file_paths
