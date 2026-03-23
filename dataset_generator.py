import os
# 🚨 CRITICAL FIXES FOR WINDOWS/INTEL i5 FREEZES 🚨
# THESE MUST BE AT THE VERY TOP BEFORE ANY OTHER IMPORTS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' 

import cv2
import numpy as np
import warnings
import random
import multiprocessing
import pandas as pd
import tensorflow as tf
from joblib import Parallel, delayed

from sklearn.model_selection import train_test_split
from scipy.stats import kurtosis, skew
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy
from skimage.filters import gabor

# ---------------------------------------------------------
# 1. REPRODUCIBILITY LOCK & CONFIGURATION
# ---------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except AttributeError:
    pass # Ignore on older TF versions

warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

IMG_SIZE = (300, 300)
BATCH_SIZE = 16
DATA_DIR = "dataset"
FEATURES_PER_CROP = 109  

PHISHING_FOLDERS = [
     "phishing ss 8th jan", "phishing ss 9th jan", "phishing ss 10th jan",
     "phishing ss 11th jan", "phishing ss 31th jan", "phishing ss tank",
     "phishing ss tank 2", "phishing ss 29th jan","phishing2","phishing ss 28th feb","phishing3","phishing4","phishing5","phishing ss 29th feb"
    
]

# 🚨 Updated to match the new legit folders in your reference code
LEGIT_FOLDERS = [
 "1000 legit images", "54 legit images", "300 legit images govt sector",
    "300 legit images health care", "300 edu legit ss", "300 legit ss","400 legit ss","200 legit ss","300 new legit ss"]

# ---------------------------------------------------------
# 2. FEATURE EXTRACTION FUNCTIONS
# ---------------------------------------------------------
def get_three_crops(img):
    h, w = img.shape[:2]
    if h < 10 or w < 10:
        res = cv2.resize(img, IMG_SIZE)
        return [res, res, res]
        
    if h >= w: 
        sq = w
        top = img[0:sq, 0:w]
        mid = img[(h-sq)//2 : (h+sq)//2, 0:w]
        bot = img[h-sq:h, 0:w]
    else:      
        sq = h
        top = img[0:h, 0:sq]
        mid = img[0:h, (w-sq)//2 : (w+sq)//2]
        bot = img[0:h, w-sq:w]
        
    return [cv2.resize(top, IMG_SIZE), cv2.resize(mid, IMG_SIZE), cv2.resize(bot, IMG_SIZE)]

def get_titan_features(img_arr):
    features = []
    try:
        img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        
        for img_space in [img_arr, img_hsv, img_lab]:
            for i in range(3):
                c = img_space[:,:,i]
                features.append(np.mean(c))
                features.append(np.std(c))
                if np.std(c) < 1e-6: features.extend([0.0, 0.0])
                else:
                    features.append(skew(c.flatten(), nan_policy='omit'))
                    features.append(kurtosis(c.flatten(), nan_policy='omit'))

        features.append(shannon_entropy(img_gray))
        glcm = graycomatrix(img_gray, [1, 3], [0, np.pi/4], 256, symmetric=True, normed=True)
        for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
            val = graycoprops(glcm, prop)
            features.append(val.mean())
            features.append(val.max() - val.min()) 

        lbp = local_binary_pattern(img_gray, 8, 1, method="uniform")
        n_bins = int(lbp.max() + 1)
        hist_lbp, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
        hist_pad = np.pad(hist_lbp, (0, max(0, 10 - len(hist_lbp))))[:10]
        features.extend(hist_pad)
        
        for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
            for freq in [0.1, 0.35]:
                filt_real, _ = gabor(img_gray, frequency=freq, theta=theta)
                features.append(np.mean(filt_real))
                features.append(np.std(filt_real))

        f = np.fft.fft2(img_gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
        features.append(np.mean(magnitude_spectrum))
        features.append(np.std(magnitude_spectrum))
        features.append(np.percentile(magnitude_spectrum, 90))
        if np.std(magnitude_spectrum) < 1e-6: features.append(0.0)
        else: features.append(skew(magnitude_spectrum.flatten()))

        edges = cv2.Canny(img_gray, 100, 200)
        features.append(np.count_nonzero(edges) / (edges.size + 1e-6)) 
        
        dst = cv2.cornerHarris(np.float32(img_gray), 2, 3, 0.04)
        features.append(np.count_nonzero(dst > 0.01 * dst.max()) / (dst.size + 1e-6))
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        features.append(len(contours))
        
        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            features.append(np.max(areas)) 
            features.append(np.mean(areas))
        else: 
            features.extend([0.0, 0.0])

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        features.append(len(lines) if lines is not None else 0)
        moments = cv2.moments(img_gray)
        hu_moments = cv2.HuMoments(moments).flatten()
        for hu in hu_moments:
            features.append(-1 * np.sign(hu) * np.log10(np.abs(hu) + 1e-6))

        s_channel = img_hsv[:,:,1]
        features.append(np.mean(s_channel))
        features.append(np.std(s_channel))
        if np.std(s_channel) < 1e-6: features.extend([0.0, 0.0])
        else:
            features.append(skew(s_channel.flatten(), nan_policy='omit'))
            features.append(kurtosis(s_channel.flatten(), nan_policy='omit'))
        
        laplacian_var = cv2.Laplacian(img_gray, cv2.CV_64F).var()
        features.append(laplacian_var)
        diff_h = np.mean(np.abs(img_gray[::8, :] - np.roll(img_gray, 1, axis=0)[::8, :]))
        diff_v = np.mean(np.abs(img_gray[:, ::8] - np.roll(img_gray, 1, axis=1)[:, ::8]))
        features.append(diff_h + diff_v)

        flat_regions = img_gray[edges == 0]
        if len(flat_regions) > 0: features.append(np.var(flat_regions))
        else: features.append(0.0)

        # ---------------------------------------------------------
        # TRUST FACTOR 1: Header/Footer Scanner
        # ---------------------------------------------------------
        h, w = img_gray.shape
        top_10 = img_gray[0:max(1, int(h*0.10)), :]
        bottom_10 = img_gray[max(1, int(h*0.90)):h, :]
        
        top_edges = cv2.Canny(top_10, 100, 200)
        bottom_edges = cv2.Canny(bottom_10, 100, 200)
        
        features.append(np.count_nonzero(top_edges) / (top_edges.size + 1e-6))
        features.append(np.count_nonzero(bottom_edges) / (bottom_edges.size + 1e-6))

        # ---------------------------------------------------------
        # TRUST FACTOR 2: Intentional Background Detector
        # ---------------------------------------------------------
        hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
        features.append(np.max(hist) / (img_gray.size + 1e-6))
        
        features.append(np.sum(img_gray > 245) / (img_gray.size + 1e-6))
        features.append(np.sum(img_gray < 10) / (img_gray.size + 1e-6))

        # ---------------------------------------------------------
        # 🚨 NEW: ERROR LEVEL ANALYSIS (ELA) FORGERY DETECTOR
        # ---------------------------------------------------------
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        
        ela_diff = cv2.absdiff(img_bgr, decimg)
        ela_gray = cv2.cvtColor(ela_diff, cv2.COLOR_BGR2GRAY)
        
        features.append(np.mean(ela_gray))                       # Average error
        features.append(np.std(ela_gray))                        # Variance in error 
        features.append(np.max(ela_gray))                        # Maximum error spike
        features.append(np.percentile(ela_gray, 95))             # 95th percentile error
        if np.std(ela_gray) < 1e-6: features.append(0.0)
        else: features.append(skew(ela_gray.flatten(), nan_policy='omit'))

        features = np.nan_to_num(np.array(features, dtype=np.float32))
        if len(features) >= FEATURES_PER_CROP:
            return features[:FEATURES_PER_CROP]
        else:
            padded = np.zeros(FEATURES_PER_CROP, dtype=np.float32)
            padded[:len(features)] = features
            return padded

    except: 
        return np.zeros(FEATURES_PER_CROP, dtype=np.float32)

# ---------------------------------------------------------
# 3. MAIN EXECUTION (BUILDING THE DATASET)
# ---------------------------------------------------------
if __name__ == '__main__':
    print("==================================================")
    print("🚀 RESEARCH DATASET GENERATOR (V33 + METADATA)")
    print("==================================================")

    data_crops = []
    labels = []
    
    # 🚨 METADATA ARRAYS
    image_names = []      
    relative_paths = []   
    source_folders = []
    orig_widths = []
    orig_heights = []
    aspect_ratios = []
    file_sizes_kb = []

    def load_images(folder_list, label_val, name):
        loaded = 0
        for folder_name in folder_list:
            full_path = os.path.join(DATA_DIR, folder_name)
            if not os.path.exists(full_path): continue
            
            for img_name in sorted(os.listdir(full_path)):
                try:
                    p = os.path.join(full_path, img_name)
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): continue
                    
                    # Read image to get original metadata BEFORE resizing
                    img = cv2.imread(p)
                    if img is not None:
                        orig_h, orig_w = img.shape[:2]
                        file_size = os.path.getsize(p) / 1024.0
                        
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        data_crops.append(get_three_crops(img))
                        labels.append(label_val)
                        
                        # Store Metadata
                        image_names.append(img_name)
                        relative_paths.append(f"{DATA_DIR}/{folder_name}/{img_name}".replace("\\", "/"))
                        source_folders.append(folder_name)
                        orig_widths.append(orig_w)
                        orig_heights.append(orig_h)
                        aspect_ratios.append(round(orig_w / orig_h, 4) if orig_h > 0 else 0.0)
                        file_sizes_kb.append(round(file_size, 2))
                        
                        loaded += 1
                except: pass
        print(f"   >>> Loaded {loaded} {name} images.")

    print("\n[1/5] Loading Images and Extracting Metadata...")
    load_images(PHISHING_FOLDERS, 1, "PHISHING")
    load_images(LEGIT_FOLDERS, 0, "LEGITIMATE")

    print("\n[2/5] Generating Permanent Train/Test Split Lock...")
    idx = np.arange(len(labels))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=SEED)
    
    suggested_split = np.empty(len(labels), dtype=object)
    suggested_split[train_idx] = "Train"
    suggested_split[test_idx] = "Test"

    all_crops = np.array([crop for img_crops in data_crops for crop in img_crops])
    total_images = len(labels)

    print(f"\n[3/5] Extracting Deep Learning Features (Max Pooling) for {total_images} images...")
    extractor = EfficientNetB3(weights='imagenet', include_top=False, pooling='max', input_shape=(300, 300, 3))
    extractor.trainable = False # 🚨 Fix to prevent memory leaks during prediction
    
    feats = []
    total_batches = (len(all_crops) // BATCH_SIZE) + 1
    for i in range(0, len(all_crops), BATCH_SIZE): 
        batch = preprocess_input(all_crops[i:i+BATCH_SIZE].astype('float32'))
        feats.append(extractor.predict(batch, verbose=0))
        if (i // BATCH_SIZE) % 10 == 0:
            print(f"   >>> CNN Batch {(i//BATCH_SIZE)+1}/{total_batches}")
    
    cnn_features_raw = np.vstack(feats)
    cnn_features = cnn_features_raw.reshape(total_images, -1) 

    print("\n[4/5] Extracting Manual & ELA Features (Parallel Processing)...")
    SAFE_CORES = min(4, multiprocessing.cpu_count()) 
    man_features_raw = np.array(Parallel(n_jobs=SAFE_CORES, backend='threading', verbose=10)(delayed(get_titan_features)(img) for img in all_crops))
    man_features = man_features_raw.reshape(total_images, FEATURES_PER_CROP * 3)

    print("\n[5/5] Assembling Academic DataFrame and Saving to CSV...")
    
    cnn_cols = []
    for crop in [1, 2, 3]:
        for f in range(1, 1537):
            cnn_cols.append(f"CNN_Crop{crop}_F{f}")
            
    man_cols = []
    for crop in [1, 2, 3]:
        for f in range(1, 110):
            man_cols.append(f"Manual_Crop{crop}_F{f}")

    all_cols = cnn_cols + man_cols
    final_feature_matrix = np.hstack([cnn_features, man_features])

    df = pd.DataFrame(final_feature_matrix, columns=all_cols)
    
    # INSERT METADATA COLUMNS AT THE START
    df.insert(0, 'Image_Name', image_names)
    df.insert(1, 'Relative_Path', relative_paths)
    df.insert(2, 'Source_Folder', source_folders)
    df.insert(3, 'Original_Width', orig_widths)
    df.insert(4, 'Original_Height', orig_heights)
    df.insert(5, 'Aspect_Ratio', aspect_ratios)
    df.insert(6, 'File_Size_KB', file_sizes_kb)
    df.insert(7, 'Suggested_Split', suggested_split)
    
    df['True_Label'] = labels

    output_filename = "Research_Dataset_V33.csv" 
    df.to_csv(output_filename, index=False)

    print("\n" + "="*50)
    print(f"✅ SUCCESS! Academic dataset saved to '{output_filename}'")
    print(f"   Total Rows (Images): {len(df)}")
    print(f"   Total Columns: {len(df.columns)}")
    print("="*50)