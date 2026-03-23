import os
# 🚨 CRITICAL FIXES FOR WINDOWS/INTEL i5 FREEZES 🚨
# THESE MUST BE AT THE VERY TOP BEFORE ANY OTHER IMPORTS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Gags TensorFlow C++ warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' # Kills the Intel OneDNN deadlock
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # Forces stable CPU execution

import cv2
import numpy as np
import pickle
import warnings
import random
import shutil
import multiprocessing
import tensorflow as tf
from joblib import Parallel, delayed

from scipy.stats import kurtosis, skew
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern
from skimage.measure import shannon_entropy
from skimage.filters import gabor
from catboost import CatBoostClassifier

# ---------------------------------------------------------
# 1. REPRODUCIBILITY LOCK
# ---------------------------------------------------------
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.config.experimental.enable_op_determinism()
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

# ---------------------------------------------------------
# 2. CONFIGURATION
# ---------------------------------------------------------
IMG_SIZE = (300, 300)
BATCH_SIZE = 16
DATA_DIR = "dataset"

# V31 ARCHITECTURE SIZES
FEATURES_PER_CROP = 109  # Upgraded to 109 to include the 5 ELA features

PHISHING_FOLDERS = [
    "phishing ss 8th jan", "phishing ss 9th jan", "phishing ss 10th jan",
    "phishing ss 11th jan", "phishing ss 31th jan", "phishing ss tank",
    "phishing ss tank 2", "phishing ss 29th jan","phishing2","phishing ss 28th feb","phishing3","phishing4","phishing5","phishing ss 29th feb"
]

LEGIT_FOLDERS = [
    "1000 legit images", "54 legit images", "300 legit images govt sector",
    "300 legit images health care", "300 edu legit ss", "300 legit ss","400 legit ss","200 legit ss","300 new legit ss"
]

# =============================================================================
# PART 3: MULTI-CROP SLICER & ADVANCED FEATURE EXTRACTOR
# =============================================================================
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
        
        # We also keep the pure white and pure black checks
        features.append(np.sum(img_gray > 245) / (img_gray.size + 1e-6))
        features.append(np.sum(img_gray < 10) / (img_gray.size + 1e-6))

        # ---------------------------------------------------------
        # 🚨 NEW: ERROR LEVEL ANALYSIS (ELA) FORGERY DETECTOR
        # ---------------------------------------------------------
        # Compress the image to 90% quality in memory
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
        decimg = cv2.imdecode(encimg, 1)
        
        # Calculate the absolute difference between original and compressed
        ela_diff = cv2.absdiff(img_bgr, decimg)
        ela_gray = cv2.cvtColor(ela_diff, cv2.COLOR_BGR2GRAY)
        
        # Extract 5 statistical clues from the error map
        features.append(np.mean(ela_gray))                       # Average error
        features.append(np.std(ela_gray))                        # Variance in error (spikes on splices)
        features.append(np.max(ela_gray))                        # Maximum error spike
        features.append(np.percentile(ela_gray, 95))             # 95th percentile error
        if np.std(ela_gray) < 1e-6: features.append(0.0)
        else: features.append(skew(ela_gray.flatten(), nan_policy='omit'))

        # Final check and padding
        features = np.nan_to_num(np.array(features, dtype=np.float32))
        if len(features) >= FEATURES_PER_CROP:
            return features[:FEATURES_PER_CROP]
        else:
            padded = np.zeros(FEATURES_PER_CROP, dtype=np.float32)
            padded[:len(features)] = features
            return padded

    except: 
        return np.zeros(FEATURES_PER_CROP, dtype=np.float32)

# =============================================================================
# 🚨 WINDOWS EXECUTION LOCK
# =============================================================================
if __name__ == '__main__':
    print("==================================================")
    print("🚀 SYSTEM V31: ELA FORGERY DETECTION ENABLED")
    print("==================================================")

    # ---------------------------------------------------------
    # PART 4: LOADING (1 Label per Page)
    # ---------------------------------------------------------
    print("\n[Step 1/5] Loading & Slicing High-Res Images...")
    data_crops, labels, original_paths = [], [], []

    def load_with_chunk_groups(folder_list, label_val, name):
        total_loaded = 0
        for folder_name in folder_list:
            full_path = os.path.join(DATA_DIR, folder_name)
            if not os.path.exists(full_path): continue
            for img_name in sorted(os.listdir(full_path)):
                try:
                    p = os.path.join(full_path, img_name)
                    if not img_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): continue
                    img = cv2.imread(p)
                    if img is not None:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        data_crops.append(get_three_crops(img))
                        labels.append(label_val)
                        original_paths.append(p)
                        total_loaded += 1
                except: pass
        print(f"   >>> Loaded {total_loaded} {name} images.")

    load_with_chunk_groups(PHISHING_FOLDERS, 1, "PHISHING")
    load_with_chunk_groups(LEGIT_FOLDERS, 0, "LEGITIMATE")

    labels = np.array(labels)
    original_paths = np.array(original_paths)

    print("\n[Step 2/5] Random Stratified Split (80% Train / 20% Test)...")
    idx = np.arange(len(labels))
    train_idx, test_idx = train_test_split(idx, test_size=0.2, stratify=labels, random_state=SEED)

    y_train = labels[train_idx]
    y_test = labels[test_idx]

    X_train_crops = np.array([crop for i in train_idx for crop in data_crops[i]])
    X_test_crops = np.array([crop for i in test_idx for crop in data_crops[i]])

    # ---------------------------------------------------------
    # PART 5: EXTRACT & CONCATENATE FEATURES
    # ---------------------------------------------------------
    print("\n[Step 3/5] Extracting Page-Level CNN Features...")
    # 🚨 Returned to pooling='max' so it strictly targets central elements (login boxes)
    extractor = EfficientNetB3(weights='imagenet', include_top=False, pooling='max', input_shape=(300, 300, 3))
    extractor.trainable = False 

    def get_deep_features(model, images, name=""):
        feats = []
        total_batches = (len(images) // BATCH_SIZE) + 1
        for i in range(0, len(images), BATCH_SIZE): 
            batch = preprocess_input(images[i:i+BATCH_SIZE].astype('float32'))
            feats.append(model.predict(batch, verbose=0))
            if (i // BATCH_SIZE) % 10 == 0:
                print(f"   >>> CNN [{name}]: Processed batch {(i//BATCH_SIZE)+1}/{total_batches}")
        return np.vstack(feats)

    cnn_train_crops = get_deep_features(extractor, X_train_crops, "Train")
    cnn_test_crops = get_deep_features(extractor, X_test_crops, "Test")

    print("\n[Step 3b/5] Extracting Page-Level Manual Features...")
    SAFE_CORES = min(4, multiprocessing.cpu_count()) 
    
    print("   >>> Extracting Manual Features (Train set)...")
    man_train_crops = np.array(Parallel(n_jobs=SAFE_CORES, backend='threading', verbose=10)(delayed(get_titan_features)(img) for img in X_train_crops))
    
    print("\n   >>> Extracting Manual Features (Test set)...")
    man_test_crops = np.array(Parallel(n_jobs=SAFE_CORES, backend='threading', verbose=10)(delayed(get_titan_features)(img) for img in X_test_crops))

    cnn_train_page = cnn_train_crops.reshape(len(train_idx), -1)
    cnn_test_page = cnn_test_crops.reshape(len(test_idx), -1)

    man_train_page = man_train_crops.reshape(len(train_idx), FEATURES_PER_CROP * 3)
    man_test_page = man_test_crops.reshape(len(test_idx), FEATURES_PER_CROP * 3)

    print("\n[Step 4/5] Scaling Manual Features (Bypassing Bouncer)...")
    scaler_man = StandardScaler()
    man_train_sel = scaler_man.fit_transform(man_train_page)
    man_test_sel = scaler_man.transform(man_test_page)

    X_train_final = np.hstack([cnn_train_page, man_train_sel])
    X_test_final = np.hstack([cnn_test_page, man_test_sel])

    imputer = SimpleImputer(strategy='mean')
    X_train_final = imputer.fit_transform(X_train_final)
    X_test_final = imputer.transform(X_test_final)

    scaler_final = StandardScaler()
    X_train_final = scaler_final.fit_transform(X_train_final)
    X_test_final = scaler_final.transform(X_test_final)

    # ---------------------------------------------------------
    # PART 6: MODEL TRAINING
    # ---------------------------------------------------------
    print("\n[Step 5/5] Training Page-Level CatBoost Model...")
    
    best_params = {
        'iterations': 1400, 
        'depth': 6,
        'learning_rate': 0.10,
        'l2_leaf_reg': 5.0,
        'loss_function': 'Logloss',
        'random_seed': SEED,
        'verbose': 200
    }

    clf = CatBoostClassifier(**best_params)
    clf.fit(X_train_final, y_train)

    best_threshold = 0.5000
    print(f"\n🎯 Locked Page-Level Threshold: {best_threshold:.4f}")

    # ---------------------------------------------------------
    # PART 7: FINAL EVALUATION
    # ---------------------------------------------------------
    probs_test = clf.predict_proba(X_test_final)[:, 1]
    preds_calibrated = (probs_test >= best_threshold).astype(int)

    print("\n" + "="*50)
    print("✅ FINAL RESULTS (V31: UNIFIED PAGE-LEVEL WITH ELA)")
    print("="*50)
    print(classification_report(y_test, preds_calibrated, target_names=["Legitimate", "Phishing"]))
    print(f"Accuracy: {accuracy_score(y_test, preds_calibrated):.4f}")

    mistakes_dir = "hard_mistakes_to_review"
    if not os.path.exists(mistakes_dir): os.makedirs(mistakes_dir)

    mistake_count = 0
    test_paths = original_paths[test_idx]
    for i in range(len(y_test)):
        if y_test[i] != preds_calibrated[i]:
            status = "Missed_Phishing" if y_test[i] == 1 else "False_Alarm_Legit"
            filename = f"{status}_{os.path.basename(test_paths[i])}"
            try:
                shutil.copy(test_paths[i], os.path.join(mistakes_dir, filename))
                mistake_count += 1
            except: pass

    print(f"\n📁 Saved {mistake_count} page-level mistakes to '{mistakes_dir}'.")

    # SAVING V31 FILES
    pickle.dump(clf, open("model_v40.pkl", "wb"))
    pickle.dump(scaler_final, open("scaler40.pkl", "wb"))
    pickle.dump(scaler_man, open("scaler_man_v40.pkl", "wb"))
    pickle.dump(imputer, open("imputer_v40.pkl", "wb"))
    pickle.dump(best_threshold, open("threshold_v40.pkl", "wb"))
    print("\n[Done] System V40 Saved.")