import os
import pandas as pd
import numpy as np
import pickle
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier

SEED = 42
np.random.seed(SEED)
warnings.filterwarnings("ignore")

# Update this to your actual CSV path
DATASET_PATH = r"C:\Users\ravim\OneDrive\Desktop\ML training\Research_Dataset_V33.csv" 
TARGET_COLUMN = "True_Label" 

CNN_FEATURE_COUNT = 4608  
MANUAL_FEATURE_COUNT = 327 # 109 features * 3 crops = 327.

def train_from_csv():
    print("==================================================")
    print("🚀 SYSTEM V40: CSV FEEDING PIPELINE (FIXED)")
    print("==================================================")

    if not os.path.exists(DATASET_PATH):
        print(f"❌ ERROR: Cannot find '{DATASET_PATH}'.")
        return

    print(f"\n[1/5] Loading Dataset: {DATASET_PATH}...")
    df = pd.read_csv(DATASET_PATH)
    
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=[TARGET_COLUMN])
    y = df[TARGET_COLUMN].astype(int).values

    # Drop Metadata 
    cols_to_drop = [
        TARGET_COLUMN, 'Image_Name', 'Relative_Path', 'Source_Folder', 
        'Suggested_Split', 'Original_Width', 'Original_Height', 
        'Aspect_Ratio', 'File_Size_KB'
    ]
    X_df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    string_cols = X_df.select_dtypes(include=['object', 'string']).columns
    if len(string_cols) > 0:
        X_df = X_df.drop(columns=string_cols)

    X_raw = X_df.values
    print(f"   >>> Total Pure Features: {X_raw.shape[1]} (Target: 4935)")
    
    print("\n[2/5] Splitting into CNN and Manual Features...")
    cnn_features = X_raw[:, :CNN_FEATURE_COUNT]
    man_features = X_raw[:, CNN_FEATURE_COUNT:]

    print(f"   >>> Manual Features Shape: {man_features.shape[1]} (Target: 327)")

    cnn_train, cnn_test, man_train, man_test, y_train, y_test = train_test_split(
        cnn_features, man_features, y, test_size=0.2, stratify=y, random_state=SEED
    )

    print("\n[3/5] Applying EXACT Master Script Preprocessing...")
    
    # 1. Scale Manual Features (No extra imputer needed here, matching master script)
    scaler_man = StandardScaler()
    man_train_sel = scaler_man.fit_transform(man_train)
    man_test_sel = scaler_man.transform(man_test)

    # 2. Combine CNN and Scaled Manual Features
    X_train_final = np.hstack([cnn_train, man_train_sel])
    X_test_final = np.hstack([cnn_test, man_test_sel])

    # 3. Impute the Combined Array
    imputer = SimpleImputer(strategy='mean')
    X_train_final = imputer.fit_transform(X_train_final)
    X_test_final = imputer.transform(X_test_final)

    # 4. Scale the Combined Array
    scaler_final = StandardScaler()
    X_train_final = scaler_final.fit_transform(X_train_final)
    X_test_final = scaler_final.transform(X_test_final)

    print("\n[4/5] Training CatBoost Model...")
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

    print("\n[5/5] Evaluating with Security Threshold...")
    probs_test = clf.predict_proba(X_test_final)[:, 1]
    
    best_threshold = 0.5000 # Matching master script threshold
    preds_calibrated = (probs_test >= best_threshold).astype(int)

    print("\n" + "="*50)
    print("✅ SYSTEM V40 CSV TRAINING RESULTS")
    print("="*50)
    print(classification_report(y_test, preds_calibrated, target_names=["Legitimate", "Phishing"]))
    
    acc = accuracy_score(y_test, preds_calibrated)
    print(f"🏆 Final Calibrated True Accuracy: {acc * 100:.2f}%\n")
    
    # 🚨 Saving EXACTLY the 5 files the master script expects
    pickle.dump(clf, open("model_v40_csv.pkl", "wb"))
    pickle.dump(scaler_final, open("scaler40_csv.pkl", "wb"))
    pickle.dump(scaler_man, open("scaler_man_v40_csv.pkl", "wb"))
    pickle.dump(imputer, open("imputer_v40_csv.pkl", "wb"))
    pickle.dump(best_threshold, open("threshold_v40_csv.pkl", "wb"))
    print("[Done] All 5 System Models Saved Successfully (Matching Master Script).")

if __name__ == '__main__':
    train_from_csv()