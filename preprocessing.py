import os
import wfdb
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from sklearn.model_selection import train_test_split

# Define the standard AAMI mapping
# See https://physionet.org/content/mitdb/1.0.0/ for annotation details
AAMI_MAPPING = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # Supraventricular (S)
    'V': 2, 'E': 2,                          # Ventricular (V)
    'F': 3,                                  # Fusion (F)
    '/': 4, 'f': 4, 'Q': 4                   # Unknown/Paced (Q)
}
AAMI_CLASSES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']

# Standard inter-patient splits (de Chazal et al.)
DS1 = [101, 106, 108, 109, 112, 114, 115, 116, 118, 119, 122, 124, 
       201, 203, 205, 207, 208, 209, 215, 220, 223, 230]
DS2 = [100, 103, 105, 111, 113, 117, 121, 123, 
       200, 202, 210, 212, 213, 214, 217, 219, 221, 222, 228, 231, 232, 233, 234]

WINDOW_SIZE = 252
LEFT_WINDOW = 126
RIGHT_WINDOW = 126

def load_patient_data(patient_id, db_dir):
    record_path = os.path.join(db_dir, str(patient_id))
    try:
        record = wfdb.rdrecord(record_path)
        annotation = wfdb.rdann(record_path, 'atr')
        
        # Get the signal (2 channels)
        signals = record.p_signal
        
        # Get annotations (peaks and symbols)
        peaks = annotation.sample
        symbols = annotation.symbol
        
        return signals, peaks, symbols
    except Exception as e:
        print(f"Error loading record {patient_id}: {e}")
        return None, None, None

def segment_beats(signals, peaks, symbols):
    X = []
    y = []
    
    for peak, symbol in zip(peaks, symbols):
        if symbol in AAMI_MAPPING:
            # Check if we have enough signal for the window
            if peak - LEFT_WINDOW >= 0 and peak + RIGHT_WINDOW < len(signals):
                window = signals[peak - LEFT_WINDOW : peak + RIGHT_WINDOW]
                X.append(window)
                y.append(AAMI_MAPPING[symbol])
                
    return np.array(X), np.array(y)

def process_dataset(db_dir, patient_list):
    X_all = []
    y_all = []
    
    print(f"Processing {len(patient_list)} records...")
    for pid in patient_list:
        signals, peaks, symbols = load_patient_data(pid, db_dir)
        if signals is not None:
            X_pid, y_pid = segment_beats(signals, peaks, symbols)
            X_all.append(X_pid)
            y_all.append(y_pid)
            
    X = np.concatenate(X_all, axis=0)
    y = np.concatenate(y_all, axis=0)
    
    return X, y

def plot_class_distribution(y_train, y_val, y_test, save_dir):
    plt.figure(figsize=(10, 6))
    
    counts_train = Counter(y_train)
    counts_val = Counter(y_val)
    counts_test = Counter(y_test)
    
    bar_width = 0.25
    r1 = np.arange(len(AAMI_CLASSES))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    
    plt.bar(r1, [counts_train.get(i, 0) for i in range(5)], width=bar_width, label='Train')
    plt.bar(r2, [counts_val.get(i, 0) for i in range(5)], width=bar_width, label='Validation')
    plt.bar(r3, [counts_test.get(i, 0) for i in range(5)], width=bar_width, label='Test')
    
    plt.xlabel('AAMI Classes', fontweight='bold')
    plt.ylabel('Number of Beats', fontweight='bold')
    plt.title('Class Distribution across Splits', fontweight='bold')
    plt.xticks([r + bar_width for r in range(len(AAMI_CLASSES))], AAMI_CLASSES, rotation=15)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'class_distribution.png'))
    plt.close()
    print("Saved class distribution plot to 'class_distribution.png'")

def plot_beat_examples(X, y, save_dir):
    plt.figure(figsize=(15, 10))
    
    for i in range(5):
        plt.subplot(3, 2, i+1)
        # Find all indices for class i
        idx = np.where(y == i)[0]
        if len(idx) > 0:
            # Plot up to 5 random examples of this class
            samples = np.random.choice(idx, min(5, len(idx)), replace=False)
            for s in samples:
                # Plot Channel 0
                plt.plot(X[s, :, 0], alpha=0.6)
            plt.title(f"Class: {AAMI_CLASSES[i]} (Channel 0)")
            plt.xlabel("Samples")
            plt.ylabel("Amplitude (mV)")
            
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'beat_examples.png'))
    plt.close()
    print("Saved beat examples plot to 'beat_examples.png'")

def get_patient_beat_counts(db_dir, patient_list):
    counts = {}
    for pid in patient_list:
        signals, peaks, symbols = load_patient_data(pid, db_dir)
        if signals is not None:
            # count valid beats
            valid = [s for p, s in zip(peaks, symbols) if s in AAMI_MAPPING and p - LEFT_WINDOW >= 0 and p + RIGHT_WINDOW < len(signals)]
            counts[pid] = len(valid)
    return counts

def custom_inter_patient_split(db_dir):
    all_patients = [100, 101, 103, 105, 106, 108, 109, 111, 112, 113, 114, 115, 116, 117, 118, 119, 121, 122, 123, 124, 
                    200, 201, 202, 203, 205, 207, 208, 209, 210, 212, 213, 214, 215, 217, 219, 220, 221, 222, 223, 228, 230, 231, 232, 233, 234]
    
    print("Calculating beat counts per patient...")
    counts = get_patient_beat_counts(db_dir, all_patients)
    total_beats = sum(counts.values())
    
    target_test = int(total_beats * 0.30)
    target_val = int(total_beats * 0.15)
    
    # Shuffle patients for random assignment
    np.random.seed(42)
    shuffled_patients = np.random.permutation(all_patients)
    
    test_patients, val_patients, train_patients = [], [], []
    test_beats, val_beats = 0, 0
    
    for pid in shuffled_patients:
        b_count = counts[pid]
        if test_beats < target_test:
            test_patients.append(pid)
            test_beats += b_count
        elif val_beats < target_val:
            val_patients.append(pid)
            val_beats += b_count
        else:
            train_patients.append(pid)
            
    return train_patients, val_patients, test_patients

def main():
    db_dir = os.path.join(os.getcwd(), 'mitdb_data')
    output_dir = os.path.join(os.getcwd(), 'processed_data')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print("=== Step 1: Generating Custom Inter-Patient Split (30% Test) ===")
    train_patients, val_patients, test_patients = custom_inter_patient_split(db_dir)
    
    print("=== Step 2: Processing Train Set ===")
    X_train, y_train = process_dataset(db_dir, train_patients)
    
    print("=== Step 3: Processing Validation Set ===")
    X_val, y_val = process_dataset(db_dir, val_patients)

    print("=== Step 4: Processing Test Set ===")
    X_test, y_test = process_dataset(db_dir, test_patients)
    
    print(f"Final Shapes:")
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test:  X={X_test.shape}, y={y_test.shape}")
    
    print("=== Step 5: Generating Visualizations ===")
    plot_class_distribution(y_train, y_val, y_test, output_dir)
    plot_beat_examples(X_train, y_train, output_dir)
    
    print("=== Step 6: Saving NumPy Arrays ===")
    np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
    np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
    np.save(os.path.join(output_dir, 'X_val.npy'), X_val)
    np.save(os.path.join(output_dir, 'y_val.npy'), y_val)
    np.save(os.path.join(output_dir, 'X_test.npy'), X_test)
    np.save(os.path.join(output_dir, 'y_test.npy'), y_test)
    print(f"All data successfully saved to {output_dir}")

if __name__ == "__main__":
    main()
