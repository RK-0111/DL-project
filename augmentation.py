import os
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE

AAMI_CLASSES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']

def plot_smote_comparison(y_before, y_after, save_dir):
    plt.figure(figsize=(12, 6))
    
    counts_before = Counter(y_before)
    counts_after = Counter(y_after)
    
    bar_width = 0.35
    r1 = np.arange(len(AAMI_CLASSES))
    r2 = [x + bar_width for x in r1]
    
    plt.bar(r1, [counts_before.get(i, 0) for i in range(5)], width=bar_width, label='Before SMOTE')
    plt.bar(r2, [counts_after.get(i, 0) for i in range(5)], width=bar_width, label='After SMOTE')
    
    plt.xlabel('AAMI Classes', fontweight='bold')
    plt.ylabel('Number of Beats', fontweight='bold')
    plt.title('Training Class Distribution (Before vs After SMOTE)', fontweight='bold')
    plt.xticks([r + bar_width/2 for r in range(len(AAMI_CLASSES))], AAMI_CLASSES, rotation=15)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'smote_distribution.png'))
    plt.close()
    print("Saved SMOTE distribution plot to 'smote_distribution.png'")

def main():
    data_dir = os.path.join(os.getcwd(), 'processed_data')
    
    print("Loading original training data...")
    X_train = np.load(os.path.join(data_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train.npy'))
    
    print(f"Original X_train shape: {X_train.shape}")
    
    # SMOTE expects 2D data (samples, features)
    # Our data is 3D (samples, 252, 2)
    n_samples, n_timesteps, n_channels = X_train.shape
    X_train_2d = X_train.reshape(n_samples, n_timesteps * n_channels)
    
    print("Applying SMOTE...")
    smote = SMOTE(random_state=42)
    X_train_resampled_2d, y_train_resampled = smote.fit_resample(X_train_2d, y_train)
    
    # Reshape back to 3D
    X_train_resampled = X_train_resampled_2d.reshape(-1, n_timesteps, n_channels)
    
    print(f"Resampled X_train shape: {X_train_resampled.shape}")
    
    print("Generating visualizations...")
    plot_smote_comparison(y_train, y_train_resampled, data_dir)
    
    print("Saving augmented data...")
    np.save(os.path.join(data_dir, 'X_train_smote.npy'), X_train_resampled)
    np.save(os.path.join(data_dir, 'y_train_smote.npy'), y_train_resampled)
    print("Data saved successfully.")

if __name__ == "__main__":
    main()
