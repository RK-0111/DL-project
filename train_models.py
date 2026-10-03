import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, average_precision_score

AAMI_CLASSES = ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', 'Fusion (F)', 'Unknown (Q)']

# --- MODEL DEFINITIONS ---

def build_mlp(input_shape=(252, 2), num_classes=5):
    inputs = layers.Input(shape=input_shape)
    x = layers.Flatten()(inputs)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name="MLP_Baseline")

def build_resnet1d(input_shape=(252, 2), num_classes=5):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv1D(64, 7, padding='same', activation='relu')(inputs)
    x = layers.BatchNormalization()(x)
    
    # ResNet Block 1
    res = layers.Conv1D(64, 5, padding='same', activation='relu')(x)
    res = layers.BatchNormalization()(res)
    res = layers.Conv1D(64, 5, padding='same')(res)
    x = layers.Add()([x, res])
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling1D(2)(x)
    
    # ResNet Block 2
    x2 = layers.Conv1D(128, 1, strides=2, padding='same')(x) # Projection
    res = layers.Conv1D(128, 5, padding='same', activation='relu')(x)
    res = layers.BatchNormalization()(res)
    res = layers.Conv1D(128, 5, padding='same')(res)
    res = layers.MaxPooling1D(2, padding='same')(res)
    x = layers.Add()([x2, res])
    x = layers.Activation('relu')(x)
    x = layers.GlobalAveragePooling1D()(x)
    
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name="ResNet-1D")

def build_cnn_lstm(input_shape=(252, 2), num_classes=5):
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv1D(64, 5, activation='relu')(inputs)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(128, 5, activation='relu')(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(64)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name="CNN-LSTM_Hybrid")

def build_transformer(input_shape=(252, 2), num_classes=5):
    inputs = layers.Input(shape=input_shape)
    # Patch embedding via Conv1D
    x = layers.Conv1D(64, kernel_size=4, strides=4, padding="same")(inputs)
    
    # Self-attention Block
    attn_output = layers.MultiHeadAttention(num_heads=4, key_dim=64)(x, x)
    x = layers.Add()([x, attn_output])
    x = layers.LayerNormalization()(x)
    
    # Feed Forward Block
    ffn_output = layers.Dense(64, activation='relu')(x)
    x = layers.Add()([x, ffn_output])
    x = layers.LayerNormalization()(x)
    
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    return models.Model(inputs, outputs, name="1D_Transformer")


# --- EVALUATION AND PLOTTING ---

def evaluate_and_plot(model, history, X_test, y_test, results_dir):
    model_name = model.name
    model_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\nEvaluating {model_name}...")
    
    # 1. Train/Val Curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title(f'{model_name} Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f'{model_name} Loss')
    plt.legend()
    plt.savefig(os.path.join(model_dir, 'train_val_curves.png'))
    plt.close()
    
    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred_classes = np.argmax(y_pred_probs, axis=1)
    
    # 2. Classification Report (Precision, Recall, F1)
    report = classification_report(y_test, y_pred_classes, target_names=AAMI_CLASSES, output_dict=True)
    
    # Extract Sensitivity (Recall) and Specificity
    cm = confusion_matrix(y_test, y_pred_classes)
    metrics_text = f"--- {model_name} Final Metrics ---\n\n"
    for i, class_name in enumerate(AAMI_CLASSES):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        fp = np.sum(cm[:, i]) - tp
        tn = np.sum(cm) - (tp + fn + fp)
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        metrics_text += f"Class: {class_name}\n"
        metrics_text += f"Sensitivity (Recall): {sensitivity:.4f}\n"
        metrics_text += f"Specificity: {specificity:.4f}\n"
        metrics_text += f"Precision: {report[class_name]['precision']:.4f}\n"
        metrics_text += f"F1-Score: {report[class_name]['f1-score']:.4f}\n\n"
        
    metrics_text += f"Overall Accuracy: {report['accuracy']:.4f}\n"
    
    with open(os.path.join(model_dir, 'metrics_report.txt'), 'w') as f:
        f.write(metrics_text)
        
    # 3. Confusion Matrix Plot
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=AAMI_CLASSES, yticklabels=AAMI_CLASSES)
    plt.title(f'{model_name} Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
    plt.close()
    
    # 4. Precision-Recall Curves (PRC)
    plt.figure(figsize=(10, 8))
    for i, class_name in enumerate(AAMI_CLASSES):
        y_true_binary = (y_test == i).astype(int)
        y_pred_prob_class = y_pred_probs[:, i]
        precision, recall, _ = precision_recall_curve(y_true_binary, y_pred_prob_class)
        ap = average_precision_score(y_true_binary, y_pred_prob_class)
        plt.plot(recall, precision, label=f'{class_name} (AP = {ap:.2f})')
        
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'{model_name} Precision-Recall Curve (PRC)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'prc_curve.png'))
    plt.close()
    
    print(f"Finished evaluating {model_name}. Results saved to {model_dir}")


def main():
    print("Loading datasets...")
    data_dir = os.path.join(os.getcwd(), 'processed_data')
    results_dir = os.path.join(os.getcwd(), 'results')
    os.makedirs(results_dir, exist_ok=True)
    
    X_train = np.load(os.path.join(data_dir, 'X_train_smote.npy'))
    y_train = np.load(os.path.join(data_dir, 'y_train_smote.npy'))
    X_val = np.load(os.path.join(data_dir, 'X_val.npy'))
    y_val = np.load(os.path.join(data_dir, 'y_val.npy'))
    X_test = np.load(os.path.join(data_dir, 'X_test.npy'))
    y_test = np.load(os.path.join(data_dir, 'y_test.npy'))
    
    # Ensure memory doesn't blow up on CPU/GPU, we use batch size 256
    batch_size = 256
    epochs = 10 # Keep small so it finishes in a reasonable timeframe
    
    # Define models
    models_to_train = [
        build_mlp(),
        build_resnet1d(),
        build_cnn_lstm(),
        build_transformer()
    ]
    
    for model in models_to_train:
        print(f"\n{'='*50}\nTraining {model.name}\n{'='*50}")
        model.compile(optimizer='adam', 
                      loss='sparse_categorical_crossentropy', 
                      metrics=['accuracy'])
        
        # Early stopping callback
        es = callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            batch_size=batch_size,
            epochs=epochs,
            callbacks=[es],
            verbose=2
        )
        
        evaluate_and_plot(model, history, X_test, y_test, results_dir)

if __name__ == "__main__":
    main()
