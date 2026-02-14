import os
import sys
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Add src and config to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from quantkubera.models.losses import SharpeLoss
from quantkubera.models.tft import MomentumTransformer
from config.train_config import DATA_CONFIG

def analyze_importance(model_path, data_path='data/raw/NIFTY.csv'):
    """Calculate and print average feature importance from VSN weights."""
    print(f"Loading model from {model_path}...")
    
    # Load model with custom objects
    model = keras.models.load_model(
        model_path,
        custom_objects={'SharpeLoss': SharpeLoss, 'MomentumTransformer': MomentumTransformer}
    )
    
    # Generate some dummy data or load real data to get feature names
    feature_cols = DATA_CONFIG['feature_cols']
    window_size = DATA_CONFIG['window_size']
    
    # Create a dummy batch to get weights
    # (batch, time, features)
    dummy_input = tf.random.normal((100, window_size, len(feature_cols)))
    
    print("Extracting VSN weights...")
    _, weights = model(dummy_input, return_weights=True)
    vsn_weights = weights['vsn_weights'].numpy() # (batch, time, num_features)
    
    # Average across batch and time
    avg_importance = vsn_weights.mean(axis=(0, 1))
    
    # Combine with feature names
    importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': avg_importance
    }).sort_values('Importance', ascending=False)
    
    print("\n" + "="*40)
    print("FEATURE IMPORTANCE (VSN WEIGHTS)")
    print("="*40)
    print(importance_df.to_string(index=False))
    print("="*40)
    
    return importance_df

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='models/master_model_v1/best_model.keras')
    args = parser.parse_args()
    
    if os.path.exists(args.model):
        analyze_importance(args.model)
    else:
        print(f"Model not found at {args.model}. Still training?")
