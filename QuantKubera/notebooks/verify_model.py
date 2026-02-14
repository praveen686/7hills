import os
import sys
import numpy as np
import tensorflow as tf

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from quantkubera.models.tft import MomentumTransformer
from quantkubera.models.losses import SharpeLoss

def verify_model():
    print("1. Instantiating MomentumTransformer...")
    model = MomentumTransformer(
        time_steps=21,
        input_size=5,  # 5 features
        output_size=1, # 1 return prediction
        hidden_size=64,
        num_heads=4,
        dropout_rate=0.1
    )
    
    print("2. Creating Dummy Input...")
    # Batch=32, Time=21, Features=5
    dummy_input = np.random.randn(32, 21, 5).astype(np.float32)
    
    print("3. Forward Pass...")
    try:
        output = model(dummy_input)
        print(f"   [OK] Model Output Shape: {output.shape}")
        
        expected_shape = (32, 21, 1)
        if output.shape == expected_shape:
            print("   [OK] Shape matches expected.")
        else:
            print(f"   [FAIL] Expected {expected_shape}, got {output.shape}")
            
    except Exception as e:
        print(f"   [FAIL] Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n4. Testing Loss Function...")
    try:
        y_true = np.random.randn(32, 21, 1).astype(np.float32)
        loss_fn = SharpeLoss()
        loss = loss_fn(y_true, output)
        print(f"   [OK] Loss calculated: {loss.numpy()}")
    except Exception as e:
        print(f"   [FAIL] Loss calculation failed: {e}")

if __name__ == "__main__":
    verify_model()
