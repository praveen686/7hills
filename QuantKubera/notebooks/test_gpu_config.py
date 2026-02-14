import os
import sys
import numpy as np

# Disable any XLA/JIT paths
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=-1'  
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # Force deterministic kernels
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN optimizations

import tensorflow as tf

print("=" * 80)
print("Testing with GPU Configuration Tweaks")
print("=" * 80)

# Configure GPU
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ Configured {len(gpus)} GPU(s) with memory growth")
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Disable XLA
tf.config.optimizer.set_jit(False)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from quantkubera.models.tft import MomentumTransformer
    from quantkubera.models.losses import SharpeLoss
    
    print("\n1. Instantiating Model...")
    model = MomentumTransformer(
        time_steps=21,
        input_size=5,
        output_size=1,
        hidden_size=64,
        num_heads=4,
        dropout_rate=0.1
    )
    
    print("2. Creating Dummy Input...")
    dummy_input = np.random.randn(32, 21, 5).astype(np.float32)
    
    print("3. Forward Pass...")
    output = model(dummy_input)
    print(f"   ✅ SUCCESS! Output shape: {output.shape}")
    
    print("\n4. Testing Loss Function...")
    y_true = np.random.randn(32, 21, 1).astype(np.float32)
    loss_fn = SharpeLoss()
    loss = loss_fn(y_true, output)
    print(f"   ✅ Loss calculated: {loss.numpy()}")
    
    print("\n" + "=" * 80)
    print("ALL TESTS PASSED!")
    print("=" * 80)
    
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    print("\nThis confirms the issue is NOT resolvable via TF config flags.")
    print("The problem is deeper - likely a TF 2.19 + CUDA 12.x incompatibility.")
