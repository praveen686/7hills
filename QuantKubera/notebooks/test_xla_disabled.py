import os
import sys

# Before importing TensorFlow, ensure all XLA paths are disabled
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=-1'  # Disable autoclustering
os.environ.pop('TF_XLA_FLAGS', None)  # Or just remove it entirely

import numpy as np
import tensorflow as tf

print("=" * 80)
print("COMPREHENSIVE XLA DIAGNOSTIC")
print("=" * 80)

# 1. Check TensorFlow build info
print("\n1. TensorFlow Build Info:")
print(f"   TF Version: {tf.__version__}")
if hasattr(tf.sysconfig, 'get_build_info'):
    build_info = tf.sysconfig.get_build_info()
    print(f"   Built with CUDA: {build_info.get('cuda_version', 'N/A')}")
    print(f"   Built with cuDNN: {build_info.get('cudnn_version', 'N/A')}")

# 2. Check XLA status BEFORE disabling
print("\n2. Initial XLA Status:")
print(f"   XLA JIT (autoclustering): {tf.config.optimizer.get_jit()}")

# 3. Disable XLA completely
print("\n3. Disabling XLA JIT...")
tf.config.optimizer.set_jit(False)
print(f"   XLA JIT after disable: {tf.config.optimizer.get_jit()}")

# 4. Check environment variables
print("\n4. Environment Variables:")
print(f"   TF_XLA_FLAGS: {os.environ.get('TF_XLA_FLAGS', 'Not set')}")

# 5. Try the model
print("\n5. Testing Model with XLA Disabled:")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

try:
    from quantkubera.models.tft import MomentumTransformer
    from quantkubera.models.losses import SharpeLoss
    
    model = MomentumTransformer(
        time_steps=21,
        input_size=5,
        output_size=1,
        hidden_size=64,
        num_heads=4,
        dropout_rate=0.1
    )
    
    dummy_input = np.random.randn(32, 21, 5).astype(np.float32)
    
    print("   Running forward pass...")
    output = model(dummy_input)
    print(f"   ✅ SUCCESS! Output shape: {output.shape}")
    
    # Test loss
    y_true = np.random.randn(32, 21, 1).astype(np.float32)
    loss_fn = SharpeLoss()
    loss = loss_fn(y_true, output)
    print(f"   ✅ Loss calculated: {loss.numpy()}")
    
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    
print("\n" + "=" * 80)
