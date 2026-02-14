import os
import sys

# Point XLA to Triton's libdevice
triton_libdevice_path = "/home/ubuntu/Desktop/7hills/QuantKubera/qk_venv/lib/python3.12/site-packages/triton/backends/nvidia/lib"
os.environ['XLA_FLAGS'] = f'--xla_gpu_cuda_data_dir={triton_libdevice_path}'

print(f"Set XLA_FLAGS: {os.environ['XLA_FLAGS']}")

import numpy as np
import tensorflow as tf

print("\n" + "=" * 80)
print("Testing with XLA_FLAGS pointing to Triton libdevice")
print("=" * 80)

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
    print("ALL TESTS PASSED! GPU JIT issue resolved.")
    print("=" * 80)
    
except Exception as e:
    print(f"   ❌ FAILED: {e}")
    import traceback
    traceback.print_exc()
