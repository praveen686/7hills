import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf

print("=" * 80)
print("ISOLATING ELU GPU FAILURE")
print("=" * 80)

# Build info
print(f"\nTF Version: {tf.__version__}")
build_info = tf.sysconfig.get_build_info()
print(f"CUDA: {build_info.get('cuda_version')}, cuDNN: {build_info.get('cudnn_version')}")
print(f"GPU Devices: {tf.config.list_physical_devices('GPU')}")

# Test 1: Simple eager execution
print("\n" + "=" * 80)
print("Test 1: Simple eager ELU on GPU (no @tf.function)")
print("=" * 80)
try:
    with tf.device('/GPU:0'):
        x = tf.constant([[1.0, -1.0, 2.0, -2.0]])
        result = tf.nn.elu(x)
        print(f"✅ SUCCESS: {result.numpy()}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 2: Inside tf.function (graph mode, no XLA)
print("\n" + "=" * 80)
print("Test 2: ELU inside @tf.function (graph mode, no JIT)")
print("=" * 80)
@tf.function
def elu_fn(x):
    return tf.nn.elu(x)

try:
    x = tf.constant([[1.0, -1.0, 2.0, -2.0]])
    result = elu_fn(x)
    print(f"✅ SUCCESS: {result.numpy()}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 3: With explicit XLA JIT
print("\n" + "=" * 80)
print("Test 3: ELU with EXPLICIT XLA jit_compile=True")
print("=" * 80)
@tf.function(jit_compile=True)
def elu_xla(x):
    return tf.nn.elu(x)

try:
    x = tf.constant([[1.0, -1.0, 2.0, -2.0]])
    result = elu_xla(x)
    print(f"✅ SUCCESS: {result.numpy()}")
except Exception as e:
    print(f"❌ FAILED (expected if XLA is the culprit): {e}")

# Test 4: Keras layer in eager mode
print("\n" + "=" * 80)
print("Test 4: Keras Dense + ELU in eager mode")
print("=" * 80)
try:
    layer = tf.keras.layers.Dense(4)
    x = tf.constant([[1.0, 2.0, 3.0]])
    h = layer(x)
    result = tf.nn.elu(h)
    print(f"✅ SUCCESS: shape={result.shape}")
except Exception as e:
    print(f"❌ FAILED: {e}")

# Test 5: Full GRN-like pattern
print("\n" + "=" * 80)
print("Test 5: GRN-like pattern (Dense → E LU → Dense)")
print("=" * 80)
try:
    x = tf.constant(tf.random.normal((32, 21, 320)))
    layer1 = tf.keras.layers.Dense(64)
    layer2 = tf.keras.layers.Dense(64)
    
    h = layer1(x)
    h = tf.nn.elu(h) 
    h = layer2(h)
    print(f"✅ SUCCESS: shape={h.shape}")
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 80)
