import os
import sys
import numpy as np

# Test 1: Check XLA JIT status
print("=" * 60)
print("Test 1: XLA JIT Configuration")
print("=" * 60)
import tensorflow as tf
print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {tf.config.list_physical_devices('GPU')}")
print(f"XLA JIT enabled: {tf.config.optimizer.get_jit()}")

# Test 2: Try simple Elu operation
print("\n" + "=" * 60)
print("Test 2: Simple Elu Operation on GPU")
print("=" * 60)
try:
    with tf.device('/GPU:0'):
        x = tf.constant([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
        result = tf.nn.elu(x)
        print(f"Simple Elu SUCCESS: {result.numpy()}")
except Exception as e:
    print(f"Simple Elu FAILED: {e}")

# Test 3: Try Elu via Activation layer
print("\n" + "=" * 60)
print("Test 3: Elu via Activation Layer")
print("=" * 60)
try:
    elu_layer = tf.keras.layers.Activation("elu")
    x = tf.constant([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    result = elu_layer(x)
    print(f"Activation('elu') SUCCESS: {result.numpy()}")
except Exception as e:
    print(f"Activation('elu') FAILED: {e}")

# Test 4: Disable XLA and retry
print("\n" + "=" * 60)
print("Test 4: Elu with XLA Disabled")
print("=" * 60)
tf.config.optimizer.set_jit(False)
print(f"XLA JIT after disable: {tf.config.optimizer.get_jit()}")
try:
    elu_layer = tf.keras.layers.Activation("elu")
    x = tf.constant([[1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]])
    result = elu_layer(x)
    print(f"Activation('elu') with XLA off SUCCESS: {result.numpy()}")
except Exception as e:
    print(f"Activation('elu') with XLA off FAILED: {e}")

# Test 5: Check CUDA/cuDNN versions
print("\n" + "=" * 60)
print("Test 5: CUDA/cuDNN Information")
print("=" * 60)
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
if hasattr(tf.sysconfig, 'get_build_info'):
    build_info = tf.sysconfig.get_build_info()
    print(f"CUDA version: {build_info.get('cuda_version', 'N/A')}")
    print(f"cuDNN version: {build_info.get('cudnn_version', 'N/A')}")
