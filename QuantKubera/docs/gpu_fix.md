# GPU JIT Compilation Fix

## Problem
TensorFlow 2.19.0 with CUDA 12.5.1 failed with "JIT compilation failed" error when executing `Elu` operations after Keras layer operations on GPU.

## Root Cause
TensorFlow's XLA GPU JIT compiler could not find `libdevice.10.bc` file, which is required for compiling GPU kernels. The system CUDA installation (12.0) was incomplete and missing this file.

## Diagnosis
- System CUDA: 12.0 (incomplete, missing libdevice files)
- TensorFlow built with: CUDA 12.5.1 / cuDNN 9  
- `libdevice.10.bc` found only in: `qk_venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/`
- TensorFlow searched for it in standard CUDA paths and current directory

## Solution (System-Level Fix)
Copied `libdevice.10.bc` from Triton package to system CUDA directory:

```bash
sudo mkdir -p /usr/local/cuda/nvvm/libdevice
sudo cp qk_venv/lib/python3.12/site-packages/triton/backends/nvidia/lib/libdevice.10.bc /usr/local/cuda/nvvm/libdevice/
```

## Verification
- Model instantiation: ✅ SUCCESS
- Forward pass on GPU: ✅ SUCCESS
- cuDNN 9 loaded correctly
- No JIT compilation errors

## Notes
- This is a **permanent, system-wide fix** that survives venv recreation
- Alternative workarounds (symlinks, environment variables) are not sustainable
- The Triton package bundles libdevice files that are compatible with TensorFlow's CUDA version
