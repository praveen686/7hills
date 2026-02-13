# EC2 Training Guide — QuantLaxmi

This document details how to connect to the AWS GPU instance and launch the TFT/X-Trend training pipeline.

## 1. Instance Details (ap-south-1)
- **ID**: `i-07acb39828993e7a8` (aysola-ec2)
- **Type**: `g4dn.8xlarge` (Tesla T4 GPU, 32 vCPUs)
- **Public IP**: `3.6.92.37`
- **User**: `ubuntu`
- **SSH Key**: `aysola-aws-key.pem`

## 2. Connection
Use the PEM key stored in `~/Desktop/aws_stuff/` (local machine).

```bash
ssh -i /path/to/aysola-aws-key.pem ubuntu@3.6.92.37
```

## 3. Environment & Workspace
- **Codebase Path**: `/home/ubuntu/Desktop/7hills/QuantLaxmi/`
- **Virtual Env**: `/home/ubuntu/venvs/pt/` (PyTorch + CUDA ready)

## 4. Running Training
The production training pipeline is located at `quantlaxmi/models/ml/tft/production/training_pipeline.py`.

```bash
# Connect and set PYTHONPATH
export PYTHONPATH=/home/ubuntu/Desktop/7hills/QuantLaxmi

# Activate environment and run
source /home/ubuntu/venvs/pt/bin/activate
python /home/ubuntu/Desktop/7hills/QuantLaxmi/quantlaxmi/models/ml/tft/production/training_pipeline.py \
    --start 2024-01-01 \
    --end 2026-02-12
```

## 5. Monitoring
- **GPU Usage**: `nvidia-smi -l 1`
- **Training Logs**: Training scripts use `tqdm` and local logging. Check the production results in `~/Desktop/7hills/QuantLaxmi/checkpoints/`.
