[中文](./README_zh.md)

# Chirp Scope

Chirp Scope is a lightweight audio classification framework designed for detecting bird calls on resource-constrained embedded devices. The model is compatible with Espressif's **ESP-DL** inference framework.

## Application
For a complete deployment example on ESP32-S3, please refer to:
https://github.com/QcQuirin/esp32s3_edgeAi_Bird-Call-Detect

## Quick Start

### Demo
We provide a simple demo script to infer on a raw PCM audio file.

1. Ensure you have the required dependencies installed:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the demo script:
   ```bash
   python demo/demo.py
   ```

The script reads `demo/input.pcm`, preprocesses it, and outputs the classification probabilities.

## Model Details

### Input Format
The model accepts raw audio waveforms.
- **Format**: 16-bit PCM (or float32 normalized to [-1, 1]).
- **Sample Rate**: 16,000 Hz (16kHz).
- **Shape**: `(Batch, 1, Samples)`. The input length is fixed (defined by `SEGMENT_SAMPLES` in config).

### Output Format (Important!)
The model outputs **raw logits** (unnormalized scores), not probabilities.

> **Note for Deployment:**
> The current version of the `espdl` library used for deployment does not natively support the Softmax layer in this specific quantization context.
> **Please manually apply Softmax to the output logits in your deployment code** to get probability distributions (e.g., summing to 1.0).

Example output (Logits): `[2.5, -1.2]`  
After Softmax: `[0.97, 0.03]`
