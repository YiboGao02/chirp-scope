[English](./README.md)

# Chirp Scope

Chirp Scope 是一个专为在资源受限的嵌入式设备上检测鸟叫声而设计的轻量级音频分类框架。该模型兼容 Espressif 的 **ESP-DL** 推理框架。

## 应用案例
关于在 ESP32-S3 上的完整部署示例，请参考：
https://github.com/QcQuirin/esp32s3_edgeAi_Bird-Call-Detect

## 快速开始

### 演示 (Demo)
我们提供了一个简单的演示脚本，用于对原始 PCM 音频文件进行推理。

1. 确保已安装所需的依赖项：
   ```bash
   pip install -r requirements.txt
   ```

2. 运行演示脚本：
   ```bash
   python demo/demo.py
   ```

该脚本读取 `demo/input.pcm`，对其进行预处理，并输出分类概率。

## 模型详情

### 输入格式
模型接受原始音频波形。
- **格式**: 16-bit PCM (或归一化到 [-1, 1] 的 float32)。
- **采样率**: 16,000 Hz (16kHz)。
- **形状**: `(Batch, 1, Samples)`。输入长度是固定的（在配置中由 `SEGMENT_SAMPLES` 定义）。

### 输出格式
模型输出的是 **原始 logits** (未归一化的分数)，而不是概率。

> **注意:**
> 用于部署的 `espdl` 库当前版本在此特定的量化上下文中不原生支持 Softmax 层。
> **在部署代码中手动对输出 logits 应用 Softmax** 以获得概率分布（例如，总和为 1.0）。

输出示例 (Logits): `[2.5, -1.2]`  
Softmax 后: `[0.97, 0.03]`
