# ZeroDefect — Phase 2: ESP32-CAM Edge Deployment

## Required Hardware
| Component | Part # / Notes |
|---|---|
| ESP32-CAM Board | AI-Thinker ESP32-CAM (camera included) |
| USB-to-TTL Adapter | CP2102 or CH340 (for flashing) |
| Jumper wires | For IO0 flash mode |
| 5V Power Supply | 2A minimum |

## Why ESP32-CAM?
The AI-Thinker ESP32-CAM integrates a 2MP OV2640 camera, 4MB PSRAM, and Wi-Fi/BT in a ~$5 module. It runs Arduino/ESP-IDF or MicroPython and hosts a TCP web server — making it the easiest path to a standalone inspection unit.

## Architecture Choices

### Option A: Full On-Device Inference (Target)
- Feature extractor compressed to TFLite Micro (INT8)
- Memory bank stored as binary file in SPIFFS/LittleFS flash
- Nearest-neighbor search done on-device
- Inference time: ~300-800ms per frame (estimated)

### Option B: Edge-Assist (Fallback)
- ESP32-CAM captures frame, sends JPEG over Wi-Fi to laptop
- Laptop runs inference, returns decision to ESP32
- ESP32 displays result on web page
- Use this if quantized model doesn't fit in ESP32 memory

## Model Conversion Pipeline
```
Phase 1 (PyTorch) → ONNX → TFLite INT8 → ESP32 Flash
```

Run conversion:
```bash
# Step 1: Export to ONNX
python esp32/quantize_model.py --export-onnx

# Step 2: Convert ONNX → TFLite (requires tensorflow)
pip install tensorflow
python esp32/quantize_model.py --export-tflite

# Step 3: Report memory budget
python esp32/quantize_model.py --report
```

## ⚠️ HARDWARE TESTING REQUIRED
> The following steps require physical access to an ESP32-CAM board and CANNOT be validated in software only:
> - Flash mode wiring (IO0 to GND during upload)
> - Camera initialization (OV2640 I2C config)
> - PSRAM detection and allocation
> - SPIFFS partition mount
> - Wi-Fi AP or STA mode connectivity
> - Web server response times under real network conditions

## Flashing Instructions
1. Connect ESP32-CAM to USB-TTL: RX→TX, TX→RX, GND→GND, 5V→5V
2. Bridge IO0 to GND (flash mode)
3. In Arduino IDE: Tools → Board → ESP32 → AI Thinker ESP32-CAM
4. Upload `esp32/firmware/ZeroDefect_ESP32.ino`
5. Remove IO0→GND bridge, press reset
6. Open Serial Monitor at 115200 baud to see IP address
7. Navigate to `http://<ip-address>/` on any browser on same network

## Wiring Diagram
```
ESP32-CAM    USB-TTL
GND    ────── GND
5V     ────── 5V
U0R    ────── TX
U0T    ────── RX
IO0    ────── GND (during flash only — remove after flashing!)
```

## Memory Budget (Approximate)
| Component | Size |
|---|---|
| ResNet18 full (FP32) | ~45 MB → TOO LARGE |
| ResNet18 layer2+3 only (FP32) | ~8 MB → likely too large |
| ResNet18 layer2+3 only (INT8) | ~2 MB → borderline |
| Memory bank (10K patches × 384 dim, FP16) | ~7.5 MB → use PSRAM |
| Memory bank (1K patches × 384 dim, INT8) | ~384 KB → fits in PSRAM |
| **Recommended: reduce coreset to 1K patches** | |

For the competition, use a smaller coreset: retrain with `--coreset-ratio 0.005` and `--max-coreset 1000`.

## Fallback Plan
If the quantized model does not fit: implement Option B (edge-assist). The laptop runs `src/live_demo.py` which also serves an HTTP endpoint. ESP32-CAM sends frames to `http://laptop-ip:8000/api/infer` and displays the result.
