/*
 * ZeroDefect — AIM (Aerospace Inspection Module) Inference + Web Server
 * 
 * HARDWARE REQUIRED:
 * - ZeroDefect AIM (Aerospace Inspection Module) board
 * - OV2640 camera module (included)
 * - 5V/2A power supply
 * 
 * SETUP:
 * - Flash using Arduino IDE with ESP32 board support (AIM is ESP32-CAM compatible)
 * - Set Wi-Fi credentials below
 * - See aim/README.md for full setup instructions
 * 
 * HARDWARE_VALIDATION_REQUIRED:
 * All sections marked with this comment need physical board testing.
 */

#include "esp_camera.h"
#include <WiFi.h>
#include "esp_http_server.h"

// Wi-Fi Config - HARDWARE_VALIDATION_REQUIRED: Edit to match local network
const char* ssid = "ZeroDefect_AIM_AP";
const char* password = "password123";

// AI-Thinker Camera Pins Config
#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27
#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

// Board LED Indicators
#define FLASH_LED_GPIO     4  // Bright white flash LED
#define STATUS_LED_GPIO   33  // Red status LED (active LOW on AI-Thinker)

// Global qc state placeholders
unsigned long inspection_count = 0;
float latest_score = 0.124;
float threshold = 0.450;
String latest_decision = "OK";
String latest_defect = "none";
float latest_confidence = 0.942;
unsigned long start_time_ms = 0;

// HTTP Server instance
httpd_handle_t server = NULL;

// ── Inference Mock - HARDWARE_VALIDATION_REQUIRED ──────────────────────────
float runInference(uint8_t* imageBuffer, size_t len) {
    /* 
     * TODO: Integrate TensorFlow Lite Micro feature extractor here.
     * 1. Load INT8 quantized weights of ResNet18 backbone
     * 2. Feed the decoded JPEG image buffer (resized to 256x256)
     * 3. Extract embedding patches
     * 4. Perform nearest neighbor search in memory bank binary file (stored in flash/SPIFFS)
     * 5. Return calculated 99th percentile distance
     */
    
    // Placeholder logic: generate a value slightly varying around threshold
    // simulating real part inspections
    inspection_count++;
    float randomVal = (float)(rand() % 100) / 1000.0;
    
    if (rand() % 10 == 0) {  // 10% chance of mock anomaly
        latest_decision = "DEFECT";
        latest_defect = (rand() % 2 == 0) ? "crack" : "scratch";
        latest_score = threshold + randomVal + 0.05;
        latest_confidence = 0.70 + ((float)(rand() % 30) / 100.0);
    } else {
        latest_decision = "OK";
        latest_defect = "none";
        latest_score = threshold - 0.1 - randomVal;
        latest_confidence = 0.85 + ((float)(rand() % 15) / 100.0);
    }
    
    return latest_score;
}

// ── Web Page Source ──────────────────────────────────────────────────────────
// Served directly from ESP32 FLASH program memory
const char index_html[] PROGMEM = R"rawliteral(
<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>ZeroDefect — AIM Live QC Node</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body { font-family: -apple-system, sans-serif; background: #0c101f; color: #e2e8f0; margin: 0; padding: 20px; text-align: center; }
    .card { background: #151d30; border-radius: 12px; padding: 24px; max-width: 500px; margin: 20px auto; box-shadow: 0 4px 20px rgba(0,0,0,0.4); border: 1px solid rgba(255,255,255,0.05); }
    h1 { color: #00f5a0; font-size: 24px; margin-bottom: 5px; }
    h2 { font-size: 32px; font-weight: 800; margin: 15px 0; }
    .ok { color: #00f5a0; }
    .defect { color: #ff4444; }
    .stat-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid rgba(255,255,255,0.05); font-size: 14px; }
    .btn { background: #00f5a0; color: #0c101f; border: none; padding: 12px 24px; border-radius: 6px; font-weight: bold; cursor: pointer; width: 100%; margin-top: 15px; }
    .footer { font-size: 11px; color: #64748b; margin-top: 40px; }
  </style>
</head>
<body>
  <div class="card">
    <h1>ZeroDefect</h1>
    <div style="font-size:12px; color:#64748b;">AIM Edge Inspection Node</div>
    <hr style="border:0; border-top: 1px solid rgba(255,255,255,0.08); margin: 15px 0;">
    
    <div id="decision-box">
      <h2>Loading...</h2>
    </div>
    
    <div class="stat-row"><span>Anomaly Score</span><span id="score" class="ok">—</span></div>
    <div class="stat-row"><span>Confidence</span><span id="conf">—</span></div>
    <div class="stat-row"><span>Total Checked</span><span id="count">0</span></div>
    <div class="stat-row"><span>Uptime</span><span id="uptime">0s</span></div>
    
    <button class="btn" onclick="triggerInspection()">Trigger Capture</button>
  </div>
  <div class="footer">Tata Technologies InnoVent Track 3.2.3.4</div>

  <script>
    function updateDOM(data) {
      const db = document.getElementById("decision-box");
      const isDefect = data.decision === "DEFECT";
      
      if (isDefect) {
        db.innerHTML = `<h2 class="defect">✗ DEFECT: ${data.defect_type.toUpperCase()}</h2>`;
      } else {
        db.innerHTML = `<h2 class="ok">✓ OK</h2>`;
      }
      
      document.getElementById("score").className = isDefect ? "defect" : "ok";
      document.getElementById("score").innerText = data.score.toFixed(4);
      document.getElementById("conf").innerText = (data.confidence * 100).toFixed(1) + "%";
      document.getElementById("count").innerText = data.inspections_count;
      document.getElementById("uptime").innerText = data.uptime_s + "s";
    }

    function triggerInspection() {
      fetch('/capture')
        .then(r => r.json())
        .then(updateDOM);
    }

    // Auto-refresh loop
    setInterval(() => {
      fetch('/api/status')
        .then(r => r.json())
        .then(updateDOM);
    }, 2000);
  </script>
</body>
</html>
)rawliteral";

// ── HTTP Endpoint Handlers ──────────────────────────────────────────────────
esp_err_t get_index_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "text/html");
    return httpd_resp_send(req, index_html, strlen(index_html));
}

esp_err_t get_status_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "application/json");
    char json_response[256];
    unsigned long uptime = (millis() - start_time_ms) / 1000;
    
    snprintf(json_response, sizeof(json_response),
             "{\"decision\":\"%s\",\"defect_type\":\"%s\",\"confidence\":%.3f,\"score\":%.3f,\"uptime_s\":%lu,\"inspections_count\":%lu}",
             latest_decision.c_str(), latest_defect.c_str(), latest_confidence, latest_score, uptime, inspection_count);
             
    return httpd_resp_send(req, json_response, strlen(json_response));
}

esp_err_t get_capture_handler(httpd_req_t *req) {
    camera_fb_t * fb = NULL;
    fb = esp_camera_fb_get();
    if (!fb) {
        Serial.println("Camera capture failed");
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    // Toggle flash briefly during exposure - HARDWARE_VALIDATION_REQUIRED
    digitalWrite(FLASH_LED_GPIO, HIGH);
    delay(100);
    digitalWrite(FLASH_LED_GPIO, LOW);

    // Call mock/real inference on frame
    float score = runInference(fb->buf, fb->len);
    esp_camera_fb_return(fb);

    // Toggle onboard red status LED (low is ON)
    if (latest_decision == "DEFECT") {
        digitalWrite(STATUS_LED_GPIO, LOW);  // Turn LED ON (RED)
    } else {
        digitalWrite(STATUS_LED_GPIO, HIGH); // Turn LED OFF
    }

    return get_status_handler(req);
}

// ── Web Server Setup ────────────────────────────────────────────────────────
void startCameraServer() {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.server_port = 80;

    httpd_uri_t index_uri = {
        .uri       = "/",
        .method    = HTTP_GET,
        .handler   = get_index_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t status_uri = {
        .uri       = "/api/status",
        .method    = HTTP_GET,
        .handler   = get_status_handler,
        .user_ctx  = NULL
    };

    httpd_uri_t capture_uri = {
        .uri       = "/capture",
        .method    = HTTP_GET,
        .handler   = get_capture_handler,
        .user_ctx  = NULL
    };

    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_register_uri_handler(server, &index_uri);
        httpd_register_uri_handler(server, &status_uri);
        httpd_register_uri_handler(server, &capture_uri);
        Serial.println("[*] HTTP Server started successfully on port 80.");
    }
}

// ── Initialization Setup ────────────────────────────────────────────────────
void setup() {
    Serial.begin(115200);
    Serial.println("\n[*] Initializing ZeroDefect AIM Node...");

    start_time_ms = millis();
    pinMode(FLASH_LED_GPIO, OUTPUT);
    pinMode(STATUS_LED_GPIO, OUTPUT);
    digitalWrite(FLASH_LED_GPIO, LOW);
    digitalWrite(STATUS_LED_GPIO, HIGH); // Off initially

    // Camera Configuration - HARDWARE_VALIDATION_REQUIRED
    camera_config_t config;
    config.ledc_channel = LEDC_CHANNEL_0;
    config.ledc_timer = LEDC_TIMER_0;
    config.pin_d0 = Y2_GPIO_NUM;
    config.pin_d1 = Y3_GPIO_NUM;
    config.pin_d2 = Y4_GPIO_NUM;
    config.pin_d3 = Y5_GPIO_NUM;
    config.pin_d4 = Y6_GPIO_NUM;
    config.pin_d5 = Y7_GPIO_NUM;
    config.pin_d6 = Y8_GPIO_NUM;
    config.pin_d7 = Y9_GPIO_NUM;
    config.pin_xclk = XCLK_GPIO_NUM;
    config.pin_pclk = PCLK_GPIO_NUM;
    config.pin_vsync = VSYNC_GPIO_NUM;
    config.pin_href = HREF_GPIO_NUM;
    config.pin_sscb_sda = SIOD_GPIO_NUM;
    config.pin_sscb_scl = SIOC_GPIO_NUM;
    config.pin_pwdn = PWDN_GPIO_NUM;
    config.pin_reset = RESET_GPIO_NUM;
    config.xclk_freq_hz = 20000000;
    config.pixel_format = PIXFORMAT_JPEG;

    // Allocate memory budget
    if(psramFound()){
        config.frame_size = FRAMESIZE_UXGA;
        config.jpeg_quality = 10;
        config.fb_count = 2;
    } else {
        config.frame_size = FRAMESIZE_SVGA;
        config.jpeg_quality = 12;
        config.fb_count = 1;
    }

    // Init Camera
    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        Serial.printf("[ERROR] Camera init failed with error 0x%x\n", err);
        return;
    }
    Serial.println("[✓] OV2640 camera sensor mounted successfully.");

    // Connect to Wi-Fi - HARDWARE_VALIDATION_REQUIRED
    Serial.printf("[*] Connecting to Wi-Fi SSID: %s\n", ssid);
    WiFi.begin(ssid, password);
    while (WiFi.status() != WL_CONNECTED) {
        delay(500);
        Serial.print(".");
    }
    Serial.println("");
    Serial.println("[✓] Wi-Fi network link established.");
    Serial.print("[*] Local Node IP Address: http://");
    Serial.println(WiFi.localIP());

    // Start Web Server
    startCameraServer();
}

void loop() {
    // Keep thread loop light
    delay(100);
}
