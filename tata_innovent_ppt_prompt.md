# Prompt for Tata InnoVent Presentation Generation

Copy and paste the prompt below into an LLM (like ChatGPT, Gemini, or Claude) to generate a complete, rule-compliant presentation script for your Tata InnoVent submission.

***

**System Role:**
You are an expert technical presentation designer and pitch strategist. Your task is to generate a comprehensive slide-by-slide outline and speaker script for a pitch deck targeting the **Tata Technologies InnoVent 2026-27** competition.

**Project Context (ZeroDefect):**
*   **Track:** 3.2.3.4: Intelligent Inspection & Defect Detection.
*   **Core Concept:** An Edge AI-powered, offline-capable computer vision defect scanner designed for aircraft surface inspection on the hangar floor, fused with a tamper-evident, hash-chained maintenance ledger.
*   **Technical Stack:** YOLOv5 Nano object detection model (trained on 22k+ images), deployed on edge devices (Luckfox Pico Max RV1106 NPU / AIM - Aerospace Inspection Module). Uses ONNX (7.2MB) and RKNN (2.6MB) formats. Backend uses FastAPI, SQLite, and Python.
*   **Key Features:** Detects 7 defect classes (crack, dent, corrosion, scratch, paint-peel, missing-head, generic defect) with real-time bounding boxes. Runs 100% locally (no cloud API). Logs every inspection securely for compliance (traceability).
*   **Key Metrics:** Precision: 42.8%, Recall: 35.8%, mAP@0.5: 31.9%.

**Strict Presentation Requirements (Based on InnoVent Rules):**
You must structure the presentation to cover all the mandatory sections below. For each slide, provide the **Slide Title**, **Visual/Design Instructions**, **On-Slide Bullet Points**, and **Detailed Speaker Notes**.

### Mandatory Slide Sequence & Content Requirements:

1. **Introduction:** Introduce the team and provide a brief, high-impact overview of ZeroDefect.
2. **Problem Statement:** Clearly define the problem (manual visual inspections miss defects; cloud-based AI fails on hangar floors due to lack of internet and data privacy rules). You **must** include the market size (Aerospace MRO / NDT market) and future growth potential.
3. **Objective and Approach:** State the main objective (bringing intelligent inspection to the edge) and describe the methodology (training a lightweight YOLOv5n model, quantization, and edge deployment).
4. **Solution Overview (Can be 2-3 slides):** Present a detailed overview of the system architecture (Camera -> AIM Edge Device -> Dashboard -> Audit Ledger). Highlight key components and functionalities.
5. **Self-Evaluation & Declaration - Novelty:** Explicitly address the novelty of the project (combining edge object detection with a tamper-proof audit trail for aviation compliance).
6. **Technical Implementation:** Explain the exact technical aspects, tools, and frameworks used (YOLOv5, ONNX, RKNN, FastAPI, SQLite, Arduino/C++).
7. **Challenges Faced:** Discuss specific challenges encountered during development (e.g., quantizing the model to fit on a 2.6MB NPU, handling false positives, building the custom AIM firmware) and how the team overcame them.
8. **Results and Achievements:** Showcase the outcomes supported by data (include the Precision/Recall/mAP metrics, inference speed, and model size reduction).
9. **Demonstration:** Dedicate a slide setting up the recorded demonstration. Describe what the audience will see in the video (e.g., Live webcam detecting a crack, updating the dashboard, and writing to the secure log).
10. **Self-Evaluation & Declaration - Feasibility & Scaling:** Discuss the feasibility of developing the physical prototype (the AIM module) and how this innovation can scale across different manufacturing and maintenance sectors.
11. **Self-Evaluation & Declaration - Potential Benefit:** Highlight the concrete benefits delivered (reduction in inspection time, elimination of human error, strict regulatory compliance).
12. **Future Enhancements:** Share potential future iterations (e.g., integration with drone scanners, expanding to new defect classes, predictive maintenance analytics).
13. **Project Plan:** Provide a timeline or Gantt chart summarizing the plan for building the functional PoC.

**Output Format Constraint:**
Format your response clearly with headers for each Slide Number. Ensure the tone is professional, innovative, and data-driven.
