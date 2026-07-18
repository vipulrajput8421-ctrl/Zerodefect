# ZeroDefect — Stage 2 Presentation Outline
## InnoVent 2026-27 | Track 3.2.3.4: Intelligent Inspection & Defect Detection

> **Instructions:** Fill in each section with YOUR data from `python data/summary_stats.py`. Never invent numbers — only use real metrics.

---

### Slide 1: Problem Statement
_Drop in:_ Photograph or schematic of your M8 bolt + a clear photo of one defect type.
→ State: "Manual inspection of automotive fasteners misses 20–30% of defects. Labeling enough images for traditional AI costs $20K–$100K per product line."

### Slide 2: Our Approach — Anomaly-First + Few-Shot
_Drop in:_ The system architecture diagram from the README (copy the mermaid or screenshot it).
→ Explain PatchCore in one sentence: The model compares patch features of the input part to a coreset memory bank of features learned from non-defective parts, calculating distances as an anomaly score.
→ Key claim: "We trained on only X good images and needed just Y–Z labeled defect examples — not thousands."

### Slide 3: Live Demo
_Drop in:_ Screen recording of `python src/live_demo.py` showing:
1. A good bolt → green OK overlay
2. A defective bolt → red DEFECT: crack overlay
→ Show the confidence % updating in real time.

### Slide 4: Validation Metrics
_Drop in:_ Output table from `python src/evaluate_model.py --labels <labels.csv>`:
→ Accuracy: `[insert from script output]`
→ Precision: `[insert]`, Recall: `[insert]`, F1: `[insert]`
→ Screenshot of `models/eval_scores.png` histogram

### Slide 5: Audit Trail & Traceability
_Drop in:_ Screenshot of `python src/view_log.py` output.
→ State: "Every inspection is logged with timestamp, image, decision, and confidence. This audit trail directly addresses EU AI Act Article 12 traceability requirements for high-risk industrial AI systems, effective August 2026."
→ Note: this is a design feature, not a certification claim.

### Slide 6: Roadmap — Edge Deployment
_Drop in:_ Photo of AIM board (or the diagram from aim/README.md).
→ Describe Phase 2: model quantized to INT8, runs on AIM standalone, serves live results over Wi-Fi to any browser.
→ Timeline: Weeks 7–8 of build plan.

### Slide 7: Q&A Prep — Pre-Written Answers

**"How many labeled defect images did you need?"**
→ Run `python data/summary_stats.py` and use: "We used [X] defect images across [N] defect types — [Y] per type on average. Traditional approaches would require 500–5000."

**"What happens with a defect type you've never seen?"**
→ "Our anomaly-first model still flags it as DEFECT (high anomaly score), even without a specific class. It just can't name the defect type — which is still valuable for operators."

**"Is this production-ready?"**
→ "This is a proof-of-concept prototype demonstrating the core technique used by commercial systems like Overview.ai. Production readiness would require additional validation datasets, safety certification, and integration with the production line PLC system."
