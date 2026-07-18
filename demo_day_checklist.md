# ZeroDefect — Demo Day Checklist
## InnoVent 2026 | Physical Hardware Demo

### T-60 Minutes: Pre-Demo Setup
- [ ] Laptop charged, charger plugged in
- [ ] Webcam connected and tested (`python src/capture_good.py --target-count 1` to verify)
- [ ] Virtual environment activated: `zerodefect_env\Scripts\activate`
- [ ] Models present: `models/memory_bank.pkl`, `models/classifier.pkl`, `models/model_info.json`
- [ ] Run environment check: `python src/environment_check.py`
- [ ] Start live demo: `python src/live_demo.py` — verify OK/DEFECT responses
- [ ] Clear old logs if desired: `python src/view_log.py --clear`
- [ ] Have 3 good bolts and 3 defective bolts physically labelled and within reach

### T-30 Minutes: Hardware Check (AIM)
- [ ] AIM Board powered on
- [ ] IP address visible in Serial Monitor (or noted from last session)
- [ ] Browser on judge's phone opened to http://[IP_ADDRESS]/ — page loads
- [ ] Camera feed responding (page auto-refreshes every 2s)
- [ ] Test: hold good bolt in front → page shows OK
- [ ] Test: hold defective bolt → page shows DEFECT (or placeholder if model not yet flashed)

### T-10 Minutes: Final Checks
- [ ] Run `python data/summary_stats.py` — screenshot the output for slides
- [ ] Browser tab open: http://[IP_ADDRESS]/
- [ ] Terminal window open with `python src/live_demo.py` running
- [ ] `python src/view_log.py` ready to run as supplementary demo

### During Demo: Sequence
1. Show laptop demo first (most reliable): live_demo.py running
2. Hold GOOD bolt → "OK" in green ✓
3. Hold DEFECTIVE bolt → "DEFECT: crack" in red ✓
4. Run `python src/view_log.py` → show audit trail
5. Switch to AIM browser demo on judge's phone (secondary)

### Hardware Failure — Webcam Fallback
If AIM hardware fails:
```bash
# Switch entirely to laptop webcam demo:
python src/live_demo.py --camera-id 0

# Show audit log:
python src/view_log.py

# Show training stats:
python data/summary_stats.py
```
This is a complete, presentable demo without any hardware.

### Key Questions — Pre-Written Answers

**"How many labeled defect images did this need?"**
→ Run `python data/summary_stats.py` first, then answer:
_"We used [X total defect images] across [N] defect types — averaging [Y] per type. A traditional classifier approach would require 500–5,000 per class."_

**"What if it sees a defect type it wasn't trained on?"**
→ _"It still flags it as DEFECT via the anomaly score — it just can't classify the specific type. That's a feature: we catch unknown defects by definition."_

**"How accurate is it?"**
→ _"On our held-out test set: [paste accuracy from evaluate_model.py output]. More importantly, recall — catching actual defects — is our priority metric."_

**"Is this real-time?"**
→ _"The laptop runs at ~3 FPS by design to stay light. The AIM checks every 2 seconds, which is appropriate for a station-based inspection setup rather than a conveyor belt."_

### Backup Materials
- [ ] USB drive with: models/, src/, presentation slides, demo video recording of live_demo.py
- [ ] Offline copy of this README
- [ ] Pre-recorded 60-second demo video of working webcam demo (fallback if everything fails)
