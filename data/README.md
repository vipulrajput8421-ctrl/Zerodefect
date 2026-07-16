# ZeroDefect — Dataset Specification

## Component Under Inspection
**Component:** M8 Hex Bolt (standard automotive fastener)
**Inspection target:** Detection of manufacturing/handling defects that would cause field failures or torque specification failures.

## Defect Types (Scope: 3 maximum)
| Defect Type | Description | Visual Signature |
|---|---|---|
| `crack` | Stress cracks on bolt head or shaft | Dark linear discontinuity |
| `scratch` | Surface scratches from handling/transport | Bright linear reflective mark |
| `dent` | Mechanical deformation of bolt head | Geometric irregularity in hex faces |

## Photography Setup
- **Background:** Plain matte black or white surface (solid-colour, non-reflective)
- **Lighting:** Consistent overhead diffuse light — avoid harsh shadows or reflections. Two LED panels at 45° if available.
- **Camera distance:** ~20–25 cm from bolt head to lens
- **Camera angle:** Top-down (0°) as primary. Capture ±15° variants for robustness.
- **Resolution:** Minimum 640×480; 1280×960 preferred.

## Data Collection Rules
1. `data/good/` — 100–150 images of NON-DEFECTIVE bolts only
2. `data/defects/<type>/` — 5–20 images per defect type
3. `data/test/` — Mixed good + defective (for evaluation only, not training)
4. NEVER mix defective images into data/good/

## Data Collection Commands
```bash
# Capture good parts:
python src/capture_good.py --target-count 120

# Capture each defect type:
python src/capture_defects.py --defect-type crack
python src/capture_defects.py --defect-type scratch
python src/capture_defects.py --defect-type dent
```

## Current Data Status
| Folder | Target | Current |
|---|---|---|
| data/good/ | 100–150 | _collect now_ |
| data/defects/crack/ | 5–20 | _collect after training_ |
| data/defects/scratch/ | 5–20 | _collect after training_ |
| data/defects/dent/ | 5–20 | _collect after training_ |
