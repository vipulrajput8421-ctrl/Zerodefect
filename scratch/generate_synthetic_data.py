import os
import sys
import math
import random
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path("d:/ZeroDefect")
GOOD_DIR = ROOT / "data" / "good"
DEFECTS_DIR = ROOT / "data" / "defects"
TEST_DIR = ROOT / "data" / "test"

# Make sure directories exist
GOOD_DIR.mkdir(parents=True, exist_ok=True)
(DEFECTS_DIR / "crack").mkdir(parents=True, exist_ok=True)
(DEFECTS_DIR / "scratch").mkdir(parents=True, exist_ok=True)
(DEFECTS_DIR / "dent").mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

def draw_hexagon(draw, center, radius, rotation_deg, fill_color, outline_color=None, outline_width=1):
    cx, cy = center
    angle_offset = math.radians(rotation_deg)
    points = []
    for i in range(6):
        angle = angle_offset + i * (2 * math.pi / 6)
        x = cx + radius * math.cos(angle)
        y = cy + radius * math.sin(angle)
        points.append((x, y))
    draw.polygon(points, fill=fill_color, outline=outline_color, width=outline_width)
    return points

def generate_bolt_image(defect_type=None, angle=0.0, dx=0, dy=0, scale=1.0, brightness=1.0):
    # 640x480 image with a solid dark grey background
    img = Image.new("RGB", (640, 480), color=(30, 30, 30))
    draw = ImageDraw.Draw(img)
    
    # Center and sizes
    cx = 320 + dx
    cy = 240 + dy
    r1 = 100 * scale  # Outer hex radius
    r2 = 60 * scale   # Inner circle radius
    r3 = 30 * scale   # Thread/hole radius
    
    # Adjust colors based on brightness factor
    hex_color = tuple(int(c * brightness) for c in (160, 160, 160))
    circle_color = tuple(int(c * brightness) for c in (130, 130, 130))
    hole_color = tuple(int(c * brightness) for c in (70, 70, 70))
    
    # 1. Draw Outer Hexagon
    draw_hexagon(draw, (cx, cy), r1, angle, fill_color=hex_color)
    
    # 2. Deform edge for "dent" defect (before rendering inner parts)
    if defect_type == "dent":
        dent_angle = math.radians(angle + random.choice([30, 90, 150, 210, 270, 330]) + random.uniform(-10, 10))
        dent_cx = cx + r1 * math.cos(dent_angle)
        dent_cy = cy + r1 * math.sin(dent_angle)
        dent_r = random.uniform(22, 35)
        # Bite a piece off the hex head using background color
        draw.ellipse([dent_cx - dent_r, dent_cy - dent_r, dent_cx + dent_r, dent_cy + dent_r], fill=(30, 30, 30))
        
    # 3. Draw Inner Circle
    draw.ellipse([cx - r2, cy - r2, cx + r2, cy + r2], fill=circle_color)
    
    # 4. Draw Center Hole
    draw.ellipse([cx - r3, cy - r3, cx + r3, cy + r3], fill=hole_color)
    
    # 5. Draw concentric circles inside the hole to mimic threads
    for i in range(1, 4):
        tr = r3 - i * 6 * scale
        if tr > 2:
            draw.ellipse([cx - tr, cy - tr, cx + tr, cy + tr], outline=(90, 90, 90), width=2)
            
    # 6. Apply crack / scratch defects on top
    if defect_type == "crack":
        # Draw dark jagged line across the bolt surface
        start_angle = random.uniform(0, 2 * math.pi)
        sx = cx + random.uniform(5, r2) * math.cos(start_angle)
        sy = cy + random.uniform(5, r2) * math.sin(start_angle)
        
        points = [(sx, sy)]
        curr_x, curr_y = sx, sy
        length = random.uniform(60, 110)
        direction = start_angle + random.uniform(-0.4, 0.4)
        
        steps = 5
        for _ in range(steps):
            step_len = length / steps
            curr_x += step_len * math.cos(direction) + random.uniform(-8, 8)
            curr_y += step_len * math.sin(direction) + random.uniform(-8, 8)
            points.append((curr_x, curr_y))
            direction += random.uniform(-0.25, 0.25)
            
        draw.line(points, fill=(15, 15, 15), width=random.choice([3, 4]))
        
    elif defect_type == "scratch":
        # Draw a thin bright line across the bolt surface (specular reflection)
        sx = cx + random.uniform(-r1, r1)
        sy = cy + random.uniform(-r1, r1)
        ex = sx + random.uniform(-90, 90)
        ey = sy + random.uniform(-90, 90)
        draw.line([(sx, sy), (ex, ey)], fill=(245, 245, 245), width=random.choice([1, 2]))
        
    # Apply random pixel noise
    img_arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 2.5, img_arr.shape)
    img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_arr)
    # Apply light blur to simulate webcam optics
    img = img.filter(ImageFilter.GaussianBlur(0.4))
    return img

def main():
    print("[*] Generating synthetic images for ZeroDefect...")
    
    # 1. Generate 120 Good Bolt images for training
    print("Generating 120 good bolt images...")
    for i in range(120):
        angle = random.uniform(0, 360)
        dx = random.randint(-12, 12)
        dy = random.randint(-12, 12)
        scale = random.uniform(0.96, 1.04)
        brightness = random.uniform(0.93, 1.07)
        img = generate_bolt_image(defect_type=None, angle=angle, dx=dx, dy=dy, scale=scale, brightness=brightness)
        img.save(GOOD_DIR / f"good_{i+1:04d}.jpg")
        
    # 2. Generate 15 images of each defect type
    defect_types = ["crack", "scratch", "dent"]
    for dt in defect_types:
        print(f"Generating 15 '{dt}' defect images...")
        for i in range(15):
            angle = random.uniform(0, 360)
            dx = random.randint(-12, 12)
            dy = random.randint(-12, 12)
            scale = random.uniform(0.96, 1.04)
            brightness = random.uniform(0.93, 1.07)
            img = generate_bolt_image(defect_type=dt, angle=angle, dx=dx, dy=dy, scale=scale, brightness=brightness)
            img.save(DEFECTS_DIR / dt / f"{dt}_{i+1:04d}.jpg")
            
    # 3. Generate test set (10 good, 10 of each defect) and write labels CSV
    print("Generating test images and labels...")
    test_rows = []
    
    # Good test images
    for i in range(10):
        angle = random.uniform(0, 360)
        dx = random.randint(-15, 15)
        dy = random.randint(-15, 15)
        scale = random.uniform(0.95, 1.05)
        brightness = random.uniform(0.90, 1.10)
        img = generate_bolt_image(defect_type=None, angle=angle, dx=dx, dy=dy, scale=scale, brightness=brightness)
        filename = f"test_good_{i+1:04d}.jpg"
        img.save(TEST_DIR / filename)
        test_rows.append(f"{filename},good")
        
    # Defect test images
    for dt in defect_types:
        for i in range(10):
            angle = random.uniform(0, 360)
            dx = random.randint(-15, 15)
            dy = random.randint(-15, 15)
            scale = random.uniform(0.95, 1.05)
            brightness = random.uniform(0.90, 1.10)
            img = generate_bolt_image(defect_type=dt, angle=angle, dx=dx, dy=dy, scale=scale, brightness=brightness)
            filename = f"test_{dt}_{i+1:04d}.jpg"
            img.save(TEST_DIR / filename)
            test_rows.append(f"{filename},defect")
            
    # Save CSV
    with open(ROOT / "data" / "test_labels.csv", "w") as f:
        for row in test_rows:
            f.write(row + "\n")
            
    print(f"[OK] Synthetic dataset successfully generated!")
    print(f"  - Good training images: {len(list(GOOD_DIR.glob('*.jpg')))}")
    print(f"  - Defect training folders:")
    for dt in defect_types:
        print(f"    * {dt}: {len(list((DEFECTS_DIR / dt).glob('*.jpg')))}")
    print(f"  - Test images: {len(list(TEST_DIR.glob('*.jpg')))}")
    print(f"  - Test labels saved to: {ROOT / 'data' / 'test_labels.csv'}")

if __name__ == "__main__":
    main()
