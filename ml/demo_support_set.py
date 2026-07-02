"""
Synthetic demo support set generator.

A real deployment of this system would use a support set built from a
clinician-curated, IRB-approved library of confirmed rare-disease fundus
photographs (K images per class, K = "K-shot"). No such dataset is bundled
here, so this module procedurally generates stylised, fundus-like images
with class-specific visual patterns (colour tint, lesion blobs, vessel
texture, noise) purely so the Prototypical Network pipeline has a support
set to compute real prototypes from and the app is fully runnable end to
end.

Swap this module out for a loader that reads real images from
static/support_set/<class_name>/*.jpg to go from "demo" to "production".
"""
import os
import random
from PIL import Image, ImageDraw, ImageFilter

RARE_DISEASE_CLASSES = [
    "Retinitis Pigmentosa",
    "Coats Disease",
    "Stargardt Disease",
    "Best Vitelliform Dystrophy",
    "Choroideremia",
]

# deterministic per-class visual signature so prototypes are separable
CLASS_STYLE = {
    "Retinitis Pigmentosa": {"tint": (120, 70, 40), "blobs": 14, "blob_color": (20, 15, 10), "blob_r": (3, 7)},
    "Coats Disease": {"tint": (200, 150, 90), "blobs": 6, "blob_color": (235, 225, 180), "blob_r": (10, 22)},
    "Stargardt Disease": {"tint": (150, 110, 40), "blobs": 20, "blob_color": (255, 215, 120), "blob_r": (2, 5)},
    "Best Vitelliform Dystrophy": {"tint": (170, 90, 30), "blobs": 1, "blob_color": (255, 200, 60), "blob_r": (28, 34)},
    "Choroideremia": {"tint": (90, 60, 55), "blobs": 10, "blob_color": (60, 30, 25), "blob_r": (8, 16)},
}


def _fundus_base(size, tint, seed):
    rnd = random.Random(seed)
    img = Image.new("RGB", (size, size), (10, 5, 5))
    draw = ImageDraw.Draw(img)
    cx, cy, r = size // 2, size // 2, int(size * 0.46)

    # radial-ish gradient disc approximated with concentric ellipses
    for i in range(r, 0, -2):
        t = i / r
        color = tuple(int(c * (0.55 + 0.45 * (1 - t)) + rnd.uniform(-6, 6)) for c in tint)
        color = tuple(max(0, min(255, c)) for c in color)
        draw.ellipse([cx - i, cy - i, cx + i, cy + i], fill=color)

    # optic disc
    od_r = int(size * 0.09)
    draw.ellipse(
        [cx - od_r - int(size * 0.18), cy - od_r, cx + od_r - int(size * 0.18), cy + od_r],
        fill=(255, 225, 170),
    )

    # simple vessel-like branching lines
    for _ in range(9):
        x0, y0 = cx - int(size * 0.18), cy
        angle = rnd.uniform(0, 6.28)
        length = rnd.uniform(size * 0.2, size * 0.42)
        x1 = x0 + length * random.Random(seed + 1).uniform(-1, 1)
        y1 = y0 + length * random.Random(seed + 2).uniform(-1, 1)
        draw.line([x0, y0, x1, y1], fill=(120, 30, 30), width=2)

    return img, draw, cx, cy, r


def _make_image(class_name, size, seed):
    style = CLASS_STYLE[class_name]
    img, draw, cx, cy, r = _fundus_base(size, style["tint"], seed)
    rnd = random.Random(seed)

    for _ in range(style["blobs"]):
        ang = rnd.uniform(0, 6.28)
        dist = rnd.uniform(0, r * 0.8)
        bx = cx + dist * random.Random(seed + int(ang * 1000)).uniform(-1, 1)
        by = cy + dist * random.Random(seed + int(ang * 999)).uniform(-1, 1)
        br = rnd.randint(*style["blob_r"])
        draw.ellipse([bx - br, by - br, bx + br, by + br], fill=style["blob_color"])

    img = img.filter(ImageFilter.GaussianBlur(radius=1.1))
    # circular mask so it looks like a fundus photo
    mask = Image.new("L", (size, size), 0)
    mdraw = ImageDraw.Draw(mask)
    mdraw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=255)
    black_bg = Image.new("RGB", (size, size), (0, 0, 0))
    img = Image.composite(img, black_bg, mask)
    return img


def ensure_support_set(support_root, k_shot=5, size=224):
    """Create the synthetic support set on disk if it doesn't exist yet."""
    os.makedirs(support_root, exist_ok=True)
    seed_counter = 1000
    for class_name in RARE_DISEASE_CLASSES:
        class_dir = os.path.join(support_root, class_name.replace(" ", "_"))
        os.makedirs(class_dir, exist_ok=True)
        existing = [f for f in os.listdir(class_dir) if f.lower().endswith((".png", ".jpg"))]
        if len(existing) >= k_shot:
            continue
        for i in range(k_shot):
            seed_counter += 1
            img = _make_image(class_name, size, seed=seed_counter)
            img.save(os.path.join(class_dir, f"support_{i+1}.png"))
    return support_root