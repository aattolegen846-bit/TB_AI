#!/usr/bin/env python3
"""Small smoke inference script for 2-3 sample images.

Usage:
    python tools/detect_tiny.py /path/to/images [output_dir] [score_threshold]

Saves annotated images to output_dir (default: data/mytiny/out).
Works on CPU or Apple MPS if available. Requires torch and torchvision.
"""
import sys
import os
from pathlib import Path

try:
    import torch
    import torchvision
    import torchvision.transforms as T
    from PIL import Image, ImageDraw, ImageFont
except Exception as e:
    print("Missing dependency:", e)
    print("Install dependencies: pip install torch torchvision pillow")
    raise


def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(device):
    # Use pretrained Faster R-CNN for quick smoke tests
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    model.to(device)
    return model


def draw_boxes(img: Image.Image, boxes, scores, labels, score_thr=0.5):
    draw = ImageDraw.Draw(img)
    # Try to load a default font, fall back to none
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None
    for box, score, label in zip(boxes, scores, labels):
        if score < score_thr:
            continue
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        txt = f"{int(label.item())}:{score:.2f}"
        draw.text((x1, max(y1 - 10, 0)), txt, fill="red", font=font)
    return img


def main(argv):
    if len(argv) < 2:
        print("Usage: python tools/detect_tiny.py /path/to/images [output_dir] [score_threshold]")
        sys.exit(1)

    img_dir = Path(argv[1])
    out_dir = Path(argv[2]) if len(argv) > 2 else Path("data/mytiny/out")
    score_thr = float(argv[3]) if len(argv) > 3 else 0.5

    if not img_dir.exists():
        print("Image directory does not exist:", img_dir)
        sys.exit(1)

    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    print("Using device:", device)

    model = load_model(device)
    transform = T.Compose([T.ToTensor()])

    imgs = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')])
    if not imgs:
        print("No images found in", img_dir)
        sys.exit(1)

    for p in imgs:
        try:
            img = Image.open(p).convert('RGB')
            tensor = transform(img).to(device)
            with torch.no_grad():
                preds = model([tensor])[0]

            boxes = preds.get('boxes', torch.tensor([])).cpu()
            scores = preds.get('scores', torch.tensor([])).cpu()
            labels = preds.get('labels', torch.tensor([])).cpu()

            out_img = img.copy()
            out_img = draw_boxes(out_img, boxes, scores, labels, score_thr)
            out_path = out_dir / p.name
            # Save atomically: write to a temp file then move into place
            tmp_path = out_dir / (p.name + '.tmp')
            try:
                # ensure RGB and explicit format
                out_img = out_img.convert('RGB')
                out_img.save(tmp_path, format='PNG')
                # atomic replace
                tmp_path.replace(out_path)
                print('Processed', p.name, '->', out_path)
            except Exception as e:
                # cleanup temp file if present
                try:
                    if tmp_path.exists():
                        tmp_path.unlink()
                except Exception:
                    pass
                print('Failed to save', p.name, ':', e)
        except Exception as e:
            print('Failed to process', p.name, ':', e)

    print('Done. Annotated images saved in', out_dir)


if __name__ == '__main__':
    main(sys.argv)
