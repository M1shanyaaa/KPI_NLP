"""
analize.py — аналіз пори року на відеопотоці
=============================================
Використання:
    python analyze.py --video input.mp4 --model season_model.pth

Результат:
    - output_annotated.mp4   (відео з підписами)
    - консольний звіт
"""

import sys, argparse
from pathlib import Path
from collections import Counter

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from torchvision.models import ResNet18_Weights
from PIL import Image

SEASONS = ["spring", "summer", "autumn", "winter"]
SEASON_UA = {"spring":"Весна", "summer":"Літо", "autumn":"Осінь", "winter":"Зима"}
COLORS    = {"spring":(80,200,80), "summer":(50,180,50),
             "autumn":(30,100,200), "winter":(220,220,255)}

TRANSFORM = transforms.Compose([
    transforms.Resize(224), transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])


def load_model(path: str, device):
    ckpt  = torch.load(path, map_location=device)
    base  = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    base.fc = nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, 4)
    )
    base.load_state_dict(ckpt["state"])
    return base.to(device).eval()


@torch.no_grad()
def predict(model, frame_bgr, device):
    img    = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    tensor = TRANSFORM(img).unsqueeze(0).to(device)
    probs  = torch.softmax(model(tensor), 1).squeeze().cpu().tolist()
    return SEASONS[probs.index(max(probs))], dict(zip(SEASONS, probs))


def annotate(frame, season, probs):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (w, 65), (0,0,0), -1)
    frame = cv2.addWeighted(ov, 0.5, frame, 0.5, 0)
    cv2.putText(frame, f"СЕЗОН: {SEASON_UA[season]}",
                (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                COLORS[season], 2, cv2.LINE_AA)
    # Смужки ймовірностей
    for i, s in enumerate(SEASONS):
        p = probs[s]; y = 10 + i*14
        cv2.rectangle(frame, (w-165, y), (w-165+int(p*140), y+10), COLORS[s], -1)
        cv2.putText(frame, f"{SEASON_UA[s][0]}{p:.2f}",
                    (w-210, y+9), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255,255,255), 1)
    return frame


def analyze(video_path: str, model_path: str,
            out_path: str = "output_annotated.mp4",
            every: int = 15):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = load_model(model_path, device)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    counts, season, probs = Counter(), "winter", {}
    idx = 0
    print(f"Обробка {video_path}  ({total} кадрів)...")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every == 0:
            season, probs = predict(model, frame, device)
            counts[season] += 1
        out.write(annotate(frame, season, probs))
        idx += 1

    cap.release(); out.release()

    # ── Звіт ──
    total_pred = sum(counts.values()) or 1
    dominant   = counts.most_common(1)[0][0]
    pct        = counts[dominant] / total_pred

    conf = ("дуже висока" if pct > .9 else
            "висока"      if pct > .75 else
            "помірна"     if pct > .6  else "низька")

    print(f"\n{'─'*50}")
    print(f"  Домінуюча пора року : {SEASON_UA[dominant]} ({pct:.1%})")
    print(f"  Впевненість         : {conf}")
    print(f"  Розподіл кадрів:")
    for s in SEASONS:
        p = counts[s]/total_pred*100
        bar = "█"*int(p/2) + "░"*(50-int(p/2))
        print(f"    {SEASON_UA[s]:<7} [{bar}] {p:4.1f}%")

    print(f"\n  ЛІНГВІСТИЧНЕ ТЛУМАЧЕННЯ:")
    desc = {
        "spring": "зелень молодого листя, помірна освітленість, відсутність снігу.",
        "summer": "густа рослинність, інтенсивне сонячне освітлення, насичені кольори.",
        "autumn": "жовто-коричневі відтінки листя, знижена яскравість, листопад.",
        "winter": "сніговий покрив або голі дерева, холодна синьо-біла палітра.",
    }
    season_genitive = {"spring":"весни","summer":"літа","autumn":"осені","winter":"зими"}
    print(f"  На відеопотоці виявлено ознаки {season_genitive[dominant]} ({conf} впевненість): {desc[dominant]}")
    print(f"{'─'*50}")
    print(f"\nАнотоване відео → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="Шлях до .mp4 файлу")
    p.add_argument("--model", default="season_model.pth")
    p.add_argument("--output", default="output_annotated.mp4")
    p.add_argument("--every",  type=int, default=15,
                   help="Класифікувати кожен N-й кадр (default: 15)")
    args = p.parse_args()

    if not Path(args.video).exists():
        sys.exit(f"Файл не знайдено: {args.video}")
    if not Path(args.model).exists():
        sys.exit(f"Модель не знайдено: {args.model}. Спочатку: python train.py")

    analyze(args.video, args.model, args.output, args.every)