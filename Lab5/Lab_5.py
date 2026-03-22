"""
Lab_5.py — визначення пори року через CLIP (zero-shot)
=============================================================
Не потребує навчання і датасету.

Встановлення:
    pip install torch torchvision pillow opencv-python
    pip install git+https://github.com/openai/CLIP.git

Використання:
    python analyze_clip.py --video input.mp4
"""

import sys, argparse
from pathlib import Path
from collections import Counter

import cv2
import torch
import clip
from PIL import Image

# ── Текстові описи для кожного сезону ────────────────────────────────────────
# Чим точніші описи — тим краще працює CLIP
SEASON_PROMPTS = {
    "spring": [
        "a photo of spring with blooming flowers and fresh green leaves",
        "spring landscape with cherry blossoms and young vegetation",
        "early spring nature with budding trees and mild weather",
    ],
    "summer": [
        "a photo of summer with lush green trees and bright sunshine",
        "summer landscape with intense sunlight and deep green foliage",
        "hot sunny summer day with dense green vegetation",
    ],
    "autumn": [
        "a photo of autumn with orange and yellow fallen leaves",
        "fall landscape with colorful red and brown foliage",
        "autumn forest with golden leaves and overcast sky",
    ],
    "winter": [
        "a photo of winter with snow covered trees and frozen ground",
        "winter landscape with snow and bare leafless trees",
        "cold winter scene with white snow and grey sky",
    ],
}

SEASON_UA = {"spring":"Весна","summer":"Літо","autumn":"Осінь","winter":"Зима"}
COLORS    = {"spring":(80,200,80),"summer":(50,180,50),
             "autumn":(30,100,200),"winter":(220,220,255)}
SEASONS   = list(SEASON_PROMPTS.keys())


class ClipSeasonClassifier:
    def __init__(self, device):
        self.device = device
        print("[CLIP] Завантаження моделі ViT-B/32...")
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.model.eval()

        # Попередньо кодуємо всі текстові описи
        self.season_features = {}
        with torch.no_grad():
            for season, prompts in SEASON_PROMPTS.items():
                tokens = clip.tokenize(prompts).to(device)
                feats  = self.model.encode_text(tokens)
                feats  = feats / feats.norm(dim=-1, keepdim=True)
                # Усереднюємо по всіх промптах сезону
                self.season_features[season] = feats.mean(dim=0)
        print("[CLIP] Готово\n")

    @torch.no_grad()
    def predict(self, frame_bgr):
        img    = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        img_feat = self.model.encode_image(tensor)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # Косинусна схожість з кожним сезоном
        sims  = {s: (img_feat @ self.season_features[s].unsqueeze(-1)).item()
                 for s in SEASONS}

        # Softmax → ймовірності
        vals  = torch.tensor(list(sims.values()))
        probs = torch.softmax(vals * 10, dim=0).tolist()
        result = dict(zip(SEASONS, probs))
        best   = max(result, key=result.get)
        return best, result


def annotate(frame, season, probs):
    h, w = frame.shape[:2]
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (w, 65), (0, 0, 0), -1)
    frame = cv2.addWeighted(ov, 0.5, frame, 0.5, 0)
    cv2.putText(frame, f"СЕЗОН: {SEASON_UA[season]}",
                (12, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                COLORS[season], 2, cv2.LINE_AA)
    for i, s in enumerate(SEASONS):
        p = probs[s]; y = 10 + i * 14
        cv2.rectangle(frame, (w-165, y), (w-165+int(p*140), y+10), COLORS[s], -1)
        cv2.putText(frame, f"{SEASON_UA[s][0]}{p:.2f}",
                    (w-210, y+9), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255,255,255), 1)
    return frame


def analyze(video_path: str, out_path: str = "output_clip.mp4", every: int = 15):
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = ClipSeasonClassifier(device)

    cap   = cv2.VideoCapture(video_path)
    fps   = int(cap.get(cv2.CAP_PROP_FPS)) or 25
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out   = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    counts, season, probs = Counter(), "winter", {s: 0.0 for s in SEASONS}
    print(f"Обробка {video_path}  ({total} кадрів) | пристрій: {device}")

    for idx in range(total):
        ok, frame = cap.read()
        if not ok:
            break
        if idx % every == 0:
            season, probs = classifier.predict(frame)
            counts[season] += 1
            if idx % (every * 20) == 0:
                print(f"  [{idx}/{total}] → {SEASON_UA[season]}")
        out.write(annotate(frame, season, probs))

    cap.release(); out.release()

    # ── Звіт ──────────────────────────────────────────────────────────────────
    total_pred = sum(counts.values()) or 1
    dominant   = counts.most_common(1)[0][0]
    pct        = counts[dominant] / total_pred

    conf = ("дуже висока" if pct > .9 else
            "висока"      if pct > .75 else
            "помірна"     if pct > .6  else "низька")

    genitive = {"spring":"весни","summer":"літа","autumn":"осені","winter":"зими"}
    desc = {
        "spring": "квітуючі дерева, свіже молоде листя, м'яке освітлення.",
        "summer": "густа темно-зелена рослинність, яскраве сонячне світло.",
        "autumn": "жовто-коричневе листя, приглушена яскравість, листопад.",
        "winter": "сніговий покрив або голі дерева, холодна синьо-біла палітра.",
    }

    print(f"\n{'─'*52}")
    print(f"  Метод               : CLIP zero-shot (ViT-B/32)")
    print(f"  Домінуюча пора року : {SEASON_UA[dominant]} ({pct:.1%})")
    print(f"  Впевненість         : {conf}")
    print(f"  Розподіл кадрів:")
    for s in SEASONS:
        p   = counts[s] / total_pred * 100
        bar = "█" * int(p/2) + "░" * (50 - int(p/2))
        print(f"    {SEASON_UA[s]:<7} [{bar}] {p:4.1f}%")
    print(f"\n  ЛІНГВІСТИЧНЕ ТЛУМАЧЕННЯ:")
    print(f"  На відеопотоці виявлено ознаки {genitive[dominant]}")
    print(f"  ({conf} впевненість): {desc[dominant]}")
    print(f"{'─'*52}")
    print(f"\nАнотоване відео → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video",  required=True)
    p.add_argument("--output", default="output_clip.mp4")
    p.add_argument("--every",  type=int, default=15)
    args = p.parse_args()

    if not Path(args.video).exists():
        sys.exit(f"Файл не знайдено: {args.video}")

    analyze(args.video, args.output, args.every)