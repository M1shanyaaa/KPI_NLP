"""
training.py — навчання моделі розпізнавання пори року
====================================================
Структура датасету:
    dataset/
        spring/  *.jpg
        summer/
        autumn/
        winter/

Зібрати датасет автоматично:
    pip install bing-image-downloader
    python train.py --collect

Навчити модель:
    pip install torch torchvision
    python train.py
"""

import os, argparse, warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
warnings.filterwarnings("ignore")

SEASONS   = ["spring", "summer", "autumn", "winter"]
DATA_DIR  = "./dataset"
MODEL_OUT = "season_model.pth"
EPOCHS    = 15
BATCH     = 32


# ── Збір датасету через Bing Images ──────────────────────────────────────────
def collect_dataset(n_per_class: int = 150, pexels_key: str = None):
    """
    Два варіанти збору датасету:

    1) Pexels API (рекомендовано) — якісні фото природи, без ілюстрацій.
       Безкоштовна реєстрація: https://www.pexels.com/api/
       Передати ключ: python train.py --collect --pexels-key YOUR_KEY

    2) icrawler + Google Images (fallback, без реєстрації).
       pip install icrawler
       python train.py --collect
    """
    os.makedirs(DATA_DIR, exist_ok=True)

    QUERIES = {
        "spring": ["spring blooming trees nature", "spring cherry blossom meadow",
                   "spring green forest flowers"],
        "summer": ["summer sunny green field nature", "summer blue sky trees meadow",
                   "summer hot bright nature landscape", "summer lush green forest"],
        "autumn": ["autumn orange red leaves forest", "autumn fall foliage trees",
                   "autumn colorful forest landscape"],
        "winter": ["winter snow forest trees", "winter snow landscape nature",
                   "winter frozen lake snow"],
    }

    if pexels_key:
        _collect_pexels(QUERIES, n_per_class, pexels_key)
    else:
        _collect_icrawler(QUERIES, n_per_class)


def _collect_pexels(queries: dict, n_per_class: int, api_key: str):
    """Завантажує фото через Pexels API — найякісніший варіант."""
    try:
        import requests
    except ImportError:
        print("Встановіть: pip install requests")
        return

    session = requests.Session()
    session.headers.update({"Authorization": api_key})
    print("[Pexels] Збір датасету через Pexels API...")

    for season, query_list in queries.items():
        dst = os.path.join(DATA_DIR, season)
        os.makedirs(dst, exist_ok=True)
        downloaded = 0
        per_query = max(1, n_per_class // len(query_list))

        for query in query_list:
            page = 1
            while downloaded < n_per_class:
                try:
                    r = session.get(
                        "https://api.pexels.com/v1/search",
                        params={"query": query, "per_page": 30,
                                "page": page, "orientation": "landscape"},
                        timeout=15,
                    )
                    r.raise_for_status()
                    photos = r.json().get("photos", [])
                except Exception as e:
                    print(f"  [!] Pexels помилка ({query}): {e}")
                    break

                if not photos:
                    break

                for photo in photos:
                    if downloaded >= n_per_class:
                        break
                    img_url = photo["src"]["large"]
                    fname = os.path.join(dst, f"{season}_{photo['id']}.jpg")
                    if os.path.exists(fname):
                        continue
                    try:
                        img_r = session.get(img_url, timeout=20)
                        img_r.raise_for_status()
                        with open(fname, "wb") as f:
                            f.write(img_r.content)
                        downloaded += 1
                    except:
                        pass

                page += 1

            if downloaded >= n_per_class:
                break

        existing = len([f for f in os.listdir(dst)
                        if f.lower().endswith(('.jpg','.jpeg','.png'))])
        print(f"  {season:<8}: {existing} фото → {dst}")

    print(f"\n[Pexels] Готово! Датасет у {DATA_DIR}/")


def _collect_icrawler(queries: dict, n_per_class: int):
    """Fallback: Google Images через icrawler (без реєстрації)."""
    try:
        from icrawler.builtin import GoogleImageCrawler
    except ImportError:
        print("Встановіть: pip install icrawler")
        print("Або використайте Pexels: python train.py --collect --pexels-key YOUR_KEY")
        return

    print("[icrawler] Збір датасету через Google Images...")

    for season, query_list in queries.items():
        dst = os.path.join(DATA_DIR, season)
        os.makedirs(dst, exist_ok=True)
        per_query = n_per_class // len(query_list)

        for query in query_list:
            print(f"  Пошук: '{query}' ({per_query} фото)...")
            crawler = GoogleImageCrawler(
                storage={"root_dir": dst},
                log_level=50,  # тихий режим
            )
            crawler.crawl(
                keyword=query,
                max_num=per_query,
                filters={"type": "photo", "size": "large"},
            )

        existing = len([f for f in os.listdir(dst)
                        if f.endswith(('.jpg','.jpeg','.png'))])
        print(f"  {season:<8}: {existing} фото → {dst}")

    print(f"\n[icrawler] Готово! Датасет у {DATA_DIR}/")
    print(f"\nДатасет зібрано у {DATA_DIR}/")


# ── Модель: ResNet18 + нова голова ────────────────────────────────────────────
def build_model(freeze=True):
    model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
    if freeze:
        for p in model.parameters():
            p.requires_grad = False
    model.fc = nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4),
        nn.Linear(256, len(SEASONS))
    )
    return model


# ── Навчання ──────────────────────────────────────────────────────────────────
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Пристрій: {device}")

    # Трансформації
    train_tf = transforms.Compose([
        transforms.Resize(256), transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(224), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

    full = datasets.ImageFolder(DATA_DIR, transform=train_tf)
    print(f"Клас→індекс: {full.class_to_idx}")
    print(f"Всього зображень: {len(full)}")

    val_n = max(1, int(len(full) * 0.2))
    train_ds, val_ds = random_split(full, [len(full)-val_n, val_n])
    val_ds.dataset.transform = val_tf

    train_ld = DataLoader(train_ds, BATCH, shuffle=True,  num_workers=0)
    val_ld   = DataLoader(val_ds,   BATCH, shuffle=False, num_workers=0)

    model    = build_model(freeze=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3
    )

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        # --- Розморожуємо backbone з 7-ї епохи ---
        if epoch == 7:
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
            print("  [Epoch 7] fine-tuning увімкнено")

        # Train
        model.train()
        t_corr = t_total = 0
        for imgs, labels in train_ld:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward(); optimizer.step()
            t_corr  += (model(imgs).argmax(1) == labels).sum().item()
            t_total += labels.size(0)

        # Val
        model.eval(); v_corr = v_total = 0
        with torch.no_grad():
            for imgs, labels in val_ld:
                imgs, labels = imgs.to(device), labels.to(device)
                v_corr  += (model(imgs).argmax(1) == labels).sum().item()
                v_total += labels.size(0)

        v_acc = v_corr / v_total
        mark  = ""
        if v_acc > best_acc:
            best_acc = v_acc
            torch.save({"state": model.state_dict(), "classes": SEASONS}, MODEL_OUT)
            mark = " ✓"

        print(f"  Epoch {epoch:>2}/{EPOCHS}  "
              f"train={t_corr/t_total:.3f}  val={v_acc:.3f}{mark}")

    print(f"\nГотово! Модель збережено: {MODEL_OUT}  (val_acc={best_acc:.3%})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--collect", action="store_true",
                        help="Зібрати датасет")
    parser.add_argument("--pexels-key", default=None,
                        help="Pexels API ключ (рекомендовано). "
                             "Отримати безкоштовно: https://www.pexels.com/api/")
    args = parser.parse_args()

    if args.collect:
        collect_dataset(n_per_class=150, pexels_key=args.pexels_key)

    if not os.path.isdir(DATA_DIR):
        print(f"Датасет не знайдено: {DATA_DIR}")
        print("Запустіть: python train.py --collect --pexels-key YOUR_KEY")
    else:
        # Перевіряємо чи є хоч якісь зображення
        total = sum(
            len([f for f in os.listdir(os.path.join(DATA_DIR, s))
                 if f.lower().endswith(('.jpg','.jpeg','.png'))])
            for s in SEASONS
            if os.path.isdir(os.path.join(DATA_DIR, s))
        )
        if total < 10:
            print(f"[!] Датасет порожній або недостатньо зображень ({total}).")
            print("    Запустіть збір: python train.py --collect --pexels-key YOUR_KEY")
        else:
            train()