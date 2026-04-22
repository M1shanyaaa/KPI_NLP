"""
Лабораторна робота №6 | Варіант 11
Тема: Методи і технології глибинного навчання для NLP
Завдання: Скрапінг платформ побутової техніки → агрегація за категоріями
          (комп'ютерна техніка, телевізори, холодильники) →
          порівняльний аналіз пропозицій + аналіз тональності відгуків (ANN)
"""

import json, re, time, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ─── 1. КОНСТАНТИ ───────────────────────────────────────────────────────────
CATEGORIES = ["computers", "tvs", "fridges"]
MAX_WORDS  = 5000
MAX_LEN    = 60
EPOCHS     = 20
BATCH      = 32
DATA_DIR   = "data"
os.makedirs(DATA_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

# ─── 2. СКРАПІНГ ────────────────────────────────────────────────────────────
def scrape_rozetka(category: str, query: str, pages: int = 2) -> list[dict]:
    """Scrape product listings from rozetka.com.ua search."""
    items = []
    for page in range(1, pages + 1):
        url = f"https://rozetka.com.ua/ua/search/?text={query}&page={page}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            cards = soup.select("li.catalog-grid__cell")
            for card in cards:
                title_el = card.select_one("span.goods-tile__title")
                price_el = card.select_one("span.goods-tile__price-value")
                if title_el:
                    items.append({
                        "source": "rozetka",
                        "category": category,
                        "title": title_el.get_text(strip=True),
                        "price": price_el.get_text(strip=True) if price_el else "N/A",
                    })
        except Exception as e:
            print(f"[rozetka] {query} page {page}: {e}")
        time.sleep(0.8)
    return items


def scrape_foxtrot(category: str, slug: str, pages: int = 2) -> list[dict]:
    """Scrape product listings from foxtrot.com.ua."""
    items = []
    for page in range(1, pages + 1):
        url = f"https://www.foxtrot.com.ua/uk/shop/{slug}.html?page={page}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            cards = soup.select("article.product-card")
            for card in cards:
                title_el = card.select_one("a.product-card__title")
                price_el = card.select_one("span.price__title")
                if title_el:
                    items.append({
                        "source": "foxtrot",
                        "category": category,
                        "title": title_el.get_text(strip=True),
                        "price": price_el.get_text(strip=True) if price_el else "N/A",
                    })
        except Exception as e:
            print(f"[foxtrot] {slug} page {page}: {e}")
        time.sleep(0.8)
    return items


def scrape_comfy(category: str, slug: str, pages: int = 2) -> list[dict]:
    """Scrape product listings from comfy.ua."""
    items = []
    for page in range(1, pages + 1):
        url = f"https://comfy.ua/ua/{slug}/?page={page}"
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            soup = BeautifulSoup(r.text, "html.parser")
            cards = soup.select("div.product-item")
            for card in cards:
                title_el = card.select_one("a.product-item__name")
                price_el = card.select_one("span.price__current")
                if title_el:
                    items.append({
                        "source": "comfy",
                        "category": category,
                        "title": title_el.get_text(strip=True),
                        "price": price_el.get_text(strip=True) if price_el else "N/A",
                    })
        except Exception as e:
            print(f"[comfy] {slug} page {page}: {e}")
        time.sleep(0.8)
    return items


def collect_all_products() -> pd.DataFrame:
    """Run scraping for all categories and sources, save JSON + CSV."""
    cfg = {
        "computers": {
            "rozetka_q": "ноутбук",
            "foxtrot_slug": "noutbuki",
            "comfy_slug": "noutbuki",
        },
        "tvs": {
            "rozetka_q": "телевізор",
            "foxtrot_slug": "televizory",
            "comfy_slug": "televizory",
        },
        "fridges": {
            "rozetka_q": "холодильник",
            "foxtrot_slug": "kholodilniki",
            "comfy_slug": "kholodilniki",
        },
    }
    all_items = []
    for cat, params in cfg.items():
        print(f"\n>>> Scraping category: {cat}")
        all_items += scrape_rozetka(cat, params["rozetka_q"])
        all_items += scrape_foxtrot(cat, params["foxtrot_slug"])
        all_items += scrape_comfy(cat, params["comfy_slug"])

    df = pd.DataFrame(all_items)
    if df.empty:
        print("[!] Scraping returned no data — generating synthetic dataset for demo.")
        df = generate_synthetic(n=300)

    df.to_csv(f"{DATA_DIR}/products.csv", index=False)
    df.to_json(f"{DATA_DIR}/products.json", orient="records", force_ascii=False)
    print(f"\n✓ Saved {len(df)} products to {DATA_DIR}/")
    return df


# ─── 3. СИНТЕТИЧНІ ДАНІ (fallback) ──────────────────────────────────────────
def generate_synthetic(n: int = 300) -> pd.DataFrame:
    """Generate realistic synthetic product data when scraping fails."""
    rng = np.random.default_rng(42)
    sources = ["rozetka", "foxtrot", "comfy"]
    templates = {
        "computers": [
            "Ноутбук ASUS VivoBook 15 Intel Core i{} 8GB RAM",
            "Laptop Lenovo IdeaPad 3 AMD Ryzen {} 16GB",
            "Ноутбук HP Pavilion 15 Intel i{} 512GB SSD",
        ],
        "tvs": [
            'Телевізор Samsung {}\" 4K UHD Smart TV',
            'LG OLED {}\" webOS HDR10',
            'TCL {}\" QLED Android TV',
        ],
        "fridges": [
            "Холодильник Bosch KGN{}XL NoFrost",
            "Холодильник LG GBB{}SWUGN Total No Frost",
            "Холодильник Samsung RB{}FHDACBC",
        ],
    }
    rows = []
    for cat, tmpl_list in templates.items():
        for _ in range(n // 3):
            tmpl = rng.choice(tmpl_list)
            suffix = int(rng.integers(3, 9))
            rows.append({
                "source": rng.choice(sources),
                "category": cat,
                "title": tmpl.format(suffix),
                "price": f"{rng.integers(5000, 80000)} грн",
            })
    return pd.DataFrame(rows)


# ─── 4. АНАЛІЗ ТОНАЛЬНОСТІ — СИНТЕТИЧНІ ВІДГУКИ ────────────────────────────
POSITIVE_REVIEWS = [
    "Відмінний товар, дуже задоволений покупкою! Рекомендую всім друзям.",
    "Чудова якість, швидка доставка, все як описано на сайті.",
    "Купив ноутбук тиждень тому — повністю відповідає очікуванням, рекомендую!",
    "Холодильник працює тихо, економить електроенергію, відмінна покупка.",
    "Телевізор з неймовірною картинкою, дуже задоволений якістю зображення.",
    "Excellent product, works perfectly from day one. Highly recommend!",
    "Great quality laptop, fast delivery, exactly as described.",
    "Amazing TV picture quality, very satisfied with this purchase.",
    "The fridge is quiet and energy efficient. Best purchase this year!",
    "Superb build quality, fast shipping, 100% recommend to everyone.",
    "Товар прийшов швидко, упакований добре, якість на висоті.",
    "Задоволений на всі 100%, буду купувати тут ще раз.",
    "Пречудова техніка, працює бездоганно вже кілька місяців.",
    "Неймовірно швидкий ноутбук за такі гроші — просто знахідка!",
    "Дуже гарний телевізор, зображення чітке, звук потужний.",
]

NEGATIVE_REVIEWS = [
    "Жахлива якість, зламався через тиждень після покупки. Не рекомендую!",
    "Розчарований повністю, товар не відповідає опису на сайті.",
    "Холодильник шумить вночі, не можна спати. Витрачені гроші.",
    "Телевізор перестав вмикатись через місяць, жахливий сервіс.",
    "Ноутбук грівся до неможливості, повернув через тиждень.",
    "Terrible quality, broke after a week. Total waste of money.",
    "Very disappointed, product does not match the description at all.",
    "The fridge makes loud noise at night, completely unusable.",
    "Stopped working after one month. Awful customer service experience.",
    "Laptop overheats constantly, returned it after just one week.",
    "Не раджу нікому, обслуговування жахливе, товар бракований.",
    "Повернув через 3 дні — не працювало з коробки.",
    "Найгірша покупка в моєму житті, шкодую що замовив.",
    "Absolutely horrible product, requested refund immediately.",
    "Do not buy this! Stopped functioning within days of purchase.",
]

NEUTRAL_REVIEWS = [
    "Непогано за свою ціну, є певні недоліки але загалом нормально.",
    "Середня якість, нічого особливого, відповідає своїй вартості.",
    "Ноутбук як ноутбук, без захоплення, але і без нарікань.",
    "Average product for the price, nothing special but works fine.",
    "Middle range quality, some minor issues but generally acceptable.",
    "Телевізор нормальний, не вражає але і не розчаровує.",
    "За такі гроші прийнятна якість, але є кращі варіанти.",
    "Decent fridge, does the job, nothing to write home about.",
    "Satisfactory performance overall, meets basic expectations.",
    "Нормальна техніка без особливих вражень, можна брати.",
]

def generate_reviews(n: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    # Build balanced-ish dataset with realistic augmentation
    for _ in range(n):
        label = rng.choice(["positive", "negative", "neutral"], p=[0.40, 0.40, 0.20])
        if label == "positive":
            base = rng.choice(POSITIVE_REVIEWS)
        elif label == "negative":
            base = rng.choice(NEGATIVE_REVIEWS)
        else:
            base = rng.choice(NEUTRAL_REVIEWS)
        # Light augmentation: sometimes add a product mention
        products = ["ноутбук", "телевізор", "холодильник", "laptop", "TV", "fridge"]
        if rng.random() > 0.5:
            prod = rng.choice(products)
            base = f"{prod.capitalize()}: {base}"
        rows.append({"text": base, "label": label})
    df = pd.DataFrame(rows)
    df.to_csv(f"{DATA_DIR}/reviews.csv", index=False)
    return df


# ─── 5. ПОБУДОВА ТА НАВЧАННЯ МОДЕЛІ ─────────────────────────────────────────
def build_sentiment_model(vocab_size: int, num_classes: int) -> Sequential:
    model = Sequential([
        Embedding(vocab_size, 64, input_length=MAX_LEN),
        GlobalAveragePooling1D(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def train_sentiment(reviews_df: pd.DataFrame):
    le = LabelEncoder()
    y  = le.fit_transform(reviews_df["label"])

    tok = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tok.fit_on_texts(reviews_df["text"])
    seqs = tok.texts_to_sequences(reviews_df["text"])
    X    = pad_sequences(seqs, maxlen=MAX_LEN, padding="post")

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    model = build_sentiment_model(MAX_WORDS, len(le.classes_))
    model.summary()

    history = model.fit(
        X_tr, y_tr,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=BATCH,
        verbose=1,
    )

    # Evaluate
    loss, acc = model.evaluate(X_te, y_te, verbose=0)
    print(f"\n✓ Test accuracy: {acc:.4f}  |  Test loss: {loss:.4f}")

    y_pred = np.argmax(model.predict(X_te, verbose=0), axis=1)
    print("\nClassification Report:")
    print(classification_report(y_te, y_pred, target_names=le.classes_))

    return model, tok, le, history, (X_te, y_te, y_pred)


# ─── 6. ВІЗУАЛІЗАЦІЯ ─────────────────────────────────────────────────────────
def plot_training(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history.history["accuracy"],     label="train")
    axes[0].plot(history.history["val_accuracy"], label="val")
    axes[0].set_title("Accuracy"); axes[0].legend()
    axes[1].plot(history.history["loss"],     label="train")
    axes[1].plot(history.history["val_loss"], label="val")
    axes[1].set_title("Loss"); axes[1].legend()
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/training_curves.png", dpi=120)
    plt.show()
    print("✓ Saved training_curves.png")


def plot_confusion(y_te, y_pred, classes):
    cm = confusion_matrix(y_te, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Sentiment")
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/confusion_matrix.png", dpi=120)
    plt.show()
    print("✓ Saved confusion_matrix.png")


def plot_category_comparison(products_df: pd.DataFrame):
    counts = products_df.groupby(["category", "source"]).size().unstack(fill_value=0)
    counts.plot(kind="bar", figsize=(9, 5), colormap="Set2")
    plt.title("Кількість пропозицій за категоріями та платформами")
    plt.ylabel("Кількість")
    plt.xlabel("Категорія")
    plt.xticks(rotation=0)
    plt.legend(title="Платформа")
    plt.tight_layout()
    plt.savefig(f"{DATA_DIR}/category_comparison.png", dpi=120)
    plt.show()
    print("✓ Saved category_comparison.png")


# ─── 7. PREDICT DEMO ─────────────────────────────────────────────────────────
def predict_demo(model, tok, le):
    samples = [
        "Відмінний холодильник, дуже задоволений покупкою!",
        "Телевізор зламався через тиждень, жахлива якість",
        "Ноутбук непоганий за свою ціну, є певні недоліки",
        "Great laptop, highly recommend for daily use",
        "Waste of money, broke after a week",
    ]
    seqs = tok.texts_to_sequences(samples)
    X    = pad_sequences(seqs, maxlen=MAX_LEN, padding="post")
    preds = model.predict(X, verbose=0)
    print("\n─── Demo Predictions ───")
    for text, probs in zip(samples, preds):
        label = le.classes_[np.argmax(probs)]
        conf  = np.max(probs)
        print(f"  [{label:8s} {conf:.2f}] {text[:60]}")


# ─── 8. ГОЛОВНА ФУНКЦІЯ ──────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Лабораторна робота №6 — Variant 11")
    print("=" * 60)

    # 8.1 Збір даних
    products = collect_all_products()
    print("\n── Product stats ──")
    print(products.groupby(["category", "source"]).size().to_string())

    # 8.2 Порівняльний аналіз
    plot_category_comparison(products)

    # 8.3 Відгуки
    reviews = generate_reviews(1500)
    print(f"\n✓ Reviews distribution:\n{reviews['label'].value_counts().to_string()}")

    # 8.4 Навчання
    model, tok, le, history, eval_data = train_sentiment(reviews)

    # 8.5 Графіки
    X_te, y_te, y_pred = eval_data
    plot_training(history)
    plot_confusion(y_te, y_pred, le.classes_)

    # 8.6 Демо
    predict_demo(model, tok, le)

    # 8.7 Зберігаємо модель
    model.save(f"{DATA_DIR}/sentiment_model.keras")
    print(f"\n✓ Model saved → {DATA_DIR}/sentiment_model.keras")
    print("\n✓ Done. All outputs in ./data/")


if __name__ == "__main__":
    main()