"""
Лабораторна робота №4 — Варіант 11
NLP: Класифікація новин (Реальне / Фейк)
Рівень: ІІІ (10 балів)

Встановити залежності:
    pip install feedparser requests beautifulsoup4 scikit-learn nltk numpy matplotlib pandas
"""

import re, warnings
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import requests, feedparser
from bs4 import BeautifulSoup
import nltk
from nltk.util import bigrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from matplotlib.patches import Patch

warnings.filterwarnings("ignore")
for pkg in ("punkt", "punkt_tab"):
    nltk.download(pkg, quiet=True)

STOPWORDS = {
    "і","та","в","у","на","з","що","як","але","або","а","це","він","вона",
    "вони","ми","ви","я","до","по","від","за","не","так","ще","вже","якщо",
    "де","коли","то","є","бути","про","для","при","під","над","між","через",
    "після","зі","більш","менш","дуже","тут","там","також","із","були","всі",
    "які","яка","який","яке","його","її","їх","цей","ця","ці","цього","того",
    "вже","коли","після","перед","тому","адже","однак","проте","якщо",
}

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# RSS стрічки українських новинних сайтів
RSS_FEEDS = [
    "https://www.ukrinform.ua/rss/block-lastnews",
    "https://www.pravda.com.ua/rss/view_news/",
    "https://suspilne.media/rss",
    "https://rss.unian.net/site/news_ukr.rss",
    "https://www.radiosvoboda.org/api/zrqiteuuiq",
    "https://www.bbc.com/ukrainian/index.xml",
    "https://www.dw.com/uk/rss",
]


# ── Скрапінг через RSS ────────────────────────────────────────────────

def scrape_rss(max_total=60) -> list[dict]:
    """Збирає реальні новини з RSS-стрічок українських ЗМІ."""
    articles = []

    for url in RSS_FEEDS:
        if len(articles) >= max_total:
            break
        try:
            r = requests.get(url, headers=HEADERS, timeout=10)
            feed = feedparser.parse(r.content)

            if not feed.entries:
                print(f"  [пусто] {url}")
                continue

            count = 0
            for entry in feed.entries:
                title   = entry.get("title", "").strip()
                summary = entry.get("summary", "") or entry.get("description", "")
                # Очищення HTML-тегів із summary
                summary = BeautifulSoup(summary, "html.parser").get_text(" ", strip=True)

                text = (title + ". " + summary).strip()
                if len(text) > 80:
                    articles.append({"text": text, "label": "real"})
                    count += 1

            print(f"  ✓ {url.split('/')[2]:<30} {count} статей")

        except Exception as e:
            print(f"  ✗ {url.split('/')[2]:<30} {e}")

    return articles


# ── Синтетичні фейки (за завданням варіанту 11) ───────────────────────

def make_fake() -> list[dict]:
    persons = ["Президент","Прем'єр","Міністр оборони","Народний депутат","Мер Києва"]
    entities = ["ФСБ","ЦРУ","МВФ","Ватикан","НАТО","Пентагон","Ротшильди"]
    topics   = ["COVID-19","вакцинації","5G","мікрочіпів","виборів"]
    suffixes = [
        " Поширте до видалення!",
        " Офіційні ЗМІ мовчать.",
        " Влада вимагає видалити цей пост.",
        " Анонімне джерело підтвердило.",
        " Кількість переглядів вже перевищила мільйон.",
    ]
    templates = [
        "СЕНСАЦІЯ! {p} таємно зустрівся з {e}. Уряд приховує правду від народу.",
        "ВИКЛЮЧНО! Вчені довели, що {t} смертельно небезпечне. Медицина мовчить.",
        "ШОКУЮЧЕ! {p} зізнався у корупції в прямому ефірі. Відео намагаються видалити.",
        "ТЕРМІНОВО! {e} планує масові скорочення. 100 тисяч лишаться без роботи.",
        "НЕЙМОВІРНО! Ліки лікують хворобу за три дні. Фармацевти блокують інформацію.",
        "УВАГА! {p} виявився агентом {e}. Докази вже передані до суду.",
        "СКАНДАЛ! Знайдено секретні документи про {t}. Влада приховує факти.",
        "ПРОРИВ! Вчені відкрили безмежне джерело енергії. {e} блокують публікацію.",
        "ТАЄМНИЦЯ! {e} фінансує терористів. Банки знають вже кілька років.",
        "ЕКСКЛЮЗИВ! {p} тікає закордон із мільярдами. Прокуратура ігнорує докази.",
        "ВІДКРИТТЯ! Мікрочіпи у вакцинах підтверджено лабораторно. Мільйони під загрозою.",
        "ТЕРМІНОВО! {e} готує державний переворот. Перехоплені переговори — доказ.",
        "ШОКУЮЧІ ФАКТИ! {p} веде подвійне таємне життя. Журналісти розкрили схему.",
        "ВИКЛЮЧНО! Заборонені дані про {t} злиті анонімними хакерами.",
        "УВАГА! Нова хвиля {t} охопить Україну через два тижні. Склади вже порожні.",
        "СКАНДАЛ У ВЛАДІ! {p} і {e} мають спільний офшорний рахунок.",
        "ТЕРМІНОВО! Армія готує введення надзвичайного стану. Списки вже складаються.",
        "РОЗКРИТТЯ! {t} — секретна програма контролю свідомості населення.",
        "СЕНСАЦІЯ! {p} підписав таємну угоду з {e}. Суверенітет під загрозою.",
        "ШОКУЮЧИЙ ФАКТ! {t} призводить до зомбування. Перевірено на тисячах людей.",
        "ВИКЛЮЧНО! {e} встановив шпигунські пристрої в телефонах мільйонів українців.",
        "СКАНДАЛ! Офіційна статистика {t} сфальсифікована на 80 відсотків.",
        "ТЕРМІНОВО! Вибухи під Києвом — уряд приховує катастрофу вже тиждень.",
        "ТАЄМНИЦЯ! {p} зник на два тижні. Двійник проводив зустрічі замість нього.",
        "ЕКСКЛЮЗИВ! {p} продав державні секрети {e} за 500 мільйонів доларів.",
    ]
    rng = np.random.default_rng(42)
    result = []
    for tpl in templates:
        text = tpl.format(
            p=str(rng.choice(persons)),
            e=str(rng.choice(entities)),
            t=str(rng.choice(topics)),
        ) + str(rng.choice(suffixes))
        result.append({"text": text, "label": "fake"})
    return result


# ── Обробка тексту ────────────────────────────────────────────────────

def clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^а-яіїєa-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def tokenize(text: str) -> list:
    return [w for w in clean(text).split() if len(w) > 2 and w not in STOPWORDS]


# ── Головна функція ───────────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  ЛР №4 — Варіант 11: Класифікація Реальне / Фейк")
    print("=" * 55)

    # 1. Збір даних
    print("\n[1] Збір реальних новин через RSS...")
    real = scrape_rss(max_total=60)

    if not real:
        print("\n  ПОМИЛКА: жоден RSS недоступний.")
        print("  Перевір інтернет-з'єднання та спробуй знову.")
        return

    fake = make_fake()
    # Балансування: щоб кількість реальних ≈ кількість фейків × 2 (або рівно)
    n = min(len(real), len(fake) * 2)
    real = real[:n]

    print(f"\n  Реальних: {len(real)} | Фейкових: {len(fake)}")

    all_data = real + fake
    texts  = [a["text"]  for a in all_data]
    labels = [a["label"] for a in all_data]

    # Токени по класах
    real_tok = [w for t, l in zip(texts, labels) if l == "real" for w in tokenize(t)]
    fake_tok = [w for t, l in zip(texts, labels) if l == "fake" for w in tokenize(t)]

    # 2. TF-IDF
    vect = TfidfVectorizer(
        preprocessor=clean,
        tokenizer=lambda x: tokenize(x),
        max_features=1000,
        ngram_range=(1, 2),
        token_pattern=None,
    )
    X = vect.fit_transform(texts)
    y = [0 if l == "fake" else 1 for l in labels]

    # 3. Класифікація з вчителем (SVM)
    print("\n[2] Класифікація з вчителем (SVM)...")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    clf = LinearSVC(C=1.0, max_iter=3000, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    print(f"    Точність: {acc:.1%}  {'✓ > 60%' if acc >= 0.6 else '✗'}")
    print(classification_report(y_te, y_pred, target_names=["fake", "real"]))

    # 4. Кластеризація без вчителя (K-Means)
    print("[3] Кластеризація без вчителя (K-Means, k=2)...")
    km = KMeans(n_clusters=2, random_state=42, n_init=10)
    cl = km.fit_predict(X)
    for c in range(2):
        mask = [i for i, v in enumerate(cl) if v == c]
        nr = sum(labels[i] == "real" for i in mask)
        nf = sum(labels[i] == "fake" for i in mask)
        print(f"    Кластер {c}: {len(mask)} статей | реал: {nr} | фейк: {nf}")

    # 5. Частотний аналіз
    print("\n[4] Частотний аналіз:")
    print("  Топ-10 слів (реальні):")
    for w, c in Counter(real_tok).most_common(10):
        print(f"    {w:<22} {c}")
    print("  Топ-10 слів (фейкові):")
    for w, c in Counter(fake_tok).most_common(10):
        print(f"    {w:<22} {c}")
    print("  Топ-5 біграм (реальні):")
    for (a, b), c in Counter(bigrams(real_tok)).most_common(5):
        print(f"    {a+' '+b:<30} {c}")
    print("  Топ-5 біграм (фейкові):")
    for (a, b), c in Counter(bigrams(fake_tok)).most_common(5):
        print(f"    {a+' '+b:<30} {c}")

    # ── Графіки ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    fig.suptitle("ЛР №3 — Варіант 11: Класифікація Реальне / Фейк", fontweight="bold")

    # TF-IDF
    feat = vect.get_feature_names_out()
    Xd = pd.DataFrame(X.toarray(), columns=feat)
    Xd["label"] = labels
    top_real = Xd[Xd.label == "real"].drop("label", axis=1).mean().nlargest(10)
    top_fake = Xd[Xd.label == "fake"].drop("label", axis=1).mean().nlargest(10)
    ax = axes[0]
    pos = range(10)
    ax.barh(list(pos),  top_real.values[::-1], color="#2176AE", alpha=0.8, label="Реальні")
    ax.barh(list(pos), -top_fake.values[::-1], color="#C1121F", alpha=0.8, label="Фейкові")
    ax.set_yticks(list(pos))
    ax.set_yticklabels(top_real.index[::-1], fontsize=8)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("TF-IDF топ-слова", fontweight="bold")
    ax.legend(fontsize=8)
    ax.set_xlabel("← Фейк   |   Реал →")

    # Розподіл довжини слів
    ax = axes[1]
    r_len = [len(w) for w in real_tok]
    f_len = [len(w) for w in fake_tok]
    ax.hist(r_len, bins=range(2, 22), alpha=0.65, color="#2176AE",
            label=f"Реальні (сер.={np.mean(r_len):.1f})", density=True)
    ax.hist(f_len, bins=range(2, 22), alpha=0.65, color="#C1121F",
            label=f"Фейкові (сер.={np.mean(f_len):.1f})", density=True)
    ax.set_title("Розподіл довжини слів", fontweight="bold")
    ax.set_xlabel("Символів у слові")
    ax.set_ylabel("Щільність")
    ax.legend(fontsize=8)

    # K-Means PCA
    ax = axes[2]
    pca = PCA(n_components=2, random_state=42)
    X2d = pca.fit_transform(X.toarray())
    colors = ["#2176AE" if l == "real" else "#C1121F" for l in labels]
    ax.scatter(X2d[:, 0], X2d[:, 1], c=colors, alpha=0.55, s=40)
    centers = pca.transform(km.cluster_centers_)
    ax.scatter(centers[:, 0], centers[:, 1], c="black", marker="*", s=300, zorder=5)
    ax.set_title("K-Means (PCA 2D)", fontweight="bold")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.legend(handles=[
        Patch(color="#2176AE", label="Реальні"),
        Patch(color="#C1121F", label="Фейкові"),
    ], fontsize=8)

    plt.tight_layout()
    plt.savefig("results.png", bbox_inches="tight", dpi=130)
    plt.show()
    print("\n[ГОТОВО] Збережено: results.png")


if __name__ == "__main__":
    main()