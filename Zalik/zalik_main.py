"""
Моніторинг новинних стрічок України
"""

import feedparser
import re
import time
import math
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── 1. ДЖЕРЕЛА (з fallback для Радіо Свобода) ─────────────────────────

FEEDS = {
    "Укрінформ": ["https://www.ukrinform.ua/rss/block-lastnews"],
    "BBC": ["https://www.bbc.com/ukrainian/index.xml"],
    "Суспільне": ["https://suspilne.media/rss/all.rss"],
    "Укр. правда": ["https://www.pravda.com.ua/rss/"],
}

# ── 2. СЛОВНИКИ (коріння — substring match) ─────────────────────────

CATEGORIES = {
    "Політика": [
        "президент", "зеленськ", "уряд", "кабмін", "верховн", "парламент",
        "міністр", "депутат", "партi", "партій", "коаліц",
        "вибор", "реформ", "закон", "законопроект", "постанов", "указ",
        "голосуван", "конституц",
        "дипломат", "посол", "переговор", "санкц", "євросоюз", "євро",
        "нато", "оон", "зустріч", "саміт", "договір", "угод",
        "мер", "губернатор", "держав", "влад", "політик",
        "байден", "трамп", "макрон", "шольц", "столтенберг",
    ],
    "Безпека": [
        "війн", "бойов", "фронт", "окупант", "ворог",
        "обстріл", "ракет", "удар", "атак", "наступ", "оборон",
        "дрон", "бпла", "танк", "зброя", "артилер",
        "пво", "протиповітр", "балістич", "крилат",
        "загибел", "поранен", "жертв", "евакуац", "руйнуван",
        "збу", "поліц", "нацгвард", "армі", "бригад", "батальйон",
        "розвідк", "тероборон", "захист",
        "сирен", "тривог", "вибух", "пожеж", "мін",
        "полон", "звільнен", "деоккупац", "херсон", "запоріжж",
        "харків", "донецьк", "луганськ", "бахмут", "авдіїв",
    ],
    "Економіка": [
        "економік", "ввп", "інфляц", "бюджет", "держборг",
        "гривн", "валют", "курс", "долар", "нбу", "банк",
        "кредит", "мвф", "світовий банк",
        "ринок", "бізнес", "підприємств", "компані", "корпорац",
        "інвестиц", "капітал", "акці",
        "зарплат", "пенсі", "субсид", "тариф", "ціна", "подорожч",
        "податок", "податков", "митн",
        "енергетик", "газ", "нафт", "зерн", "агро", "урожай",
        "торгівл", "експорт", "імпорт",
        "відновлен", "реконструкц", "допомог",
    ],
}

STOPWORDS = {
    "і", "та", "що", "як", "від", "до", "з", "у", "в", "на", "за",
    "по", "про", "для", "але", "це", "він", "вона", "вони", "ми",
    "ви", "не", "а", "бо", "де", "коли", "якщо", "через", "після",
    "під", "над", "між", "при", "без", "яка", "який", "яке", "які",
    "його", "її", "їх", "був", "була", "було", "були", "буде",
    "є", "або", "чи", "вже", "ще", "також", "лише", "тільки",
    "більше", "менше", "дуже", "так", "ні", "більш", "може",
    "навіть", "саме", "нові", "нова", "новий", "нових",
}


# ── 3. ЗБІР НОВИН ────────────────────────────────────────────────

def fetch_news(feeds, limit=50):
    articles = []
    for source, urls in feeds.items():
        print(f"  ↳ {source}...")
        entries = []
        for url in urls:
            try:
                feed = feedparser.parse(url)
                entries = feed.entries
                if entries:
                    print(f"      OK ({len(entries)} записів): {url}")
                    break
                else:
                    print(f"      порожньо: {url}")
            except Exception as e:
                print(f"      помилка {url}: {e}")
            time.sleep(0.3)

        for entry in entries[:limit]:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            full = f"{title} {title} {title} {summary}"
            articles.append({"source": source, "title": title, "text": full})
        time.sleep(0.4)

    print(f"\n  Зібрано {len(articles)} статей\n")
    return articles


# 4. ПРЕПРОЦЕСІНГ ──────────────────────────────────────

def clean(text):
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    return text.lower()


def tokenize(text):
    return [t for t in clean(text).split() if t not in STOPWORDS and len(t) > 2]


# 5. КЛАСИФІКАЦІЯ ──────────────
THRESHOLD = 0.38
MIN_SCORE = 2


def score_text(text, title=""):
    text_cl = clean(text)
    title_cl = clean(title)
    scores = {cat: 0.0 for cat in CATEGORIES}

    for cat, keywords in CATEGORIES.items():
        for kw in keywords:
            if kw in text_cl:
                scores[cat] += 1.0
            if kw in title_cl:
                scores[cat] += 1.0

    total = sum(scores.values())

    if total < MIN_SCORE:
        return {cat: round(1 / 3, 3) for cat in CATEGORIES}, "Невизначено"

    probs = {cat: round(scores[cat] / total, 3) for cat in CATEGORIES}
    sorted_cats = sorted(probs, key=probs.get, reverse=True)
    best, second = sorted_cats[0], sorted_cats[1]

    if probs[best] >= THRESHOLD:
        dominant = best
    elif probs[best] - probs[second] < 0.08:
        dominant = "Мішане"
    else:
        dominant = best  # відносний лідер

    return probs, dominant


# 6. TF-IDF ─────────────────────────

def compute_tfidf(articles):
    N = len(articles)
    tf_list, df = [], Counter()
    for art in articles:
        tokens = tokenize(art["text"])
        tf = Counter(tokens)
        tf_list.append(tf)
        df.update(set(tokens))

    tfidf_global = Counter()
    for tf in tf_list:
        total = sum(tf.values()) or 1
        for word, cnt in tf.items():
            tfidf_global[word] += (cnt / total) * math.log((N + 1) / (df[word] + 1))
    return tfidf_global


# ── 7. АНАЛІЗ ──────────────────────────────────────────────

def analyse(articles):
    all_tokens = []
    source_counts = Counter()
    cat_counts = Counter()
    cat_by_source = {src: Counter() for src in FEEDS}
    cat_probs_all = defaultdict(list)

    for art in articles:
        all_tokens.extend(tokenize(art["text"]))
        source_counts[art["source"]] += 1
        probs, dominant = score_text(art["text"], art["title"])
        for cat, p in probs.items():
            cat_probs_all[cat].append(p)
        cat_counts[dominant] += 1
        if dominant not in ("Невизначено", "Мішане"):
            cat_by_source[art["source"]][dominant] += 1

    return all_tokens, source_counts, cat_counts, cat_by_source, cat_probs_all


# ── 8. ВІЗУАЛІЗАЦІЯ ─────────────────────────────────────────────

CAT_COLORS = {"Політика": "#4a7fcb", "Безпека": "#e05a5a", "Економіка": "#5ab87e"}
SRC_COLORS = ["#4a7fcb", "#e05a5a", "#5ab87e", "#f4a340"]
CATS_MAIN = ["Політика", "Безпека", "Економіка"]


def plot_all(all_tokens, source_counts, cat_counts, cat_by_source, cat_probs_all, tfidf):
    fig = plt.figure(figsize=(18, 13), facecolor="#f8f9fa")
    fig.suptitle(
        f"Моніторинг українських новин · {datetime.now().strftime('%d.%m.%Y %H:%M')}",
        fontsize=15, fontweight="bold", y=0.98,
    )
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.38)

    #  TF-IDF топ-20
    ax1 = fig.add_subplot(gs[0, :2])
    cat_kws = {kw for kws in CATEGORIES.values() for kw in kws}
    top20 = [(w, s) for w, s in tfidf.most_common(60) if w not in cat_kws][:20]
    words, scores = zip(*top20)
    bars = ax1.barh(words[::-1], scores[::-1], color="#4a7fcb", alpha=0.85)
    ax1.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    ax1.set_title("Частотний аналіз: топ-20 слів (TF-IDF)", fontweight="bold")
    ax1.set_xlabel("TF-IDF сума")
    ax1.set_facecolor("#f0f4ff")

    # 8.2 Статей за джерелом
    ax2 = fig.add_subplot(gs[0, 2])
    srcs = [s for s in source_counts if source_counts[s] > 0]
    cnts = [source_counts[s] for s in srcs]
    ax2.pie(cnts, labels=srcs, autopct="%1.0f%%", colors=SRC_COLORS[:len(srcs)],
            startangle=140, textprops={"fontsize": 8},
            wedgeprops={"linewidth": 1.5, "edgecolor": "white"})
    ax2.set_title("Статей за джерелом", fontweight="bold")

    # 8.3 Середнє P(cat) ±σ
    ax3 = fig.add_subplot(gs[1, 0])
    means = [np.mean(cat_probs_all[c]) for c in CATS_MAIN]
    stds = [np.std(cat_probs_all[c]) for c in CATS_MAIN]
    colors3 = [CAT_COLORS[c] for c in CATS_MAIN]
    bars3 = ax3.bar(CATS_MAIN, means, yerr=stds, color=colors3, alpha=0.85,
                    capsize=7, edgecolor="white", linewidth=1.5)
    ax3.bar_label(bars3, fmt="%.2f", padding=6, fontsize=10, fontweight="bold")
    ax3.set_ylim(0, max(means) * 1.6)
    ax3.set_title("Середня P(категорія | стаття) ±σ", fontweight="bold")
    ax3.set_ylabel("Ймовірність")
    ax3.set_facecolor("#f5f0ff")

    # 8.4 Категоріальний розподіл
    ax4 = fig.add_subplot(gs[1, 1])
    label_map = {"Мішане": "#aaaaaa", "Невизначено": "#cccccc"}
    cats_all = [c for c in cat_counts if cat_counts[c] > 0]
    vals_all = [cat_counts[c] for c in cats_all]
    pie_colors = [CAT_COLORS.get(c, label_map.get(c, "#888888")) for c in cats_all]
    ax4.pie(vals_all, labels=cats_all, autopct="%1.1f%%",
            colors=pie_colors, startangle=90,
            textprops={"fontsize": 9},
            wedgeprops={"linewidth": 1.5, "edgecolor": "white"})
    ax4.set_title("Категоріальна\nналежність статей", fontweight="bold")

    # 8.5 Grouped bar: категорії × джерела
    ax5 = fig.add_subplot(gs[1, 2])
    active_srcs = [s for s in FEEDS if source_counts[s] > 0]
    x = np.arange(len(active_srcs))
    width = 0.22
    for i, cat in enumerate(CATS_MAIN):
        vals = [cat_by_source[src].get(cat, 0) for src in active_srcs]
        ax5.bar(x + i * width, vals, width, label=cat,
                color=CAT_COLORS[cat], alpha=0.85, edgecolor="white")
    ax5.set_xticks(x + width)
    ax5.set_xticklabels([s.replace(" ", "\n") for s in active_srcs], fontsize=7)
    ax5.set_title("Категорії\nза джерелами", fontweight="bold")
    ax5.set_ylabel("Статей")
    ax5.legend(fontsize=8)
    ax5.set_facecolor("#f0fff4")

    plt.savefig("news_analysis.png", dpi=150, bbox_inches="tight")
    print("✓ Збережено → news_analysis.png")
    plt.show()


# ── 9. MAIN ────────────────────────────

def main():
    print("Моніторинг новинних стрічок\n")
    articles = fetch_news(FEEDS, limit=40)

    if not articles:
        print("[!] Статті не знайдено.")
        return

    tfidf = compute_tfidf(articles)
    all_tokens, source_counts, cat_counts, cat_by_source, cat_probs_all = analyse(articles)

    # Таблиця
    header = f"{'Джерело':<22} {'Статей':>7} {'Пол':>6} {'Без':>6} {'Ек':>6}  P(Пол) P(Без) P(Ек)"
    print(f"\n{header}")
    print("-" * len(header))
    for src in FEEDS:
        n = source_counts[src]
        p = cat_by_source[src].get("Політика", 0)
        b = cat_by_source[src].get("Безпека", 0)
        e = cat_by_source[src].get("Економіка", 0)
        src_arts = [a for a in articles if a["source"] == src]
        if src_arts:
            def mp(cat):
                return np.mean([score_text(a["text"], a["title"])[0][cat] for a in src_arts])

            pp, pb, pe = mp("Політика"), mp("Безпека"), mp("Економіка")
        else:
            pp = pb = pe = 0.0
        print(f"{src:<22} {n:>7} {p:>6} {b:>6} {e:>6}  {pp:.2f}  {pb:.2f}  {pe:.2f}")
    print("-" * len(header))
    total = sum(cat_counts.values())
    for cat in list(CATS_MAIN) + ["Мішане", "Невизначено"]:
        n = cat_counts.get(cat, 0)
        print(f"  {cat:<14}: {n:>3}  ({100 * n / total:.1f}%)")
    print(f"\nУсього токенів: {len(all_tokens)} | унікальних: {len(set(all_tokens))}")

    plot_all(all_tokens, source_counts, cat_counts, cat_by_source, cat_probs_all, tfidf)


if __name__ == "__main__":
    main()
