"""
ЛР №2 — NLP: Токенізація та нормалізація
Варіант 11/12: Домінантні вимоги до Data Analyst

Встановлення:
    pip install requests beautifulsoup4 matplotlib numpy nltk spacy
    python -m spacy download en_core_web_sm
    python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
"""

import os, re, json, time, random
from collections import Counter

import requests
from bs4 import BeautifulSoup
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

import nltk
for _r in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(_r, quiet=True)
from nltk.tokenize import word_tokenize, sent_tokenize, MWETokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer, SnowballStemmer

import spacy
from spacy.lang.en.stop_words import STOP_WORDS as SPACY_STOP

# ─── НАЛАШТУВАННЯ ───────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
HEADERS    = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"}
os.makedirs(OUTPUT_DIR, exist_ok=True)

COLOR_PRIMARY   = "#4C72B0"
COLOR_SECONDARY = "#DD8452"
COLOR_SUCCESS   = "#55A868"
COLOR_BG        = "#F8F9FA"
COLOR_GRID      = "#E0E0E0"

nlp = spacy.load("en_core_web_sm")

# ════════════════════════════════════════════════════════════════════════════
#  СКРАПІНГ
# ════════════════════════════════════════════════════════════════════════════
def fetch_page(url):
    try:
        r = requests.get(url, headers=HEADERS, timeout=12)
        r.raise_for_status()
        time.sleep(random.uniform(0.9, 1.3))
        return r
    except Exception as e:
        print(f"  [!] {url[:55]} → {e}"); return None

def scrape_jobs(list_url, item_sel, desc_sel, base=""):
    jobs, r = [], fetch_page(list_url)
    if not r: return jobs
    links = [(a.get_text(strip=True), base + a["href"])
             for a in BeautifulSoup(r.text, "html.parser").select(item_sel) if a.get("href")]
    print(f"  {list_url[:45]}… → {len(links)} вакансій")
    for title, url in links[:15]:
        r2 = fetch_page(url)
        if not r2: continue
        d = BeautifulSoup(r2.text, "html.parser").select_one(desc_sel)
        text = d.get_text(" ", strip=True) if d else ""
        if len(text) > 80:
            jobs.append({"source": base or list_url[:20], "title": title, "text": text})
            print(f"    ✓ {title[:55]}")
    return jobs

def collect_all_jobs():
    print("\n── ЗБІР ВАКАНСІЙ ──────────────────────────────────────────────")
    jobs = (
        scrape_jobs("https://jobs.dou.ua/vacancies/?search=data+analyst&category=Data+Science",
                    ".l-vacancy .title a", ".vacancy-section") +
        scrape_jobs("https://djinni.co/jobs/?primary_keyword=Data+Analyst",
                    ".job-list-item__title a", ".job-description__content", "https://djinni.co") +
        scrape_jobs("https://www.work.ua/jobs-data+analyst/",
                    "h2.add-top-sm a", "#job-description", "https://www.work.ua")
    )
    if not jobs:
        raise RuntimeError("Не вдалося зібрати вакансії. Перевірте інтернет-з'єднання.")
    print(f"\n  ✓ Зібрано: {len(jobs)}")
    json.dump(jobs, open(os.path.join(OUTPUT_DIR, "jobs.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    return jobs

# ════════════════════════════════════════════════════════════════════════════
#  СПІЛЬНЕ
# ════════════════════════════════════════════════════════════════════════════
EXTRA_STOP = {
    "experience", "skill", "skills", "knowledge", "ability", "use", "work", "working",
    "used", "using", "need", "required", "preferred", "strong", "excellent", "good",
    "year", "years", "level", "familiar", "understand", "understanding", "develop",
    "team", "company", "role", "join", "looking", "plus", "nice", "also", "well",
}

NLTK_STOP  = set(stopwords.words("english")) | EXTRA_STOP
SPACY_STOP = SPACY_STOP | EXTRA_STOP

def clean_text(text):
    text = re.sub(r"https?://\S+|\S+@\S+", " ", text)
    text = re.sub(r"[^a-zA-Zа-яА-ЯіІїЇєЄёЁ\s]", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()

# ════════════════════════════════════════════════════════════════════════════
#  ПІДХІД 1: NLTK
#  word_tokenize · sent_tokenize · MWETokenizer · WordNetLemmatizer · Porter · Snowball
# ════════════════════════════════════════════════════════════════════════════
def run_approach_1(jobs):
    t0  = time.time()
    raw = clean_text("\n\n".join(j["text"] for j in jobs))

    # 3 типи токенізації
    tok_word = word_tokenize(raw.lower())
    tok_sent = sent_tokenize(raw)
    mwe      = MWETokenizer([("machine","learning"),("power","bi"),("data","science"),
                              ("deep","learning"),("time","series"),("data","analyst")])
    tok_mwe  = mwe.tokenize(tok_word)

    # Стоп-слова
    clean = [t for t in tok_word if t.isalpha() and t.lower() not in NLTK_STOP and len(t) > 2]

    # Лематизація
    lemmatizer = WordNetLemmatizer()
    lems = [lemmatizer.lemmatize(w) for w in clean]

    # Стемінг
    port = [PorterStemmer().stem(w)   for w in clean]
    snow = [SnowballStemmer("english").stem(w) for w in clean]

    return {
        "elapsed": round(time.time() - t0, 4),
        "tok":  {"word": len(tok_word), "sent": len(tok_sent), "mwe": len(tok_mwe)},
        "n_stop": len(clean),
        "uniq": {"lemma": len(set(lems)), "porter": len(set(port)), "snowball": len(set(snow))},
        "top_lm": Counter(lems).most_common(12),
        "top_po": Counter(port).most_common(12),
        "top_sn": Counter(snow).most_common(12),
        "examples": [{"o":o,"lm":l,"po":p,"sn":s}
                     for o,l,p,s in zip(clean[:12], lems[:12], port[:12], snow[:12])],
    }

# ════════════════════════════════════════════════════════════════════════════
#  ПІДХІД 2: spaCy
#  Tokenizer · sentencizer · Lemmatizer · NER · TF-IDF · RAKE
# ════════════════════════════════════════════════════════════════════════════
def calculate_tfidf(docs):
    from collections import defaultdict
    import math
    N  = len(docs); df = defaultdict(int); tfs = []
    for doc in docs:
        freq = Counter(doc); total = max(len(doc), 1)
        tfs.append({w: c/total for w,c in freq.items()})
        for w in set(doc): df[w] += 1
    idf   = {w: math.log(N/(1+c)) for w,c in df.items()}
    total = defaultdict(float)
    for tf in tfs:
        for w, s in tf.items(): total[w] += s * idf.get(w, 0)
    return sorted(total.items(), key=lambda x: x[1], reverse=True)[:12]

def calculate_rake(doc_spacy, top_n=8):
    """RAKE через spaCy: фрази між стоп-словами."""
    phrases, current = [], []
    for token in doc_spacy:
        if token.is_stop or token.is_punct:
            if current: phrases.append(current); current = []
        elif token.is_alpha and len(token.text) > 2:
            current.append(token.lemma_.lower())
    if current: phrases.append(current)

    word_freq = Counter(w for ph in phrases for w in ph)
    scores = {" ".join(ph): sum(word_freq[w] for w in ph) / len(ph)
              for ph in phrases if 0 < len(ph) <= 4}
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_n]

def run_approach_2(jobs):
    t0  = time.time()
    raw = clean_text("\n\n".join(j["text"] for j in jobs))
    doc = nlp(raw[:nlp.max_length])

    # 3 типи токенізації
    tok_all  = [t.text for t in doc]
    tok_alpha= [t.text for t in doc if t.is_alpha]
    tok_sent = list(doc.sents)

    # Стоп-слова + лематизація через spaCy
    clean = [t for t in doc if t.is_alpha and t.lower_ not in SPACY_STOP and len(t.text) > 2]
    lems  = [t.lemma_.lower() for t in clean]

    # NER
    ents = [(e.text, e.label_) for e in doc.ents if e.label_ in {"ORG","PRODUCT","LANGUAGE"}]

    # TF-IDF по документах
    docs_words = [[t.lemma_.lower() for t in nlp(clean_text(j["text"]))
                   if t.is_alpha and t.lower_ not in SPACY_STOP and len(t.text) > 2]
                  for j in jobs]
    tf = calculate_tfidf(docs_words)

    return {
        "elapsed": round(time.time() - t0, 4),
        "tok":  {"all": len(tok_all), "alpha": len(tok_alpha), "sent": len(tok_sent)},
        "n_stop": len(clean),
        "uniq": {"lemma": len(set(lems))},
        "top_lm": Counter(lems).most_common(12),
        "tfidf":  tf,
        "rake":   calculate_rake(doc),
        "ner":    Counter(e for e,_ in ents).most_common(8),
    }

# ════════════════════════════════════════════════════════════════════════════
#  ГРАФІКИ
# ════════════════════════════════════════════════════════════════════════════
def style_axis(ax, title):
    ax.set_facecolor(COLOR_BG)
    ax.set_title(title, fontsize=10, fontweight="bold", pad=8)
    ax.yaxis.grid(True, color=COLOR_GRID, lw=0.6, ls="--")
    ax.set_axisbelow(True)
    [sp.set_edgecolor(COLOR_GRID) for sp in ax.spines.values()]

def generate_chart1(r, num_jobs):
    fig = plt.figure(figsize=(16, 11), facecolor="white")
    fig.suptitle(f"Підхід 1 — NLTK  |  {num_jobs} вакансій  |  {r['elapsed']}с",
                 fontsize=13, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38)

    # ① Типи токенізації
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(["word_tokenize", "sent_tokenize", "MWE"],
                  [r["tok"]["word"], r["tok"]["sent"], r["tok"]["mwe"]],
                  color=[COLOR_PRIMARY, COLOR_SECONDARY, COLOR_SUCCESS], alpha=0.87, width=0.5)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_ylabel("Кількість токенів")
    style_axis(ax, "① Типи токенізації (NLTK)")

    # ② Топ-10 лем
    ax = fig.add_subplot(gs[0, 1])
    w10 = [w for w,_ in r["top_lm"][:10]][::-1]
    c10 = [c for _,c in r["top_lm"][:10]][::-1]
    ax.barh(w10, c10, color=COLOR_PRIMARY, alpha=0.87)
    ax.bar_label(ax.containers[0], padding=2, fontsize=8)
    ax.set_xlabel("Частота")
    style_axis(ax, "② Топ-10 лем (WordNetLemmatizer)")

    # ③ Унікальні форми
    ax = fig.add_subplot(gs[0, 2])
    bars = ax.bar(["Леми", "Porter", "Snowball"],
                  [r["uniq"]["lemma"], r["uniq"]["porter"], r["uniq"]["snowball"]],
                  color=[COLOR_PRIMARY, COLOR_SECONDARY, COLOR_SUCCESS], alpha=0.87, width=0.45)
    ax.bar_label(bars, padding=3, fontsize=11)
    ax.set_ylabel("Унікальних форм")
    style_axis(ax, "③ Унікальні форми після нормалізації")

    # ④ Porter vs Snowball
    ax = fig.add_subplot(gs[1, :2])
    dp  = dict(r["top_po"]); ds = dict(r["top_sn"])
    sts = sorted(set(list(dp)+list(ds)), key=lambda w: dp.get(w,0)+ds.get(w,0), reverse=True)[:10]
    x, bw = np.arange(len(sts)), 0.38
    b1 = ax.bar(x-bw/2, [dp.get(s,0) for s in sts], bw, label="PorterStemmer",   color=COLOR_PRIMARY, alpha=0.87)
    b2 = ax.bar(x+bw/2, [ds.get(s,0) for s in sts], bw, label="SnowballStemmer", color=COLOR_SUCCESS, alpha=0.87)
    ax.bar_label(b1, padding=1, fontsize=7); ax.bar_label(b2, padding=1, fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(sts, rotation=28, ha="right", fontsize=9)
    ax.set_ylabel("Частота"); ax.legend(fontsize=9)
    style_axis(ax, "④ PorterStemmer vs SnowballStemmer")

    # ⑤ Таблиця трансформацій
    ax = fig.add_subplot(gs[1, 2]); ax.axis("off")
    rows = [["Оригінал","Лема","Porter","Snowball"]] + \
           [[e["o"], e["lm"], e["po"], e["sn"]] for e in r["examples"]]
    tbl = ax.table(cellText=rows[1:], colLabels=rows[0], loc="center", cellLoc="center")
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    tbl.auto_set_column_width([0,1,2,3])
    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(COLOR_GRID)
        cell.set_facecolor(COLOR_PRIMARY if row == 0 else ("#EEF2F8" if row%2==0 else "white"))
        if row == 0: cell.set_text_props(color="white", fontweight="bold")
    ax.set_title("⑤ Приклади трансформацій", fontsize=10, fontweight="bold", pad=6)

    plt.savefig(os.path.join(OUTPUT_DIR, "approach1.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ outputs/approach1.png")

def generate_chart2(r, num_jobs):
    fig = plt.figure(figsize=(16, 11), facecolor="white")
    fig.suptitle(f"Підхід 2 — spaCy  |  {num_jobs} вакансій  |  {r['elapsed']}с",
                 fontsize=13, fontweight="bold", y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.38)

    # ① Типи токенізації
    ax = fig.add_subplot(gs[0, 0])
    bars = ax.bar(["Всі токени", "Alpha", "Sentences"],
                  [r["tok"]["all"], r["tok"]["alpha"], r["tok"]["sent"]],
                  color=[COLOR_PRIMARY, COLOR_SECONDARY, COLOR_SUCCESS], alpha=0.87, width=0.5)
    ax.bar_label(bars, padding=3, fontsize=9)
    ax.set_ylabel("Кількість токенів")
    style_axis(ax, "① Типи токенізації (spaCy)")

    # ② Топ-10 лем
    ax = fig.add_subplot(gs[0, 1])
    w10 = [w for w,_ in r["top_lm"][:10]][::-1]
    c10 = [c for _,c in r["top_lm"][:10]][::-1]
    ax.barh(w10, c10, color=COLOR_SECONDARY, alpha=0.87)
    ax.bar_label(ax.containers[0], padding=2, fontsize=8)
    ax.set_xlabel("Частота")
    style_axis(ax, "② Топ-10 лем (spaCy Lemmatizer)")

    # ③ NER
    ax = fig.add_subplot(gs[0, 2])
    if r["ner"]:
        ne = [e for e,_ in r["ner"][:8]][::-1]
        nc = [c for _,c in r["ner"][:8]][::-1]
        ax.barh(ne, nc, color=COLOR_SUCCESS, alpha=0.87)
        ax.bar_label(ax.containers[0], padding=2, fontsize=8)
        ax.set_xlabel("Кількість згадок")
    style_axis(ax, "③ NER — організації та технології")

    # ④ TF-IDF
    ax = fig.add_subplot(gs[1, :2])
    tw = [w for w,_ in r["tfidf"][:12]][::-1]
    tv = [float(v) for _,v in r["tfidf"][:12]][::-1]
    bars = ax.barh(tw, tv, color=COLOR_SECONDARY, alpha=0.87)
    ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=8)
    ax.set_xlabel("TF-IDF Score")
    style_axis(ax, "④ TF-IDF топ-12")

    # ⑤ RAKE
    ax = fig.add_subplot(gs[1, 2])
    rl = [p for p,_ in r["rake"]][::-1]
    rs = [s for _,s in r["rake"]][::-1]
    bars = ax.barh(rl, rs, color=COLOR_SUCCESS, alpha=0.87)
    ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xlabel("RAKE Score")
    style_axis(ax, "⑤ RAKE ключові фрази")

    plt.savefig(os.path.join(OUTPUT_DIR, "approach2.png"), dpi=150, bbox_inches="tight")
    plt.close(); print("  ✓ outputs/approach2.png")

# ════════════════════════════════════════════════════════════════════════════
#  ЗАПУСК
# ════════════════════════════════════════════════════════════════════════════
def main():
    print("╔═══════════════════════════════════════════════════╗")
    print("║  ЛР №2 · Варіант 11/12 · Вимоги до Data Analyst  ║")
    print("╚═══════════════════════════════════════════════════╝")

    jobs = collect_all_jobs()

    print("\n── Аналіз ─────────────────────────────────────────────────────")
    r1 = run_approach_1(jobs)
    r2 = run_approach_2(jobs)
    print(f"  NLTK:  {r1['elapsed']}с | Топ-3: {[w for w,_ in r1['top_lm'][:3]]}")
    print(f"  spaCy: {r2['elapsed']}с | Топ-3 TF-IDF: {[w for w,_ in r2['tfidf'][:3]]}")

    print("\n── Графіки ─────────────────────────────────────────────────────")
    generate_chart1(r1, len(jobs))
    generate_chart2(r2, len(jobs))

    json.dump({"nltk": {k:v for k,v in r1.items() if k != "examples"}, "spacy": r2},
              open(os.path.join(OUTPUT_DIR, "results.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    print(f"\n  Готово! Файли у: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()