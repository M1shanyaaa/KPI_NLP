"""
МКР №13 — Порівняння LLM для тематичної кластеризації новин
Збирає RSS, класифікує 3 моделями Ollama, будує графіки, визначає переможця.

Встановлення:
  ollama pull llama3.2 && ollama pull mistral && ollama pull phi3
  pip install requests matplotlib numpy
Запуск: python mkr13.py
"""

import re, sys, time, json, threading
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime

import requests
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Конфіг ──────────────────────────────────────────────────────────
MODELS   = ["llama3.2", "mistral", "phi3"]
CATS     = ["Політика","Економіка","Безпека/Війна","Технології","Суспільство","Спорт","Інше"]
RSS      = {
    "BBC":        "https://feeds.bbci.co.uk/news/rss.xml",
    "AlJazeera":  "https://www.aljazeera.com/xml/rss/all.xml",
    "Укрінформ":  "https://www.ukrinform.ua/rss/block-lastnews",
    "Guardian":   "https://www.theguardian.com/world/rss",
}
MAX_NEWS = 30
TIMEOUT  = 90
OLLAMA   = "http://localhost:11434/api/chat"

# Підсилений промпт — менше «Інше», чіткіші правила
SYS_PROMPT = (
    "Ти — точний класифікатор новин. Визнач категорію заголовку.\n"
    "Категорії та їх зміст:\n"
    "  Політика      — вибори, уряди, дипломатія, лідери країн\n"
    "  Економіка     — ринки, торгівля, компанії, фінанси, ціни, нафта\n"
    "  Безпека/Війна — збройні конфлікти, теракти, катастрофи, злочини\n"
    "  Технології    — IT, AI, наука, космос, винаходи\n"
    "  Суспільство   — культура, освіта, спорт, знаменитості, здоров'я\n"
    "  Спорт         — футбол, олімпіада, змагання\n"
    "  Інше          — ЛИШЕ якщо жодна з вищих категорій не підходить\n\n"
    "Відповідай ЛИШЕ одним словом або фразою з наведеного списку. БЕЗ пояснень."
)

# ── RSS ────────────────────────────────────────────────────────────
def fetch_rss(name, url):
    try:
        r = requests.get(url, timeout=12, headers={"User-Agent": "NLP-Bot/1.0"})
        root = ET.fromstring(r.content)
    except Exception as e:
        print(f"  ! {name}: {e}"); return []
    items, per_feed = [], MAX_NEWS // len(RSS)
    for item in root.iter("item"):
        t = (item.findtext("title") or "").strip()
        d = re.sub(r"<[^>]+>", "", (item.findtext("description") or ""))[:200]
        if t: items.append({"title": t, "desc": d, "src": name})
        if len(items) >= per_feed: break
    print(f"  + {name}: {len(items)} новин")
    return items

def collect_news():
    print("\n[RSS] Збираю новини...")
    news = []
    for n, u in RSS.items(): news += fetch_rss(n, u)
    return news[:MAX_NEWS]

# ── LLM ─────────────────────────────────────
def normalize(raw):
    raw = raw.strip().lower()
    for c in CATS:
        if c.lower() in raw: return c
    kw = {
        "політ": "Політика", "election": "Політика", "gov": "Політика",
        "економ": "Економіка", "oil": "Економіка", "market": "Економіка", "price": "Економіка",
        "безпек": "Безпека/Війна", "війн": "Безпека/Війна", "war": "Безпека/Війна",
        "attack": "Безпека/Війна", "kill": "Безпека/Війна", "strike": "Безпека/Війна",
        "техно": "Технології", "tech": "Технології", "ai": "Технології",
        "суспіл": "Суспільство", "health": "Суспільство", "celeb": "Суспільство",
        "спорт": "Спорт", "sport": "Спорт", "football": "Спорт",
    }
    for k, v in kw.items():
        if k in raw: return v
    return "Інше"

def classify(model, title, desc):
    t0 = time.time()
    try:
        r = requests.post(OLLAMA, timeout=TIMEOUT, json={
            "model": model, "stream": False,
            "options": {"temperature": 0.0, "num_predict": 15},
            "messages": [
                {"role": "system", "content": SYS_PROMPT},
                {"role": "user",   "content": f"Новина: {title}. {desc}"}
            ]})
        cat = normalize(r.json()["message"]["content"])
    except requests.exceptions.ConnectionError: cat = "ERR_CONN"
    except requests.exceptions.Timeout:         cat = "ERR_TIMEOUT"
    except Exception:                            cat = "ERR"
    return cat, round(time.time() - t0, 2)

def run_all(news):
    results = {m: [] for m in MODELS}
    print(f"\n[LLM] Класифікую {len(news)} новин x {len(MODELS)} моделей...\n")
    for i, item in enumerate(news):
        print(f"  [{i+1:2d}/{len(news)}] {item['title'][:58]}...")
        row = {}

        def worker(m, it=item):
            c, t = classify(m, it["title"], it["desc"])
            row[m] = (c, t)
            ok = "+" if not c.startswith("ERR") else "-"
            print(f"    {ok} {m:<12} -> {c:<18} ({t}s)")

        ts = [threading.Thread(target=worker, args=(m,)) for m in MODELS]
        for t in ts: t.start()
        for t in ts: t.join(timeout=TIMEOUT + 5)

        for m in MODELS:
            c, t = row.get(m, ("ERR", 0))
            results[m].append({"cat": c, "time": t, "title": item["title"], "src": item["src"]})
        print()
    return results

# ── Метрики ────────────────────────────────────────────────────────────────
def metrics(results):
    m_data = {}
    for m in MODELS:
        valid = [r for r in results[m] if not r["cat"].startswith("ERR")]
        times = [r["time"] for r in valid if r["time"] > 0]
        dist  = Counter(r["cat"] for r in valid)
        probs = [c / len(valid) for c in dist.values()] if valid else [1]
        m_data[m] = {
            "speed":   round(np.mean(times), 2) if times else 0,
            "entropy": round(-sum(p * np.log2(p) for p in probs if p > 0), 2),
            "errors":  round((len(results[m]) - len(valid)) / len(results[m]) * 100, 1),
            "other_pct": round(dist.get("Інше", 0) / len(valid) * 100 if valid else 0, 1),
            "dist":    dist,
        }
    n = len(results[MODELS[0]])
    agr = {}
    for i, m1 in enumerate(MODELS):
        for m2 in MODELS[i + 1:]:
            pct = sum(results[m1][j]["cat"] == results[m2][j]["cat"] for j in range(n)) / n * 100
            agr[f"{m1}+{m2}"] = round(pct, 1)
    agr["Всі три"] = round(
        sum(len({results[m][j]["cat"] for m in MODELS}) == 1 for j in range(n)) / n * 100, 1)
    return m_data, agr

def best_model(m_data, agr):
    scores = defaultdict(int)
    scores[max(MODELS, key=lambda m: m_data[m]["entropy"])]   += 3
    scores[min(MODELS, key=lambda m: m_data[m]["other_pct"])] += 2
    scores[min(MODELS, key=lambda m: m_data[m]["speed"])]     += 2
    agr_sum = defaultdict(float)
    for k, v in agr.items():
        if "+" in k and "три" not in k:
            for m in MODELS:
                if m in k: agr_sum[m] += v
    scores[max(MODELS, key=lambda m: agr_sum[m])] += 3
    return max(MODELS, key=lambda m: scores[m]), dict(scores)

# ── Графіки ────────────────────────────────────────────────────────────────
CLR     = {"llama3.2": "#2196F3", "mistral": "#4CAF50", "phi3": "#FF9800"}
CAT_CLR = ["#E53935","#1E88E5","#43A047","#8E24AA","#FB8C00","#00ACC1","#757575"]

def plot(results, m_data, agr, news, winner, scores):
    fig = plt.figure(figsize=(18, 15))
    fig.patch.set_facecolor("#0D1117")
    gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.48, wspace=0.35)

    def ax_style(ax):
        ax.set_facecolor("#161B22")
        [s.set_color("#30363D") for s in ax.spines.values()]
        ax.tick_params(colors="#888", labelsize=8)

    tkw = dict(color="white", fontsize=11, fontweight="bold", pad=10)

    # 1. Grouped bar — розподіл категорій
    ax1 = fig.add_subplot(gs[0, :2]); ax_style(ax1)
    cats_used = [c for c in CATS if any(m_data[m]["dist"].get(c, 0) > 0 for m in MODELS)]
    x, w = np.arange(len(cats_used)), 0.26
    for i, m in enumerate(MODELS):
        vals = [m_data[m]["dist"].get(c, 0) for c in cats_used]
        bars = ax1.bar(x + i*w, vals, w, label=m, color=CLR[m], alpha=0.85, zorder=3)
        for b, v in zip(bars, vals):
            if v: ax1.text(b.get_x() + w/2, b.get_height() + .1, str(v),
                           ha="center", color="white", fontsize=7)
    ax1.set_xticks(x + w)
    ax1.set_xticklabels(cats_used, rotation=25, ha="right", color="#888", fontsize=8)
    ax1.set_title("Розподіл категорій по моделях", **tkw)
    ax1.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="white", fontsize=8)
    ax1.grid(axis="y", color="#30363D", lw=0.5, zorder=0)

    # 2. Метрики: ентропія / швидкість / % «Інше»
    ax2 = fig.add_subplot(gs[0, 2]); ax_style(ax2)
    labels = ["Ентропія", "Швидк.(с)", "% Інше"]
    data   = [[m_data[m]["entropy"], m_data[m]["speed"], m_data[m]["other_pct"]] for m in MODELS]
    x2, w2 = np.arange(3), 0.25
    for i, (m, row) in enumerate(zip(MODELS, data)):
        bars2 = ax2.bar(x2 + i*w2, row, w2, label=m, color=CLR[m], alpha=0.85, zorder=3)
        for b, v in zip(bars2, row):
            ax2.text(b.get_x() + w2/2, b.get_height() + 0.05, f"{v}",
                     ha="center", color="white", fontsize=7)
    ax2.set_xticks(x2 + w2)
    ax2.set_xticklabels(labels, color="#888", fontsize=9)
    ax2.set_title("Ключові метрики", **tkw)
    ax2.legend(facecolor="#161B22", edgecolor="#30363D", labelcolor="white", fontsize=8)
    ax2.grid(axis="y", color="#30363D", lw=0.5, zorder=0)

    # 3–5. Pie для кожної моделі
    for idx, m in enumerate(MODELS):
        ax = fig.add_subplot(gs[1, idx])
        ax.set_facecolor("#161B22")
        d = m_data[m]["dist"]
        if d:
            labs, sizes = zip(*d.items())
            cols = [CAT_CLR[CATS.index(l) % len(CAT_CLR)] if l in CATS else "#777" for l in labs]
            wedges, _, ats = ax.pie(
                sizes, autopct="%1.0f%%", colors=cols, startangle=90,
                wedgeprops=dict(edgecolor="#161B22", lw=1.5), pctdistance=0.78)
            [a.set(fontsize=7, color="white") for a in ats]
            ax.legend(wedges, labs, loc="lower center", bbox_to_anchor=(0.5, -0.28),
                      facecolor="#161B22", edgecolor="#30363D", labelcolor="white",
                      fontsize=6.5, ncol=2)
        is_winner = (m == winner)
        label = f"{m}  [WINNER]" if is_winner else m
        ax.set_title(label, color="#F59E0B" if is_winner else "white",
                     fontsize=11, fontweight="bold")

    # Узгодженість + скор у підписі
    agr_str = "  |  ".join(f"{k}: {v}%" for k, v in agr.items())
    score_str = "  |  ".join(f"{m}: {scores.get(m,0)} очок" for m in MODELS)
    fig.text(0.5, 0.025, agr_str, ha="center", color="#AAAAAA", fontsize=8.5)
    fig.text(0.5, 0.005, score_str, ha="center", color="#666", fontsize=8)

    fig.suptitle(
        f"Порівняння LLM для кластеризації новин  |  {len(news)} статей  |  "
        f"{datetime.now().strftime('%d.%m.%Y %H:%M')}",
        color="white", fontsize=13, fontweight="bold", y=0.99)

    plt.savefig("results.png", dpi=150, bbox_inches="tight", facecolor="#0D1117")
    plt.close()
    print("[OK] Збережено: results.png")

# ── Main ──────────────────────────────────────────
if __name__ == "__main__":
    # Перевірка Ollama
    try:
        avail = [m["name"].split(":")[0] for m in
                 requests.get("http://localhost:11434/api/tags", timeout=5)
                 .json().get("models", [])]
        miss = [m for m in MODELS if m not in avail]
        if miss:
            print(f"! Завантажте: {['ollama pull ' + m for m in miss]}"); sys.exit(1)
        print(f"[OK] Ollama: {avail}")
    except Exception:
        print("[ERR] Запустіть: ollama serve"); sys.exit(1)

    news = collect_news()
    if not news:
        print("[ERR] Немає новин"); sys.exit(1)

    results = run_all(news)
    m_data, agr = metrics(results)
    winner, scores = best_model(m_data, agr)

    plot(results, m_data, agr, news, winner, scores)

    # Підсумок
    print(f"\n{'='*60}")
    print(f"  {'Модель':<14} {'Ентропія':<11} {'Швидк.':<10} {'% Інше':<10} Очки")
    print(f"  {'-'*56}")
    for m in MODELS:
        mark = "  <- WINNER" if m == winner else ""
        print(f"  {m:<14} {m_data[m]['entropy']:<11} {m_data[m]['speed']:<10} "
              f"{m_data[m]['other_pct']:<10}{scores.get(m, 0)}{mark}")

    print(f"\n  Найкраща модель: {winner.upper()}")
    print(f"  Обгрунтування: найвища ентропія розподілу ({m_data[winner]['entropy']} bits) —")
    print(f"  збалансована класифікація по всіх категоріях.")
    print(f"{'='*60}\n")

    json.dump({
        "winner": winner, "scores": scores,
        "metrics": {m: {k: v for k, v in d.items() if k != "dist"} for m, d in m_data.items()},
        "agreements": agr,
        "classifications": {m: [{"title": r["title"], "src": r["src"], "cat": r["cat"]}
                                 for r in results[m]] for m in MODELS},
    }, open("results.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("[OK] Збережено: results.json")