import requests, re, os, warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from collections import Counter
from datetime import datetime

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

CSV_FILE = "monitoring_results.csv"

RSS = {
    "BBC":"https://feeds.bbci.co.uk/news/rss.xml",
    "AlJazeera":"https://www.aljazeera.com/xml/rss/all.xml",
    "NPR":"https://feeds.npr.org/1001/rss.xml",
    "NYTimes":"https://rss.nytimes.com/services/xml/rss/nyt/World.xml",
}

STOP = {
"the","and","to","of","in","is","for","on","with","at","by","from",
"news","that","this","are","was","has","have","been","its","not",
"but","than","as","an","a","it","be","or","will","about","more",
"after","their","also","he","she","they","his","her","said","says",
"new","can","had","all","one","what","were","which","do","up","out",
"so","if","over","into","who","could","no","just","how","when",
"us","get","would","we","our","you","your","report","reports"
}

# RSS отримання тексту
# ─────────────────────────────────────────────
def fetch(url):
    try:
        r = requests.get(url,timeout=10)
        soup = BeautifulSoup(r.content,"html.parser")
        text = " ".join(
            [t.get_text() for t in soup.find_all("title")] +
            [d.get_text() for d in soup.find_all("description")]
        )
        text = re.sub("<[^>]+>"," ",text)
        text = re.sub("&[a-z]+;"," ",text)
        return text
    except:
        return ""

# NLP фільтрація
# ─────────────────────────────────────────────
def top_words(text,n=5):
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b",text.lower())
    tokens = [t for t in tokens if t not in STOP]
    return Counter(tokens).most_common(n)

# час доби
# ─────────────────────────────────────────────
def period():
    h = datetime.now().hour
    if h < 12: return "Ранок"
    if h < 17: return "Обід"
    return "Вечір"

# CSV
# ─────────────────────────────────────────────
def save(rows):
    df = pd.DataFrame(rows)
    if os.path.exists(CSV_FILE):
        df.to_csv(CSV_FILE,mode="a",header=False,index=False,encoding="utf-8-sig")
    else:
        df.to_csv(CSV_FILE,index=False,encoding="utf-8-sig")


# BAR графік
# ─────────────────────────────────────────────
def bar(top5,label):
    w=[i[0] for i in top5][::-1]
    f=[i[1] for i in top5][::-1]

    plt.figure(figsize=(8,4))
    plt.barh(w,f)
    plt.title(label)
    plt.xlabel("Частота")
    plt.tight_layout()
    plt.savefig("top5_bar.png")


# Тренд
# ─────────────────────────────────────────────
def trend(df):
    blocks = df.drop_duplicates(["День","Час"])
    if len(blocks) < 2:
        return

    y = blocks["Сума частот"].values
    x = np.arange(len(y))
    a, b = np.polyfit(x, y, 1)
    xp = np.arange(len(y) + 7)
    yp = a * xp + b

    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(x, y, "o-", label="Факт")
    ax.plot(xp, yp, "--", label="Тренд")
    ax.ticklabel_format(style='plain', axis='y')
    ax.set_title("Динаміка частоти")
    ax.set_xlabel("Час")
    ax.set_ylabel("Сума частот")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("trend_forecast.png")


# WordCloud
# ─────────────────────────────────────────────
def cloud(df):
    try:
        from wordcloud import WordCloud
    except:
        return

    freq = df.groupby("Топ 5")["Частота"].sum().to_dict()
    wc = WordCloud(width=900,height=450,background_color="white")
    wc = wc.generate_from_frequencies(freq)
    plt.figure(figsize=(10,5))
    plt.imshow(wc)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("wordcloud.png")


# MAIN
# ─────────────────────────────────────────────
def main():
    now=datetime.now()
    day = now.strftime("%d.%m.%Y")
    clock = now.strftime("%H:%M")
    p = period()

    print(f"\nМоніторинг: {day} {clock} [{p}]\n")

    text=""
    for name,url in RSS.items():
        print("►",name)
        text += fetch(url)
    if not text:
        print("Дані не отримані")
        return

    top5 = top_words(text)
    print("\nТоп 5:")
    for w,c in top5:
        print(w,c)

    sum_freq=sum(c for _,c in top5)
    rows=[
        {"День":day,"Час":f"{p}: {clock}",
         "Топ 5":w,"Частота":c,
         "Сума частот":sum_freq,"Коментар":"OK"}
        for w,c in top5
    ]

    save(rows)
    df=pd.read_csv(CSV_FILE,encoding="utf-8-sig")
    bar(top5,f"{day} {clock}")
    trend(df)
    if len(df.drop_duplicates(["День","Час"]))>=3:
        cloud(df)

    print("\nГотово")

if __name__=="__main__":
    main()