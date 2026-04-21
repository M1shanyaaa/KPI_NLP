import re
import requests
from bs4 import BeautifulSoup
import spacy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from deep_translator import GoogleTranslator
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import warnings

warnings.filterwarnings(action='ignore')

# Завантаження моделей spaCy
try:
    nlp_uk = spacy.load("uk_core_news_sm")
    nlp_en = spacy.load("en_core_web_sm")
except OSError:
    print("Помилка: Моделі spaCy не знайдені. Виконайте команди:")
    print("python -m spacy download uk_core_news_sm")
    print("python -m spacy download en_core_web_sm")
    exit()


# =====================================================================
# І РІВЕНЬ: WEB-СКРАПІНГ ТА АГРЕГАЦІЯ (IN-MEMORY)
# =====================================================================

def scrape_category(url, platform_name, category_name):
    """
    Базовий web-скрапер. Повертає зібраний текст замість збереження у файл.
    """
    headers = {
        'User-Agent': 'Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:109.0) Gecko/20100101 Firefox/116.0'
    }

    scraped_text = ""
    label = f"{platform_name}_{category_name}"

    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'lxml')

            items = soup.find_all(['h3', 'a', 'div'], class_=re.compile(r'title|name|product', re.I))

            for item in items[:15]:
                text = item.get_text(strip=True)
                if len(text) > 10:
                    scraped_text += text + ".\n"

            if scraped_text:
                print(f"Дані успішно зібрано з {url}")
                return {"label": label, "text": scraped_text}

    except Exception as e:
        print(f"Помилка доступу до {url}: {e}")

    # ФОЛБЕК (Mock Data)
    print(f"Згенерувано тестові дані для {platform_name} -> {category_name}")
    mock_data = generate_mock_data(category_name, platform_name)

    return {"label": label, "text": mock_data}


def generate_mock_data(category, platform):
    data = {
        "computers": f"Ігровий ноутбук ASUS ROG Strix на платформі {platform}. Офісний комп'ютер Lenovo ThinkPad. Ноутбук Apple MacBook Air M2.",
        "televisions": f"Телевізор Samsung 4K Smart TV ({platform}). OLED телевізор LG 55 дюймів. Бюджетний телевізор Kivi з Android TV.",
        "refrigerators": f"Двокамерний холодильник Bosch з системою No Frost від {platform}. Холодильник Beko. Вбудований холодильник Samsung."
    }
    return data.get(category, f"Товари категорії {category} на {platform}.")


# =====================================================================
# І РІВЕНЬ: ПОРІВНЯЛЬНИЙ АНАЛІЗ (NLP Similarity)
# =====================================================================

def text_filter_ukr(text):
    """
    Фільтрація та лематизація україномовного тексту.
    """
    text = text.replace("\n", " ").lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[0-9]+', '', text)

    doc = nlp_uk(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and token.text.strip()]
    return " ".join(lemmas)


def comparative_analysis(data_list):
    """
    Аналіз подібності пропозицій, приймає список словників із даними.
    """
    print("\n--- Проведення порівняльного аналізу (Косинусна подібність) ---")
    documents = []
    labels = []

    for item in data_list:
        processed_text = text_filter_ukr(item["text"])
        documents.append(processed_text)
        labels.append(item["label"])

    if not documents:
        print("Немає даних для аналізу.")
        return

    vectorizer = CountVectorizer()
    sparse_matrix = vectorizer.fit_transform(documents)
    doc_term_matrix = sparse_matrix.todense()

    df = pd.DataFrame(
        doc_term_matrix,
        columns=vectorizer.get_feature_names_out(),
        index=labels,
    )

    similarity_matrix = cosine_similarity(df, df)
    print("Матриця косинусної подібності:\n")
    print(pd.DataFrame(similarity_matrix, index=labels, columns=labels))

    # Візуалізація
    plt.figure(figsize=(8, 6))
    sns.heatmap(similarity_matrix, annot=True, cmap="YlGnBu", xticklabels=labels, yticklabels=labels)
    plt.title("Косинусна подібність пропозицій між платформами")
    plt.tight_layout()
    plt.show()


# =====================================================================
# ІІ РІВЕНЬ: АНАЛІЗ ТОНАЛЬНОСТІ ВІДГУКІВ
# =====================================================================

def preprocess_text_en(text):
    doc = nlp_en(text)
    filtered_tokens = [token.lemma_ for token in doc if not token.is_stop and token.text.isalpha()]
    return " ".join(filtered_tokens)


def translate_to_en(text_uk):
    return GoogleTranslator(source='uk', target='en').translate(text_uk)


def analyze_sentiment(reviews):
    print("\n--- Аналіз тональності відгуків про товар ---")
    analyzer = SentimentIntensityAnalyzer()

    results = []
    for review in reviews:
        en_review = translate_to_en(review)
        clean_en_review = preprocess_text_en(en_review)

        blob = TextBlob(clean_en_review)
        blob_score = blob.sentiment.polarity

        vader_scores = analyzer.polarity_scores(clean_en_review)
        vader_compound = vader_scores['compound']

        if vader_compound >= 0.05:
            sentiment = "Позитивний 🟢"
        elif vader_compound <= -0.05:
            sentiment = "Негативний 🔴"
        else:
            sentiment = "Нейтральний ⚪"

        results.append({
            "Оригінал": review,
            "VADER Score": round(vader_compound, 2),
            "TextBlob Score": round(blob_score, 2),
            "Тональність": sentiment
        })

    df_results = pd.DataFrame(results)
    print(df_results.to_string(index=False))


# =====================================================================
# ГОЛОВНИЙ БЛОК ВИКОНАННЯ
# =====================================================================

if __name__ == '__main__':
    platforms = {
        "Rozetka": {
            "computers": "https://rozetka.com.ua/ua/computers-notebooks/c80253/",
            "televisions": "https://rozetka.com.ua/ua/teleapparatura/c46060/",
            "refrigerators": "https://rozetka.com.ua/ua/refrigerators/c80125/"
        },
        "Comfy": {
            "computers": "https://comfy.ua/ua/computer/",
            "televisions": "https://comfy.ua/ua/flat-tvs/",
            "refrigerators": "https://comfy.ua/ua/refrigerators/"
        },
        "Foxtrot": {
            "computers": "https://www.foxtrot.com.ua/uk/shop/noutbuki.html",
            "televisions": "https://www.foxtrot.com.ua/uk/shop/led_televizory.html",
            "refrigerators": "https://www.foxtrot.com.ua/uk/shop/holodilniki.html"
        }
    }

    # Тепер збираємо словники з даними, а не імена файлів
    scraped_data = []

    print("Починаємо збір даних (І рівень)...")
    for platform_name, categories_dict in platforms.items():
        for category, url in categories_dict.items():
            # Отримуємо словник {"label": "...", "text": "..."}
            data = scrape_category(url, platform_name, category)
            if data:
                scraped_data.append(data)

    # Передаємо список словників для аналізу
    comparative_analysis(scraped_data)

    sample_reviews = [
        "Цей ноутбук просто супер! Працює дуже швидко, екран яскравий.",
        "Жахливий телевізор. Зламався через два тижні, звук як із відра.",
        "Нормальний холодильник за свої гроші. Морозить добре, але трохи шумить.",
        "Відмінний сервіс Розетки, доставили товар на наступний день. Задоволений.",
        "Не рекомендую купувати цю модель, батарея тримає дуже погано."
    ]

    analyze_sentiment(sample_reviews)