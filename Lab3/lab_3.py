"""
Лабораторна робота №3 — Векторизація та Аналіз (NLP)
Варіант 11: Чат-бот (англійська мова), тегування, векторизація, обробка відгуків та FAQ.
"""

import spacy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings('ignore')

# Завантажуємо spaCy модель з векторним уявленням слів
try:
    nlp = spacy.load("en_core_web_md")
except OSError:
    print("Помилка: завантажте модель -> python -m spacy download en_core_web_md")
    exit()

# ЧАСТИНА 1: БАЗА ДАНИХ ЧАТ-БОТА (FAQ - Топ 10 запитань)
# =====================================================================
FAQ_DATABASE = {
    "What are your working hours?": "We are open 24/7 for online support.",
    "How can I track my order?": "You can track your order in your personal account under 'My Orders'.",
    "Do you offer international shipping?": "Yes, we ship worldwide. Shipping costs apply.",
    "What is your return policy?": "You can return the product within 30 days of receipt.",
    "How long does delivery take?": "Standard delivery takes 3 to 5 business days.",
    "What payment methods do you accept?": "We accept Visa, MasterCard, PayPal, and Apple Pay.",
    "How can I contact a human agent?": "You can email us at support@company.com or call 555-0192.",
    "Is my personal data secure?": "Yes, we use industry-standard encryption to protect your data.",
    "How do I reset my password?": "Click 'Forgot Password' on the login page to receive a reset link.",
    "Are there any discounts for new customers?": "Yes, use code WELCOME10 for 10% off your first order."
}

FAQ_QUESTIONS = list(FAQ_DATABASE.keys())

# ЧАСТИНА 2: КЛАС ЧАТ-БОТА (Логіка NLP)
# =====================================================================
class SmartNLPChatbot:
    def __init__(self):
        self.manager_inbox = {"questions": [], "positive_reviews": [], "negative_reviews": []}

        # Векторизація FAQ (Метод 1: TF-IDF)
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.faq_tfidf_matrix = self.tfidf.fit_transform(FAQ_QUESTIONS)

        # Сід-вектори для векторного аналізу тональності (Sentiment by Embeddings)
        self.pos_vector = nlp("excellent perfect great amazing good love satisfied").vector
        self.neg_vector = nlp("terrible bad awful worst disappointed poor slow hate").vector

    def is_question(self, doc):
        """POS-Tagging: Визначає, чи є текст запитанням"""
        if doc.text.strip().endswith('?'):
            return True
        if len(doc) > 0:
            first_tag = doc[0].tag_
            if first_tag in ['WDT', 'WP', 'WP$', 'WRB', 'VBD', 'VBZ', 'VBP', 'MD']:
                return True
        return False

    def analyze_sentiment(self, doc):
        """Векторний аналіз тональності на основі прикметників (POS-Tagging + Word2Vec)"""
        # Витягуємо лише прикметники та прислівники
        modifiers = [token.text for token in doc if token.pos_ in ['ADJ', 'ADV']]

        # Якщо є модифікатори - беремо їх вектор, якщо ні - вектор всього речення
        doc_vec = nlp(" ".join(modifiers)).vector if modifiers else doc.vector

        sim_pos = cosine_similarity([doc_vec], [self.pos_vector])[0][0]
        sim_neg = cosine_similarity([doc_vec], [self.neg_vector])[0][0]

        # Нейтральний лише якщо обидві схожості низькі (бракує яскравих емоційних слів)
        if sim_pos < 0.7 and sim_neg < 0.7:
            return "neutral"

        return "positive" if sim_pos > sim_neg else "negative"

    def match_faq(self, text, threshold=0.55):
        """Векторизація: Пошук у FAQ через TF-IDF"""
        user_vec = self.tfidf.transform([text])
        similarities = cosine_similarity(user_vec, self.faq_tfidf_matrix)[0]
        best_match_idx = np.argmax(similarities)

        if similarities[best_match_idx] >= threshold:
            matched_q = FAQ_QUESTIONS[best_match_idx]
            return FAQ_DATABASE[matched_q]
        return None

    def process_message(self, text):
        """Головний конвеєр обробки повідомлення"""
        print(f"\n[USER]: {text}")
        doc = nlp(text)

        if self.is_question(doc):
            answer = self.match_faq(text)
            if answer:
                print(f"[BOT FAQ]: {answer}")
            else:
                self.manager_inbox["questions"].append(text)
                print("[BOT]: I'm not sure about that. I have forwarded your question to our human managers. They will reply shortly.")
        else:
            sentiment = self.analyze_sentiment(doc)
            if sentiment == "negative":
                self.manager_inbox["negative_reviews"].append(text)
                print("[BOT URGENT]: We are very sorry you had a bad experience! A manager has been immediately notified to resolve this issue for you.")
            elif sentiment == "neutral":
                # Додано: обробка нейтральних відгуків
                self.manager_inbox["positive_reviews"].append(text)
                print("[BOT]: Thank you for your feedback! It has been recorded.")
            else:
                self.manager_inbox["positive_reviews"].append(text)
                print("[BOT]: Thank you for your wonderful feedback! We are glad you are satisfied.")

    def print_manager_report(self):
        """Звіт для менеджерів (Збережені запити)"""
        print("\n" + "="*50)
        print("MANAGER INBOX REPORT")
        print("="*50)
        print(f"Negative Reviews (Needs Immediate Action): {len(self.manager_inbox['negative_reviews'])}")
        for i, msg in enumerate(self.manager_inbox['negative_reviews'], 1): print(f"  {i}. {msg}")

        print(f"\nPositive / Neutral Reviews (For Analytics): {len(self.manager_inbox['positive_reviews'])}")
        for i, msg in enumerate(self.manager_inbox['positive_reviews'], 1): print(f"  {i}. {msg}")

        print(f"\n❓ Unanswered Questions (Please Reply): {len(self.manager_inbox['questions'])}")
        for i, msg in enumerate(self.manager_inbox['questions'], 1): print(f"  {i}. {msg}")
        print("="*50 + "\n")

# ЧАСТИНА 3: ДОСЛІДЖЕННЯ ВЕКТОРІВ ТА POS-ТЕГІВ
# =====================================================================
def plot_pos_tags(text, title):
    """Додано: Частотний аналіз POS-тегів"""
    plt.figure(figsize=(8, 4))
    doc = nlp(text)
    tags = [t.pos_ for t in doc if not t.is_punct and not t.is_space]
    pd.Series(tags).value_counts().plot(kind='bar', title=f"Frequency of POS Tags: {title}", color="#4C72B0")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(f"pos_{title.replace(' ', '_')}.png")
    plt.close()
    print(f"Графік POS-тегів збережено у файл 'pos_{title.replace(' ', '_')}.png'.")

def run_vector_research():
    print("\n--- ЗАПУСК ДОСЛІДЖЕННЯ ВЕКТОРНИХ ПРЕДСТАВЛЕНЬ ТА POS ---")

    text = "Machine learning algorithms build a model based on sample data, known as training data, in order to make predictions."
    segment1 = "Machine learning algorithms build a model."
    segment2 = "Sample data is used to make predictions."
    phrase = "training data"

    # 1. Створюємо графік частотності POS-тегів для основного тексту
    plot_pos_tags(text, "EN_POS")

    # 2. Створюємо корпус для TF-IDF
    corpus = [text, segment1, segment2, phrase]

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(corpus).toarray()
    cos_sim_tfidf = cosine_similarity(tfidf_matrix)

    spacy_docs = [nlp(t) for t in corpus]
    cos_sim_spacy = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            cos_sim_spacy[i][j] = spacy_docs[i].similarity(spacy_docs[j])

    labels = ['Full Text', 'Segment 1', 'Segment 2', 'Phrase']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(cos_sim_tfidf, annot=True, xticklabels=labels, yticklabels=labels, cmap="Blues", ax=axes[0])
    axes[0].set_title("Векторизація TF-IDF (Косинусна схожість)")

    sns.heatmap(cos_sim_spacy, annot=True, xticklabels=labels, yticklabels=labels, cmap="Greens", ax=axes[1])
    axes[1].set_title("Векторизація spaCy Embeddings (Косинусна схожість)")

    plt.tight_layout()
    plt.savefig("vector_comparison.png")
    plt.close()
    print("Результати дослідження збережено у файл 'vector_comparison.png'.")

# ГОЛОВНА ТОЧКА ВХОДУ
# =====================================================================
if __name__ == "__main__":
    run_vector_research()

    bot = SmartNLPChatbot()

    test_messages = [
        "How long does delivery take?",
        "The application is terrible and very slow, I hate it!",
        "What is the capital of France?",
        "I love this service, the support team is excellent and fast.",
        "The application is okay, nothing special.",
        "Are there any discounts for new customers?",
        "Do you repair broken screens?"
    ]

    print("\n--- СИМУЛЯЦІЯ ЧАТ-БОТА ---")
    for msg in test_messages:
        bot.process_message(msg)

    bot.print_manager_report()

