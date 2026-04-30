"""
=============================================================================
КАВА-БОТ «АРОМА КАВА» — Голосовий помічник для людей зі слабким зором
=============================================================================
Лабораторна робота №7, Варіант 11
=============================================================================
"""

import sys
import json
import random
import pickle
import logging
import time
import re
from pathlib import Path

# ── Налаштування логування ──────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("coffee_bot.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("CoffeeBot")

# ── Залежності ─────────────────────────
try:
    import numpy as np
    import nltk
    from nltk.stem import SnowballStemmer
    from gtts import gTTS
    import speech_recognition as sr
    from tensorflow import keras
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from sklearn.preprocessing import LabelEncoder
    import pygame
except ImportError as exc:
    logger.error("Відсутній пакет: %s", exc)
    sys.exit(1)

# ── NLTK дані ─────────────────────────
for corpus in ("punkt", "stopwords", "punkt_tab"):
    try:
        nltk.download(corpus, quiet=True)
    except Exception:
        pass

# ── Константи ────────────────
INTENTS_FILE  = Path(__file__).parent / "intents_uk.json"
MODEL_FILE    = Path(__file__).parent / "nn_model.keras"
VECTORIZER_FILE = Path(__file__).parent / "vectorizer.pkl"
ENCODER_FILE  = Path(__file__).parent / "label_encoder.pkl"
AUDIO_CACHE   = Path(__file__).parent / "audio_cache"
AUDIO_CACHE.mkdir(exist_ok=True)

LANGUAGES = {"uk": "Ukrainian", "en": "English"}
GTTS_LANG_MAP = {"uk": "uk", "en": "en"}

RESPONSE_THRESHOLD = 0.55   # мінімальна впевненість моделі
UNKNOWN_RESP_UK = (
    "Вибачте, я не зрозумів ваш запит. "
    "Спитайте про меню, ціни, рекомендації або години роботи."
)
UNKNOWN_RESP_EN = (
    "Sorry, I did not understand your request. "
    "Ask about the menu, prices, recommendations or opening hours."
)


# ──────────────────────────────────────────────────────────────────────────
#  NLP helper
class NLPProcessor:
    _stemmer_cache: dict = {}
    @classmethod
    def _stemmer(cls, lang: str) -> SnowballStemmer:
        lang_map = {"uk": "russian", "en": "english"}  # uk → близька база
        key = lang_map.get(lang, "english")
        if key not in cls._stemmer_cache:
            cls._stemmer_cache[key] = SnowballStemmer(key)
        return cls._stemmer_cache[key]

    @classmethod
    def tokenize(cls, text: str, lang: str = "uk") -> list[str]:
        text = text.lower().strip()
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = text.split()
        stemmer = cls._stemmer(lang)
        return [stemmer.stem(t) for t in tokens if len(t) > 1]

    @classmethod
    def bag_of_words(cls, text: str, vocabulary: list[str], lang: str) -> np.ndarray:
        tokens = set(cls.tokenize(text, lang))
        return np.array([1.0 if w in tokens else 0.0 for w in vocabulary])


#  Neural-Network Intent Classifier
class IntentClassifier:

    def __init__(self, lang: str = "uk"):
        self.lang = lang
        self.model: Sequential | None = None
        self.vocabulary: list[str] = []
        self.label_encoder = LabelEncoder()
        self.intents: dict = {}

    # ── Підготовка даних ─────────────────────────────────────────
    def _prepare_data(self, intents_data: dict):
        X_raw, y_raw = [], []
        for intent in intents_data["intents"]:
            tag = intent["tag"]
            patterns_key = f"patterns_{self.lang}"
            for pattern in intent.get(patterns_key, []):
                tokens = NLPProcessor.tokenize(pattern, self.lang)
                X_raw.append(tokens)
                y_raw.append(tag)

        # Словник (відсортований для відтворюваності)
        vocab = sorted(set(t for tokens in X_raw for t in tokens))
        self.vocabulary = vocab

        X = np.array([
            np.array([1.0 if w in tokens else 0.0 for w in vocab])
            for tokens in X_raw
        ])
        y = self.label_encoder.fit_transform(y_raw)
        y_cat = keras.utils.to_categorical(y, num_classes=len(self.label_encoder.classes_))
        return X, y_cat

    # ── Побудова моделі ────────────────────────────────────────────────────
    def _build_model(self, input_dim: int, output_dim: int) -> Sequential:
        model = Sequential([
            Dense(256, activation="relu", input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(0.30),
            Dense(128, activation="relu"),
            BatchNormalization(),
            Dropout(0.25),
            Dense(64, activation="relu"),
            Dropout(0.20),
            Dense(output_dim, activation="softmax"),
        ])
        model.compile(
            optimizer=Adam(learning_rate=1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ── Навчання ─────────────────────────────────────────────────────
    def train(self, intents_data: dict, epochs: int = 300, verbose: int = 0):
        self.intents = intents_data
        X, y = self._prepare_data(intents_data)
        n_classes = y.shape[1]

        self.model = self._build_model(len(self.vocabulary), n_classes)

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="accuracy", patience=30, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="loss", factor=0.5, patience=15, min_lr=1e-5
            ),
        ]

        history = self.model.fit(
            X, y,
            epochs=epochs,
            batch_size=8,
            callbacks=callbacks,
            verbose=verbose,
        )

        final_acc = history.history["accuracy"][-1]
        logger.info(
            "[%s] Навчання завершено. Епох: %d, Точність: %.2f%%",
            self.lang.upper(), len(history.history["accuracy"]), final_acc * 100
        )
        return history

    # ── Збереження / завантаження ───────────────────────────────
    def save(self, model_path: Path, vec_path: Path, enc_path: Path):
        self.model.save(str(model_path))
        with open(vec_path, "wb") as f:
            pickle.dump(self.vocabulary, f)
        with open(enc_path, "wb") as f:
            pickle.dump(self.label_encoder, f)
        logger.info("Модель збережена: %s", model_path)

    def load(self, model_path: Path, vec_path: Path, enc_path: Path):
        self.model = load_model(str(model_path))
        with open(vec_path, "rb") as f:
            self.vocabulary = pickle.load(f)
        with open(enc_path, "rb") as f:
            self.label_encoder = pickle.load(f)
        logger.info("Модель завантажена: %s", model_path)

    # ── Передбачення ────────────────────────────────────────────────────
    def predict(self, text: str) -> tuple[str, float]:
        bow = NLPProcessor.bag_of_words(text, self.vocabulary, self.lang)
        bow = bow.reshape(1, -1)
        proba = self.model.predict(bow, verbose=0)[0]
        idx = int(np.argmax(proba))
        confidence = float(proba[idx])
        tag = self.label_encoder.inverse_transform([idx])[0]
        return tag, confidence


# ─────────────────────────────────────────────────────────────────────────────
#  Text-To-Speech
class VoiceSynthesizer:
    def __init__(self):
        pygame.mixer.init()

    def speak(self, text: str, lang: str = "uk", slow: bool = False):
        gtts_lang = GTTS_LANG_MAP.get(lang, "uk")
        cache_key = re.sub(r"\W+", "_", text[:40]) + f"_{lang}"
        audio_path = AUDIO_CACHE / f"{cache_key}.mp3"

        if not audio_path.exists():
            try:
                tts = gTTS(text=text, lang=gtts_lang, slow=slow)
                tts.save(str(audio_path))
            except Exception as e:
                logger.warning("TTS помилка: %s", e)
                print(f"[БОТ]: {text}")
                return

        try:
            pygame.mixer.music.load(str(audio_path))
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            logger.warning("Відтворення: %s", e)
            print(f"[БОТ]: {text}")


# ─────────────────────────────────────────────────────────────────────────────
#  Speech Recognizer
class VoiceRecognizer:
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.recognizer.pause_threshold = 0.8
        self.recognizer.energy_threshold = 300

    def listen(self, lang: str = "uk", timeout: int = 5) -> str | None:
        gtts_lang_map = {"uk": "uk-UA", "en": "en-US"}
        sr_lang = gtts_lang_map.get(lang, "uk-UA")
        try:
            with sr.Microphone() as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=0.5)
                logger.info("Слухаю... (мова: %s)", sr_lang)
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=8)
            text = self.recognizer.recognize_google(audio, language=sr_lang)
            logger.info("Розпізнано: '%s'", text)
            return text.lower().strip()
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            logger.error("SR API помилка: %s", e)
            return None


# ─────────────────────────────────────────────────────────────────────────────
#  Основний бот
class CoffeeBot:

    def __init__(self):
        self.lang: str = "uk"          # поточна мова
        self.active: bool = True

        # Завантаження intents
        with open(INTENTS_FILE, encoding="utf-8") as f:
            self.intents_data = json.load(f)

        # TTS / ASR
        self.tts = VoiceSynthesizer()
        self.asr = VoiceRecognizer()

        # Класифікатори для обох мов
        self.classifiers: dict[str, IntentClassifier] = {}
        self._load_or_train_models()

    # ── Моделі ─────────────────────────────────────────────────────────────
    def _load_or_train_models(self):
        for lang in ("uk", "en"):
            clf = IntentClassifier(lang=lang)
            mp  = Path(str(MODEL_FILE).replace(".keras", f"_{lang}.keras"))
            vp  = Path(str(VECTORIZER_FILE).replace(".pkl", f"_{lang}.pkl"))
            ep  = Path(str(ENCODER_FILE).replace(".pkl",   f"_{lang}.pkl"))

            if mp.exists() and vp.exists() and ep.exists():
                clf.load(mp, vp, ep)
            else:
                logger.info("Треную модель для мови: %s", lang.upper())
                clf.train(self.intents_data, epochs=400, verbose=0)
                clf.save(mp, vp, ep)

            clf.intents = self.intents_data
            self.classifiers[lang] = clf

    # ── Отримання відповіді ────────────────────────────────────────────────
    def get_response(self, text: str) -> str:
        clf = self.classifiers[self.lang]
        tag, confidence = clf.predict(text)

        logger.info("Інтент: '%s' (впевненість: %.2f%%)", tag, confidence * 100)

        if confidence < RESPONSE_THRESHOLD:
            return UNKNOWN_RESP_UK if self.lang == "uk" else UNKNOWN_RESP_EN

        # Перемикання мови
        if tag == "language":
            text_lower = text.lower()
            if "english" in text_lower or "англійськ" in text_lower or "англ" in text_lower:
                self.lang = "en"
                return random.choice(
                    self._get_responses_for_tag("language", "en")
                )
            elif "укра" in text_lower:
                self.lang = "uk"
                return random.choice(
                    self._get_responses_for_tag("language", "uk")
                )

        # Прощання
        if tag == "goodbye":
            self.active = False

        resp_key = f"responses_{self.lang}"
        for intent in self.intents_data["intents"]:
            if intent["tag"] == tag:
                responses = intent.get(resp_key, [])
                if responses:
                    return random.choice(responses)

        return UNKNOWN_RESP_UK if self.lang == "uk" else UNKNOWN_RESP_EN

    def _get_responses_for_tag(self, tag: str, lang: str) -> list[str]:
        key = f"responses_{lang}"
        for intent in self.intents_data["intents"]:
            if intent["tag"] == tag:
                return intent.get(key, [])
        return []

    # ── Привітання ───────────────────────────────────────────────────────────
    def _greet(self):
        greeting = (
            "Вітаємо у голосовому боті кав'ярні «Арома Кава»! "
            "Скажіть 'українська' або 'English' для вибору мови. "
            "Для виходу скажіть 'до побачення'."
        )
        print("\n" + "="*60)
        print("  КАВА-БОТ «АРОМА КАВА» — Голосовий Асистент")
        print("="*60)
        print(f"[БОТ]: {greeting}\n")
        self.tts.speak(greeting, lang="uk")

    # ── Текстовий режим (fallback коли немає мікрофона) ─────────────────────
    def run_text_mode(self):
        self._greet()
        prompt = "Ви (UK) > " if self.lang == "uk" else "You (EN) > "
        while self.active:
            try:
                prompt = f"Ви [{self.lang.upper()}] > "
                user_input = input(prompt).strip()
                if not user_input:
                    continue
                response = self.get_response(user_input)
                print(f"[БОТ]: {response}\n")
                self.tts.speak(response, lang=self.lang)
            except (KeyboardInterrupt, EOFError):
                bye = "До побачення!" if self.lang == "uk" else "Goodbye!"
                print(f"\n[БОТ]: {bye}")
                self.tts.speak(bye, lang=self.lang)
                break
        print("\n[Сесію завершено]")

    # ── Голосовий режим ──────────────────────────
    def run_voice_mode(self):
        """Повний голосовий режим."""
        self._greet()
        prompt_uk = "Слухаю вас... Говоріть!"
        prompt_en = "Listening... Please speak!"

        while self.active:
            # Підказка перед прослуховуванням
            hint = prompt_uk if self.lang == "uk" else prompt_en
            print(f"\n[{hint}]")

            text = self.asr.listen(lang=self.lang)

            if text is None:
                retry = (
                    "Вибачте, не почув. Повторіть, будь ласка."
                    if self.lang == "uk"
                    else "Sorry, I did not hear. Please repeat."
                )
                print(f"[БОТ]: {retry}")
                self.tts.speak(retry, lang=self.lang)
                continue

            print(f"Ви: {text}")
            response = self.get_response(text)
            print(f"[БОТ]: {response}\n")
            self.tts.speak(response, lang=self.lang)

        print("\n[Сесію завершено]")

    # ── Демо-режим (без апаратури) ───────────────────────────────────
    def run_demo(self):
        demo_queries = [
            ("uk", "привіт"),
            ("uk", "що у вас є в меню"),
            ("uk", "розкажіть про гарячу каву"),
            ("uk", "що порадите"),
            ("uk", "скільки коштує капучино"),
            ("uk", "у вас є безлактозне молоко"),
            ("uk", "до якого часу ви відкриті"),
            ("en", "english"),
            ("en", "what desserts do you have"),
            ("en", "what is your recommendation"),
            ("uk", "українська"),
            ("uk", "дякую"),
            ("uk", "до побачення"),
        ]

        self._greet()
        time.sleep(1)

        print("\n" + "─"*60)
        print("  ДЕМО-РЕЖИМ: автоматичні тестові запити")
        print("─"*60 + "\n")

        for lang_hint, query in demo_queries:
            if not self.active:
                break
            # Виставляємо мову якщо потрібно для правильної класифікації
            if lang_hint != self.lang and lang_hint in ("uk", "en"):
                # Все одно перевіримо через модель
                pass
            print(f"Ви [{self.lang.upper()}]: {query}")
            response = self.get_response(query)
            print(f"[БОТ]: {response}\n")
            self.tts.speak(response, lang=self.lang)
            time.sleep(0.5)

        print("\n[Демо завершено]")


# ─────────────────────────────────────────────────────────────────────────────
#  Точка входу
def main():
    import argparse
    parser = argparse.ArgumentParser(description="CoffeeBot — Голосовий асистент кав'ярні")
    parser.add_argument(
        "--mode",
        choices=["voice", "text", "demo"],
        default="text",
        help="Режим роботи: voice | text | demo (default: text)",
    )
    args = parser.parse_args()

    bot = CoffeeBot()

    if args.mode == "voice":
        bot.run_voice_mode()
    elif args.mode == "demo":
        bot.run_demo()
    else:
        bot.run_text_mode()


if __name__ == "__main__":
    main()