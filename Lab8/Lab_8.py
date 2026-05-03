"""
Лабораторна робота №8, Варіант 11/12
Бот-помічник для людей із слабким зором — кав'ярня «Арома Кава»
Три локальні LLM через Ollama, голосовий I/O.

Запуск:
  python coffee_bot.py # голосовий режим
  python coffee_bot.py --text # текстовий режим (без мікрофона)
"""

import os
import ctypes
import subprocess
import threading
import tempfile
import argparse
import random
import re
import requests

_ALSA_CALLBACK_TYPE = ctypes.CFUNCTYPE(None, ctypes.c_char_p, ctypes.c_int,
                                       ctypes.c_char_p, ctypes.c_int,
                                       ctypes.c_char_p)


def _null_alsa_handler(*args):
    pass


_ALSA_HANDLER_REF = _ALSA_CALLBACK_TYPE(_null_alsa_handler)

try:
    _libasound = ctypes.cdll.LoadLibrary("libasound.so.2")
    _libasound.snd_lib_error_set_handler(_ALSA_HANDLER_REF)
except Exception:
    pass

_stderr_bak = os.dup(2)
_devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(_devnull, 2)

try:
    import speech_recognition as sr

    SR_OK = True
except ImportError:
    SR_OK = False

os.dup2(_stderr_bak, 2)
os.close(_stderr_bak)
os.close(_devnull)

try:
    from gtts import gTTS

    GTTS_OK = True
except ImportError:
    GTTS_OK = False


# ══════════════════════════════════════════════
#  ВІДТВОРЕННЯ АУДІО (subprocess, без pygame)

def _find_player() -> list[str] | None:
    candidates = [
        ["mpg123", "-q"],
        ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"],
        ["mplayer", "-really-quiet"],
        ["cvlc", "--play-and-exit", "--quiet"],
    ]
    for cmd in candidates:
        try:
            subprocess.run([cmd[0], "--version"],
                           stdout=subprocess.DEVNULL,
                           stderr=subprocess.DEVNULL,
                           check=False)
            return cmd
        except FileNotFoundError:
            continue
    return None


_PLAYER = _find_player()


def play_mp3(path: str):
    if _PLAYER is None:
        return
    subprocess.run(
        _PLAYER + [path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ══════════════════════════════════════════════
#  МЕНЮ

MENU = {
    "кава": {
        "Еспресо": 55, "Ристретто": 55, "Американо": 65,
        "Капучино": 85, "Латте": 95, "Флет уайт": 90,
        "Раф кава": 100, "Моккачино": 95, "Айс-латте": 105,
        "Матча-латте": 110, "В'єтнамська кава": 90,
    },
    "чай та напої": {
        "Зелений чай": 60, "Чорний чай": 55, "Трав'яний чай": 65,
        "Гарячий шоколад": 90, "Лимонад": 80, "Фреш апельсиновий": 110,
    },
    "їжа": {
        "Круасан з маслом": 55, "Круасан шинка&сир": 95,
        "Сендвіч з авокадо": 145, "Чізкейк": 110,
        "Тірамісу": 120, "Вафлі з ягодами": 130,
        "Банановий хліб": 85, "Брауні": 75,
    },
}
ALT_MILK_EXTRA = 20


def build_menu_text() -> str:
    icons = {"кава": "☕", "чай та напої": "🍵", "їжа": "🥐"}
    lines = ["КАВ'ЯРНЯ «АРОМА КАВА» — Київ, вул. Хрещатик, 15\n"]
    for cat, items in MENU.items():
        lines.append(f"{icons[cat]} {cat.upper()}:")
        for name, price in items.items():
            lines.append(f"  {name} — {price} грн")
        lines.append("")
    lines.append(f"Альтернативне молоко (+{ALT_MILK_EXTRA} грн): вівсяне, мигдальне, кокосове, соєве")
    lines.append("ℹПовідомте бариста про алергії.")
    return "\n".join(lines)


MENU_TEXT = build_menu_text()

SYSTEM_PROMPT = (
        "Ти — голосовий помічник кав'ярні «Арома Кава» для людей із слабким зором.\n"
        "Допомагай обирати страви та напої ЛИШЕ з меню нижче.\n"
        "Відповідай коротко (2–4 речення), чітко, завжди називай ціну.\n"
        "Відповідай ТІЛЬКИ українською мовою.\n\n"
        + MENU_TEXT
)


# ══════════════════════════════════════════════
#  ШВИДКІ КОМАНДИ (без LLM)

def cmd_full_menu() -> str:
    parts = []
    for cat, items in MENU.items():
        names = ", ".join(f"{n} — {p} грн" for n, p in items.items())
        parts.append(f"{cat}: {names}")
    return ". ".join(parts) + f". Альтернативне молоко плюс {ALT_MILK_EXTRA} гривень."


def cmd_cheapest() -> str:
    all_items = [(n, p) for cat in MENU.values() for n, p in cat.items()]
    top = sorted(all_items, key=lambda x: x[1])[:3]
    return "Найдешевші позиції: " + ", ".join(f"{n} за {p} грн" for n, p in top) + "."


def cmd_most_expensive() -> str:
    all_items = [(n, p) for cat in MENU.values() for n, p in cat.items()]
    top = sorted(all_items, key=lambda x: x[1], reverse=True)[:3]
    return "Найдорожчі позиції: " + ", ".join(f"{n} за {p} грн" for n, p in top) + "."


def cmd_random_rec() -> str:
    cat = random.choice(list(MENU.keys()))
    name, price = random.choice(list(MENU[cat].items()))
    return f"Сьогодні рекомендую {name} з розділу «{cat}» — {price} гривень. Смачного!"


def cmd_budget(limit: int) -> str:
    fits = sorted(
        [(n, p) for cat in MENU.values() for n, p in cat.items() if p <= limit],
        key=lambda x: x[1],
    )
    if not fits:
        return f"На жаль, немає позицій до {limit} гривень."
    return f"До {limit} гривень: " + ", ".join(f"{n} — {p} грн" for n, p in fits) + "."


def cmd_category(cat_key: str) -> str:
    items = MENU.get(cat_key, {})
    return f"У розділі «{cat_key}»: " + ", ".join(f"{n} за {p} грн" for n, p in items.items()) + "."


def cmd_help() -> str:
    return (
        "Доступні команди: «меню» — зачитати все меню; "
        "«порекомендуй» — випадкова рекомендація; "
        "«найдешевше» або «найдорожче»; "
        "«до 100 гривень» — вибір по бюджету; "
        "або просто запитайте про будь-який напій чи страву. "
        "«дякую» — завершити."
    )


QUICK_COMMANDS = [
    ({"меню", "покажи меню", "що є", "що у вас є", "весь список", "все меню"}, cmd_full_menu),
    ({"найдешевше", "що дешевше", "економ", "дешеві варіанти", "найдешевші"}, cmd_cheapest),
    ({"найдорожче", "найдорожчі", "преміум"}, cmd_most_expensive),
    ({"порекомендуй", "що порадиш", "рекомендація", "що спробувати",
      "випадкове", "сюрприз"}, cmd_random_rec),
    ({"допоможи", "що ти вмієш", "команди", "підказка", "help"}, cmd_help),
]

CATEGORY_KEYWORDS = {
    "кава": ["кав", "еспресо", "латте", "капучин", "американо", "раф", "мокка", "матча", "ристрет"],
    "чай та напої": ["чай", "шоколад", "лимонад", "фреш", "сік"],
    "їжа": ["їж", "круасан", "сендвіч", "чізкейк", "тіраміс", "вафл", "банан", "брауні"],
}


def try_quick_command(text: str) -> str | None:
    t = text.lower()

    # Бюджетний запит: «до 80 гривень», «за 100», «не більше 90»
    m = re.search(r"(?:до|за|не більше|менше)\s*(\d+)", t)
    if m:
        return cmd_budget(int(m.group(1)))

    for phrases, fn in QUICK_COMMANDS:
        if any(p in t for p in phrases):
            return fn()

    for cat, keywords in CATEGORY_KEYWORDS.items():
        if any(kw in t for kw in keywords) and any(
                w in t for w in ("список", "все", "варіанти", "які є", "що є")
        ):
            return cmd_category(cat)

    return None


# ══════════════════════════════════════════════
#  OLLAMA

MODELS = ["llama3.2", "mistral", "phi3"]
MODEL_TIMEOUT = 120


def ask_model(model: str, messages: list, timeout: int = MODEL_TIMEOUT) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 300},
    }
    try:
        r = requests.post("http://localhost:11434/api/chat", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()["message"]["content"].strip()
    except requests.exceptions.ConnectionError:
        return "[Ollama не запущено]"
    except requests.exceptions.Timeout:
        return f"[{model}: timeout]"
    except Exception as e:
        return f"[{model}: {e}]"


def ask_all_parallel(messages: list) -> dict:
    results, lock = {}, threading.Lock()

    def worker(model):
        ans = ask_model(model, messages)
        with lock:
            results[model] = ans
        ok = "✓" if not ans.startswith("[") else "✗"
        print(f"  {ok} {model}: {'OK' if ok == '✓' else ans}")

    threads = [threading.Thread(target=worker, args=(m,), daemon=True) for m in MODELS]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=MODEL_TIMEOUT + 10)
    return results


def synthesize(responses: dict, question: str) -> str:
    valid = {m: r for m, r in responses.items() if not r.startswith("[")}
    if not valid:
        return "Вибачте, всі моделі зараз недоступні. Перевірте, чи запущено Ollama."
    if len(valid) == 1:
        return list(valid.values())[0]

    parts = "\n".join(
        f"Відповідь {i + 1} ({m}): {r}" for i, (m, r) in enumerate(valid.items())
    )
    synth = [{"role": "user", "content": (
        f"Клієнт запитав: «{question}»\n\n{parts}\n\n"
        "Об'єднай у одну коротку відповідь (2–4 речення) для людини з поганим зором. "
        "Назви продукт і ціну. Тільки українська мова."
    )}]

    for model in valid:
        result = ask_model(model, synth, timeout=60)
        if not result.startswith("["):
            return result

    return max(valid.values(), key=len)


# ══════════════════════════════════════════════
#  ГОЛОС

def speak(text: str):
    print(f"\n🤖 Бот: {text}\n")
    if not GTTS_OK or _PLAYER is None:
        return
    try:
        tts = gTTS(text=text, lang="uk", slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            path = f.name
        tts.save(path)
        play_mp3(path)
        os.unlink(path)
    except Exception as e:
        print(f"  [TTS: {e}]")


def listen() -> str | None:
    if not SR_OK:
        return None
    recognizer = sr.Recognizer()
    recognizer.dynamic_energy_threshold = True
    with sr.Microphone() as source:
        print("🎤 Слухаю...")
        recognizer.adjust_for_ambient_noise(source, duration=0.6)
        try:
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=18)
            text = recognizer.recognize_google(audio, language="uk-UA")
            print(f"👤 Ви: {text}")
            return text
        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            return None
        except sr.RequestError as e:
            print(f"  [Розпізнавання: {e}]")
            return None


# ══════════════════════════════════════════════
#  ПЕРЕВІРКА OLLAMA
# ══════════════════════════════════════════════

def check_ollama():
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        available = [m["name"].split(":")[0] for m in r.json().get("models", [])]
        print(f"  Доступні моделі: {available}")
        missing = [m for m in MODELS if m not in available]
        if missing:
            print(f"\nВідсутні моделі: {missing}")
            for m in missing:
                print(f"     ollama pull {m}")
            print()
    except Exception:
        print("\n❌ Ollama не запущено! Виконайте: ollama serve\n")


def check_player():
    if _PLAYER is None:
        print("Аудіо-плеєр не знайдено. Голос вимкнено.")
        print("   Встановіть: sudo apt install mpg123\n")
    else:
        print(f"  Аудіо-плеєр: {_PLAYER[0]}")


# ══════════════════════════════════════════════
#  ОСНОВНИЙ ЦИКЛ

EXIT_PHRASES = {"дякую", "вихід", "стоп", "вийди", "до побачення", "закінчити", "exit", "quit"}


def run(voice_mode: bool):
    print("=" * 55)
    print("  ☕  БОТ-ПОМІЧНИК КАВ'ЯРНІ «АРОМА КАВА»")
    print("      Для людей із слабким зором")
    print("      Три LLM (Ollama) + Голосовий I/O")
    print("=" * 55)
    check_ollama()
    check_player()

    history: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]

    speak(
        "Вітаю у кав'ярні Арома Кава! Я ваш голосовий помічник. "
        "Запитайте про напій або страву, скажіть «меню» щоб почути все меню, "
        "або «порекомендуй» для рекомендації. "
        "Щоб завершити — скажіть «дякую»."
    )

    no_input_streak = 0

    while True:
        if voice_mode:
            user_text = listen()
            if user_text is None:
                no_input_streak += 1
                if no_input_streak >= 3:
                    speak("Ви ще тут? Просто запитайте — я допоможу.")
                    no_input_streak = 0
                continue
            no_input_streak = 0
        else:
            try:
                user_text = input("👤 Ви: ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not user_text:
                continue

        if any(w in user_text.lower() for w in EXIT_PHRASES):
            speak("Дякую за відвідування кав'ярні Арома Кава! До побачення!")
            break

        # Швидкі команди — без LLM, миттєво
        quick = try_quick_command(user_text)
        if quick:
            speak(quick)
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": quick})
            continue

        # Три LLM паралельно
        speak("Підбираю відповідь, зачекайте...")
        history.append({"role": "user", "content": user_text})

        ctx = [history[0]] + history[-6:]

        print(f"\n⚙Запитую {len(MODELS)} моделі паралельно...")
        responses = ask_all_parallel(ctx)

        print("\nСирі відповіді:")
        for model, resp in responses.items():
            print(f"  [{model}]: {resp[:110].replace(chr(10), ' ')}...")

        print("\nСинтезую...")
        final = synthesize(responses, user_text)

        speak(final)
        history.append({"role": "assistant", "content": final})


# ══════════════════════════════════════════════
#  ENTRY POINT

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Голосовий бот кав'ярні (Ollama LLM)")
    parser.add_argument("--text", action="store_true", help="Текстовий режим (без мікрофона)")
    args = parser.parse_args()

    voice = not args.text
    if voice and not SR_OK:
        print("SpeechRecognition не встановлено → текстовий режим.\n   pip install SpeechRecognition\n")
        voice = False

    run(voice_mode=voice)
