import streamlit as st
import pandas as pd
import polars as pl  # БИБЛИОТЕКА ДЛЯ СКОРОСТИ
import json
import re
import os
import random
import numpy as np
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
from collections import Counter, defaultdict
from datetime import datetime, timedelta
import emoji 
from PIL import Image, UnidentifiedImageError
from streamlit_lottie import st_lottie

# ПОПЫТКА ИМПОРТА ФРАГМЕНТОВ (Для оптимизации Викторины)
try:
    from streamlit import fragment
except ImportError:
    try:
        from streamlit import experimental_fragment as fragment
    except ImportError:
        def fragment(func):
            return func
import time
# === НАСТРОЙКИ ДАТЫ ===
# Укажите здесь ту же дату, от которой идет отсчет в хедере
# Формат: Год, Месяц, День
REL_START_DATE = pd.Timestamp(datetime(2025, 9, 14))
# --- DEBUG & PROFILING TOOL (ОТЛАДКА) ---
# Этот класс поможет нам понять, на чем именно тормозит приложение
class Profiler:
    def __init__(self):
        self.log = []
        self.start_global = time.time()
        self.last_check = self.start_global

    def checkpoint(self, label):
        now = time.time()
        duration = now - self.last_check
        self.log.append(f"⏱ {label}: {duration:.4f} сек")
        self.last_check = now

    def finish(self):
        total = time.time() - self.start_global
        self.log.append(f"🏁 ВСЕГО: {total:.4f} сек")
        # Выводим в сайдбар (можно свернуть)
        with st.sidebar.expander("🛠 Отладка производительности", expanded=False):
            st.code("\n".join(self.log), language="text")

# Инициализируем профайлер в начале скрипта
profiler = Profiler()
# ---------------- НАСТРОЙКИ ----------------
st.set_page_config(page_title="100 дней вместе", page_icon="🎀", layout="wide")
CACHE_FILE = "optimized_chat.parquet"
# ---------------- СТИЛЬ (НОВЫЙ ДИЗАЙН) ----------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;800&family=Pacifico&display=swap');

    /* Глобальный фон */
    .stApp { 
        background: linear-gradient(135deg, #fff0f5 0%, #fff5ee 100%);
        font-family: 'Nunito', sans-serif;
    }
    
    h1, h2, h3 { color: #FF69B4 !important; font-family: 'Nunito', sans-serif; font-weight: 800; }
    h1 { font-family: 'Pacifico', cursive; letter-spacing: 2px; }

    /* --- НОВЫЙ HERO HEADER --- */
    .hero-container {
        background: linear-gradient(120deg, #ff9a9e 0%, #fecfef 100%);
        border-radius: 25px;
        padding: 40px 20px;
        text-align: center;
        color: white;
        box-shadow: 0 15px 30px rgba(255, 105, 180, 0.3);
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }
    .hero-title { font-family: 'Pacifico', cursive; font-size: 3em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .hero-days { font-size: 5em; font-weight: 800; line-height: 1; margin: 10px 0; }
    .hero-subtitle { font-size: 1.2em; font-weight: 600; opacity: 0.9; }
    .heart-beat { animation: heartbeat 1.5s infinite; display: inline-block; }
    
    .winner-box { background: rgba(255,255,255,0.6); backdrop-filter: blur(10px); padding: 20px; border-radius: 20px; border: 2px solid rgba(255, 204, 213, 0.5); text-align: center; margin-bottom: 15px; box-shadow: 0 8px 32px 0 rgba(31,38,135,0.07); transition: transform 0.3s ease; }
    .winner-box:hover { transform: translateY(-5px); border: 2px solid rgba(255, 204, 213, 1); }
    .winner-name { color: #FF69B4; font-size: 20px; font-weight: 800; margin: 5px 0; }
                    
    @keyframes heartbeat {
        0% { transform: scale(1); }
        50% { transform: scale(1.2); }
        100% { transform: scale(1); }
    }

    /* --- НОВЫЙ TIMELINE (ИСТОРИЯ) --- */
    .timeline-container {
        position: relative;
        padding: 20px 0;
    }
    .timeline-item {
        position: relative;
        padding-left: 40px;
        margin-bottom: 30px;
        border-left: 3px solid #ffccd5;
    }
    .timeline-dot {
        position: absolute;
        left: -9px;
        top: 0;
        width: 15px;
        height: 15px;
        border-radius: 50%;
        background: #FF69B4;
        border: 3px solid white;
        box-shadow: 0 0 0 2px #FF69B4;
    }
    .timeline-date {
        font-size: 0.85em;
        color: #FF69B4;
        font-weight: 700;
        text-transform: uppercase;
        margin-bottom: 5px;
        display: block;
    }
    .timeline-card {
        background: white;
        border-radius: 15px;
        padding: 15px;
        box-shadow: 0 5px 15px rgba(0,0,0,0.05);
        border: 1px solid #fff0f5;
        transition: transform 0.2s;
    }
    .timeline-card:hover { transform: translateX(5px); border-color: #FFB6C1; }
    .timeline-icon { font-size: 1.5em; margin-right: 10px; float: left; }
    .timeline-content { margin-left: 40px; }
    .timeline-title { font-weight: 800; color: #444; font-size: 1.1em; margin-bottom: 5px; }
    .timeline-text { font-size: 0.95em; color: #666; font-style: italic; }
    .timeline-author { font-size: 0.8em; color: #aaa; margin-top: 5px; text-align: right; }

    /* ОСТАЛЬНЫЕ СТИЛИ (Сохраняем старые для совместимости) */
        div[data-testid="stMetric"], .first-time-box, .prediction-box {
        background-color: white;
        border-radius: 20px;
        padding: 20px;
        box-shadow: 0 10px 25px rgba(255, 105, 180, 0.1);
        border: 1px solid #fff0f5;
        transition: transform 0.2s ease;
    }
    
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: #FFB6C1;
    }
    div[data-testid="stVerticalBlockBorderWrapper"], .stVerticalBlockBorderWrapper {
        background-color: white !important; border: 1px solid #ffeef2 !important; 
        border-radius: 20px !important; padding: 20px !important; margin-bottom: 20px !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px; background-color: rgba(255, 255, 255, 0.6); padding: 15px;
        border-radius: 25px; flex-wrap: wrap; box-shadow: 0 4px 15px rgba(0,0,0,0.03);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: white; border-radius: 15px; padding: 10px 25px;
        border: 1px solid #ffe4e1; font-weight: 600; color: #888;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, #FF69B4, #FFB6C1) !important; color: white !important;
        border: none; box-shadow: 0 4px 12px rgba(255, 105, 180, 0.4);
    }
    .sticker-context-box {
        background-color: #f8f9fa; border-left: 4px solid #FF69B4; padding: 8px 12px;
        margin-top: 8px; border-radius: 0 8px 8px 0; font-size: 14px; color: #333; font-weight: 600;
    }
    .quiz-text{ text-align: center;
                font-size: 20px;
                font-weight: 800;
                margin: 5px 0;
                background-color: white;
                border-radius: 10px;
                width: 100%;
                height: 100px;
                border: 2px solid rgba(255, 204, 213, 0.5);
                transition: ease 0.5s;
                padding:10px;
                align-content: center;
               }
    .quiz-text:hover{ text-align: center;
                font-size: 20px;
                font-weight: 850;
                margin: 5px 0;
                background-color: white;
                border-radius: 10px;
                width: 100%;
                height: 100px;
                border: 2px solid rgba(255, 204, 213, 1);
                box-shadow: rgba(0,0,0,0.1) 3px 4px;
                align-content: center;
               }  
    .stProgress > div > div > div > div {
                background: linear-gradient(120deg, #ff9a9e 0%, #fecfef 100%);
            }   
    .st-emotion-cache-17qp3xt {
                width: calc(100% - 1rem);
                flex: 1 1 calc(100% - 1rem);
            }
    .st-emotion-cache-1ne20ew{
            -moz-box-pack: start;
            border-radius: 0.5rem;
            overflow: visible;
            display: flex;
            gap: 1rem;
            width: 100%;
            max-width: 100%;
            height: auto;
            min-width: 1rem;
            flex-flow: column;
            flex: 1 1 0%;
            -moz-box-align: start;
            align-items: start;
            justify-content: start;
            border: 2px solid rgba(255, 204, 213, 0.3);
            background-color: white;
            padding: calc(2rem);
            transition:ease 0.5s;
            }
    .st-emotion-cache-1ne20ew:hover{
            transform: translateY(-5px); border: 2px solid rgba(255, 204, 213, 1);
            }
    .st-f0 {
            background-color: rgba(49, 51, 63, 0);
            }
    input[type=text]{background-color:white;}
    input[type=text]::placeholder {
            color: rgba(49, 51, 63, 0.6);
            }
    div[data-baseweb="tab-highlight"],div[data-baseweb="tab-border"]{visibility:hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------- ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ----------------
STOP_WORDS = {'и','в','во','не','что','он','на','я','с','со','как','а','то','все','она','так','его','но','да','ты','к','у','же','вы','за','бы','по','ее','мне','вот','от','меня','еще','нет','о','из','ему','когда','ну','или','мы','тебя','их','была','чтоб','без','будто','будет','тогда','кто','это','просто','очень','ладно','щас','почему','через','всё','ещё','про','только','было','теперь','даже','вдруг','ли','если','уже','ни','быть','был','него','до','вас','нибудь','опять','уж','вам','ведь','там','потом','себя','ничего','ей','может','они','тут','где','есть','надо','ней','для','чем','сам','чего','раз','тоже','себе','под','ж','этот','того','потому','этого','какой','совсем','ним','здесь','этом','один','почти','мой','тем','чтобы','вообще','типо','капец','наверное','блин','ахаха','пхпх','хаха','кажется','такой','который','хотя','буду','тебе','привет','знаю','пххпхпхп','вхвхахах','вхвхвххв'}


def clean_text(text):
    if text is None: return ""
    # Оставляем только буквы и пробелы
    text = re.sub(r'[^а-яё\s]', '', str(text).lower())
    return " ".join(w for w in text.split() if w not in STOP_WORDS and len(w) > 2)

def clean_text_for_prediction(text):
    text = re.sub(r'[^а-яё\s]', '', str(text).lower())
    return text.split()
@st.cache_data
def get_ngrams(text_series, n=2, top_k=10):
    all_text = " ".join(text_series.dropna().apply(clean_text))
    words = all_text.split()
    if len(words) < n: return []
    ngrams = zip(*[words[i:] for i in range(n)])
    return Counter([" ".join(ngram) for ngram in ngrams]).most_common(top_k)

def extract_emojis(text):
    return [c for c in text if c in emoji.EMOJI_DATA]


def format_time(minutes):
    if pd.isna(minutes) or minutes == 0: return "0 сек"
    seconds = int(minutes * 60)
    mins = seconds // 60
    secs = seconds % 60
    parts = []
    if mins > 0: parts.append(f"{mins} мин")
    if secs > 0 or mins == 0: parts.append(f"{secs} сек")
    return " ".join(parts)

@st.cache_data
def build_markov_model(text_series):
    model = defaultdict(list)
    for text in text_series:
        words = clean_text_for_prediction(text)
        for i in range(len(words) - 1):
            model[words[i]].append(words[i+1])
    return model

def predict_phrase(model, seed_word, length=7):
    current_word = seed_word.lower().strip()
    sentence = [current_word]
    for _ in range(length):
        if current_word in model:
            next_options = model[current_word]
            word_counts = Counter(next_options)
            words, counts = zip(*word_counts.items())
            current_word = random.choices(words, weights=counts, k=1)[0]
            sentence.append(current_word)
        else:
            break
    return " ".join(sentence).capitalize()

@st.cache_data
def parse_discord_data(filepath):
    """Парсинг JSON от DiscordChatExporter (Fix: Исправлен формат времени)"""
    if not os.path.exists(filepath):
        return pd.DataFrame()
    
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
        
        msgs = data.get("messages", [])
        
        parsed = []
        for m in msgs:
            # Пропускаем системные сообщения
            if m.get("type") not in ["Default", "Reply"]:
                continue
                
            parsed.append({
                "date": m.get("timestamp"),
                "from": m.get("author", {}).get("name", "Unknown"),
                "text": m.get("content", ""),
                "file": m.get("attachments", [{}])[0].get("url") if m.get("attachments") else None,
                "media_type": "photo" if m.get("attachments") else None
            })
            
        df = pd.DataFrame(parsed)
        if not df.empty:
            # !!! ИЗМЕНЕНИЕ ЗДЕСЬ !!!
            # 1. format='mixed' позволяет Pandas самому разобраться, где день, а где месяц, даже если форматы скачут.
            # 2. utc=True приводит таймзону к нулю (чтобы +02:00 и +03:00 стали одним временем).
            # 3. .dt.tz_localize(None) убирает информацию о таймзоне совсем, чтобы можно было склеить с данными Telegram.
            df["date"] = pd.to_datetime(df["date"], format='mixed', utc=True).dt.tz_localize(None)
            
            df["text"] = df["text"].astype(str)
            
        return df
    except Exception as e:
        # Выводим детальную ошибку, если снова упадет
        st.error(f"Ошибка обработки дат в Discord: {e}")
        return pd.DataFrame()
@st.cache_data
def Create_word_Cloud():
    profiler.checkpoint("Начало генерации облака")
    st.subheader("☁️ Облако любви")
    all_words = " ".join(df["text"].apply(clean_text))
    
    if all_words:
        try:
            mask = np.array(Image.open("heart_mask.png"))
        except:
            mask = None

        wc = WordCloud(
            width=1000, height=800, 
            background_color="white", 
            colormap="Reds",
            mask=mask,
            contour_width=2, 
            contour_color='firebrick',
            font_path="arial.ttf" if os.path.exists("arial.ttf") else None
        ).generate(all_words)
        
        st.image(wc.to_array(), width='stretch')
        profiler.checkpoint("Генерация облака завершена")
# ================== ФИНАЛЬНЫЙ БЛОК ЗАГРУЗКИ (ИСПРАВЛЕННЫЙ) ==================

# ================== ИСПРАВЛЕННАЯ ЗАГРУЗКА (ВЕРНУЛИ file) ==================

def prepare_text_for_polars(val):
    if isinstance(val, str):
        return val
    elif isinstance(val, list):
        res = []
        for part in val:
            if isinstance(part, str):
                res.append(part)
            elif isinstance(part, dict) and "text" in part:
                res.append(str(part["text"]))
        return "".join(res)
    elif isinstance(val, dict):
        return val.get("text", "")
    elif val is None:
        return ""
    else:
        return str(val)

@st.cache_data(show_spinner="Обработка архива... (в первый раз это займет время)")
def load_data():
    # 1. ПРОВЕРКА КЭША
    if os.path.exists(CACHE_FILE):
        try:
            return pl.read_parquet(CACHE_FILE).to_pandas()
        except Exception as e:
            st.warning(f"Кэш поврежден, пересоздаем: {e}")

    # 2. ЗАГРУЗКА ИЗ JSON
    data_frames = []
    
    if os.path.exists("result.json"):
        try:
            with open("result.json", encoding="utf-8") as f:
                json_data = json.load(f)
            
            # --- Шаг A: Грузим в Pandas ---
            df_pandas = pd.DataFrame(json_data["messages"])
            
            if "from" in df_pandas.columns:
                df_pandas = df_pandas.dropna(subset=["from"])
            
            # --- Шаг B: Подготовка данных ---
            # 1. Чистим текст
            if "text" in df_pandas.columns:
                df_pandas["text"] = df_pandas["text"].apply(prepare_text_for_polars)
            
            # 2. Чистим даты
            if "date" in df_pandas.columns:
                df_pandas["date"] = pd.to_datetime(df_pandas["date"], format="%Y-%m-%dT%H:%M:%S", errors='coerce')

            # 3. ГАРАНТИРУЕМ НАЛИЧИЕ КОЛОНОК (Fix KeyError: 'file')
            # Если каких-то колонок нет в JSON, создаем их пустыми, чтобы код не падал
            needed_cols = ["file", "photo", "media_type", "thumbnail"]
            for col in needed_cols:
                if col not in df_pandas.columns:
                    df_pandas[col] = None

            # 4. Выбираем колонки для сохранения (добавили file, photo и т.д.)
            cols_to_keep = ["id", "type", "date", "from", "text", "file", "photo", "media_type"]
            # Оставляем только те, что реально есть (на случай если JSON совсем странный)
            final_cols = [c for c in cols_to_keep if c in df_pandas.columns]
            df_pandas = df_pandas[final_cols]

            # --- Шаг C: Конвертация в Polars ---
            # Конвертируем все объекты в строки, чтобы Polars не ругался на смешанные типы в file/photo
            # (там может быть null или string)
            df_pl = pl.from_pandas(df_pandas)
            
            data_frames.append(df_pl)

        except Exception as e:
            st.error(f"Ошибка чтения result.json: {e}")

    if os.path.exists("discord.json"):
        try:
             pass # Тут код дискорда
        except:
            pass

    if not data_frames:
        return None

    df_final = pl.concat(data_frames, how="diagonal")

    # 3. ПРЕДВЫЧИСЛЕНИЯ
    df_final = df_final.with_columns(
        pl.col("text").str.len_chars().fill_null(0).alias("len")
    )
    
    df_final = df_final.with_columns(
        ((pl.col("len") > 30) & 
         (pl.col("len") < 250) & 
         (~pl.col("text").str.contains("http"))).alias("is_quiz_candidate")
    )

    name_mapping = {
        "my princess🖤": "Принцесса", "kiss_freak": "Принцесса",
        "tenfy_": "Милый", "April": "Милый"
    }
    # Безопасная замена имен
    if "from" in df_final.columns:
        df_final = df_final.with_columns(
            pl.col("from").replace(name_mapping, default=pl.col("from"))
        )

    df_final = df_final.sort("date")

    # 4. СОХРАНЕНИЕ
    df_final.write_parquet(CACHE_FILE)
    
    return df_final.to_pandas()

# ОПРЕДЕЛЯЕМ МОДАЛЬНОЕ ОКНО (DIALOG)
@st.dialog("Подробная статистика")
def show_winner_details(title, description, sorted_items, suffix):
    st.markdown(f"### {title}")
    st.info(description)
    st.markdown("---")
    st.markdown("#### 📊 Рейтинг участников:")
    
    if sorted_items:
        winner_name = sorted_items[0][0]
        for name, score in sorted_items:
            if isinstance(score, float):
                val = format_time(score) if suffix == "мин" else f"{score:.1f}"
            else:
                val = str(score)
            
            final_suffix = "" if suffix == "мин" else suffix
            
            if name == winner_name:
                st.markdown(f"🥇 **{name}: {val} {final_suffix}**")
            else:
                st.markdown(f"🔹 {name}: {val} {final_suffix}")

# ФУНКЦИЯ КАРТОЧЕК
def draw_winner_card(title, stats_dict, emoji_icon="🏆", suffix="", reverse=False, description=""):
    if not stats_dict: return
    
    sorted_items = sorted(stats_dict.items(), key=lambda item: item[1], reverse=not reverse)
    if not sorted_items: return
    
    winner, win_score = sorted_items[0]
    
    if isinstance(win_score, float):
        score_display = format_time(win_score) if suffix == "мин" else f"{win_score:.1f}"
    else:
        score_display = str(win_score)
        
    final_suffix = "" if suffix == "мин" else suffix

    # Рисуем красивую HTML карточку
    st.markdown(f"""
    <div class="winner-box">
        <div class="winner-icon" style="font-size:35px; margin-bottom:10px;">{emoji_icon}</div>
        <div class="winner-title" style="color: #aaa; font-size: 13px; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;">{title}</div>
        <div class="winner-name">{winner}</div>
        <div class="winner-score" style="color:#555; font-weight:700;">{score_display} {final_suffix}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Кнопка для открытия модального окна
    if st.button("🔍 Подробнее", key=f"btn_{title}",width='stretch'):
        show_winner_details(title, description, sorted_items, suffix)


# ---------------- ВИКТОРИНА С ВИЗУАЛИЗАЦИЕЙ ЗАГРУЗКИ ----------------
@fragment
def render_quiz_tab(df, selected):
    st.subheader("🎮 Угадай автора")
    
    # Инициализация состояния сессии для викторины
    if 'quiz_state' not in st.session_state:
        st.session_state.quiz_state = "intro"
        st.session_state.quiz_score = 0
        st.session_state.quiz_index = 0
        st.session_state.quiz_questions = []
        st.session_state.quiz_last_res = None

    # СОСТОЯНИЕ 1: ЭКРАН ПРИВЕТСТВИЯ
    if st.session_state.quiz_state == "intro":
        st.markdown("""<div style="text-align:center; padding: 20px;"><h3>Попробуй угадать, кто это написал!</h3></div>""", unsafe_allow_html=True)
        
        # Кнопка старта
        if st.button("🚀 НАЧАТЬ ИГРУ", width='stretch'):
            # Показываем пользователю, что мы работаем
            with st.spinner('🎲 Перемешиваем миллион сообщений... ищем лучшие вопросы...'):
                start_search = time.time() # Таймер для отладки
                
                # 1. Фильтрация (самая тяжелая часть)
                # Проверяем, есть ли колонка-оптимизатор (из прошлого шага)
                if "is_quiz_candidate" in df.columns:
                    # Быстрая фильтрация по булевой маске
                    quiz_pool = df[
                        (df["is_quiz_candidate"] == True) & 
                        (df["from"].isin(selected))
                    ]
                else:
                    # Резервный медленный вариант (если optimization не сработала)
                    st.warning("⚠️ Работаем в медленном режиме (нет индекса)")
                    quiz_pool = df[
                        (df["text"].str.len() > 30) & 
                        (df["text"].str.len() < 250) & 
                        (df["from"].isin(selected))
                    ]
                
                # Отладка: сколько нашли кандидатов
                # st.toast(f"Найдено кандидатов: {len(quiz_pool)}") 

                if len(quiz_pool) < 10:
                    st.error(f"Слишком мало сообщений для игры ({len(quiz_pool)}). Выберите больше авторов!")
                else:
                    # 2. Выборка 10 случайных вопросов
                    try:
                        subset = quiz_pool.sample(10)
                        # Конвертируем в список словарей (самый быстрый формат для работы)
                        st.session_state.quiz_questions = subset[['text', 'from', 'date']].to_dict('records')
                        
                        # 3. Смена состояния
                        st.session_state.quiz_state = "playing"
                        st.session_state.quiz_score = 0
                        st.session_state.quiz_index = 0
                        st.session_state.quiz_last_res = None
                        
                        # Замер времени для отладки
                        # st.toast(f"Подготовка заняла: {time.time() - start_search:.2f} сек")
                        
                        st.rerun()
                    except Exception as e:
                        st.error(f"Ошибка при выборке вопросов: {e}")

    # СОСТОЯНИЕ 2: ИГРОВОЙ ПРОЦЕСС
    elif st.session_state.quiz_state == "playing":
        q_idx = st.session_state.quiz_index
        
        # Если вопросы кончились
        if q_idx >= 10:
            st.session_state.quiz_state = "finished"
            st.rerun()
            
        q_data = st.session_state.quiz_questions[q_idx]
        
        # Прогресс бар
        st.progress((q_idx) / 10)
        st.markdown(f"**Вопрос {q_idx + 1}/10**")
        
        # Само сообщение
        st.markdown(f"""
        <div class="quiz-container" style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center; font-size: 1.2em;">
            "{q_data['text']}"
        </div>
        """, unsafe_allow_html=True)
        
        # Кнопки ответов
        if st.session_state.quiz_last_res is None:
            st.write("Кто это написал?")
            cols = st.columns(len(selected))
            for i, author in enumerate(selected):
                # Ключ buttons должен быть уникальным для каждого шага
                if cols[i].button(author, key=f"ans_{q_idx}_{i}", width='stretch'):
                    if author == q_data['from']:
                        st.session_state.quiz_score += 1
                        st.session_state.quiz_last_res = "correct"
                    else:
                        st.session_state.quiz_last_res = "wrong"
                    st.rerun()
        else:
            # Показ результата
            if st.session_state.quiz_last_res == "correct":
                st.success(f"✅ ВЕРНО! Это действительно {q_data['from']}")
            else:
                st.error(f"❌ МИМО! Это был(а) {q_data['from']}")
                
            if st.button("Следующий вопрос ➡️", width='stretch', key="next_btn"):
                st.session_state.quiz_index += 1
                st.session_state.quiz_last_res = None
                st.rerun()

    # СОСТОЯНИЕ 3: ФИНАЛ
    elif st.session_state.quiz_state == "finished":
        score = st.session_state.quiz_score
        
        st.markdown(f"""
        <div class="winner-box" style="padding: 40px; text-align: center; background-color: #d4edda; border-radius: 15px; border: 2px solid #c3e6cb;"> 
            <h1 style="color: #155724;">🏁 Твой счет: {score}/10</h1> 
            <p style="font-size: 1.2em;">Ты отлично знаешь вашу переписку!</p> 
        </div>
        """, unsafe_allow_html=True)
        
        if score > 8:
            st.balloons()
            
        if st.button("🔄 Сыграть снова", width='stretch'):
            st.session_state.quiz_state = "intro"
            st.rerun()

# ---------------- ЗАГРУЗКА ДАННЫХ ----------------
df_raw = load_data()
if df_raw is None:
    st.error("⚠️ Файл result.json не найден! Положи его в папку с проектом.")
    st.stop()
profiler.checkpoint("Загрузка данных (load_data)")

authors = df_raw["from"].unique().tolist()
with st.sidebar:
    st.header("⚙️ Настройки")
    selected = st.multiselect("Участники", authors, default=authors)
    if st.button("🔄 Обновить данные"):
        st.session_state.clear()
        st.cache_data.clear()
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 🎲 Момент из жизни")
    if st.button("Показать случайное"):
        random_msg = df_raw.sample(1).iloc[0]
        st.info(f"**{random_msg['date'].strftime('%d.%m.%Y')}:**\n\n{random_msg['text']}")

df = df_raw[df_raw["from"].isin(selected)].copy()
df["hour"] = df["date"].dt.hour
df["len"] = df["text"].apply(len)

markov_model = build_markov_model(df["text"])

# ---------------- ГЛАВНАЯ СТРАНИЦА (HERO HEADER) ----------------
# Убираем старый st.title("💖 100 Дней Вместе"), так как у нас теперь красивый header
start_date = datetime(2025, 9, 13, 22, 35, 0)
now = datetime.now()
diff = now - start_date

days = diff.days
hours = (diff.seconds // 3600)
minutes = (diff.seconds % 3600) // 60

# CSS стили вынесены отдельно и экранированы, HTML формируется отдельно
st.markdown(f"""
<style>
    .hero-container {{
        background: linear-gradient(120deg, #ff9a9e 0%, #fecfef 100%);
        border-radius: 25px;
        padding: 40px 20px;
        text-align: center;
        color: white;
        box-shadow: 0 15px 30px rgba(255, 105, 180, 0.3);
        margin-bottom: 30px;
        position: relative;
        overflow: hidden;
    }}
    .hero-title {{ font-family: 'Pacifico', cursive; font-size: 3em; margin: 0; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }}
    .hero-days {{ font-size: 5em; font-weight: 800; line-height: 1; margin: 10px 0; }}
    .hero-subtitle {{ font-size: 1.2em; font-weight: 600; opacity: 0.9; }}
    .heart-beat {{ animation: heartbeat 1.5s infinite; display: inline-block; }}
    
    @keyframes heartbeat {{
        0% {{ transform: scale(1); }}
        50% {{ transform: scale(1.2); }}
        100% {{ transform: scale(1); }}
    }}
</style>

<div class="hero-container">
    <div class="hero-title">🎀100 Дней Вместе🎀</div>
    <div class="hero-days">{days}<span style="font-size:0.3em; margin-left:10px;">дней</span></div>
    <div class="hero-subtitle">
        {hours} ч. {minutes} мин. <span class="heart-beat">❤️</span> бесконечной любви
    </div>
</div>
""", unsafe_allow_html=True)
# ---------------- КЭШИРОВАНИЕ ГЛОБАЛЬНОЙ СТАТИСТИКИ ----------------
@st.cache_data(show_spinner="Подсчет слов и символов (1 млн сообщений)...")
def get_global_metrics(df):
    """
    Считает общую статистику: слова, символы, топы отправителей.
    Использует векторизацию для скорости.
    """
    # 1. Считаем количество слов (быстрый метод через подсчет пробелов)
    # Это работает в 50 раз быстрее, чем split() каждого сообщения
    if "text" in df.columns:
        # Конвертируем в строки и считаем пробелы + 1 = примерное кол-во слов
        # fillna('') нужно, чтобы не упало на пустых
        text_series = df["text"].fillna("").astype(str)
        word_counts = text_series.str.count(' ') + 1
        total_words = word_counts.sum()
        
        # Статистика по авторам (группировка по from)
        # Создаем временный DF для группировки, чтобы не копировать весь огромный df
        temp_df = pd.DataFrame({
            'from': df['from'],
            'words': word_counts,
            'chars': df['len'] # Мы посчитали len еще в load_data
        })
        
        author_stats = temp_df.groupby('from').sum()
    else:
        total_words = 0
        author_stats = pd.DataFrame()

    # 2. Общие цифры
    total_msg = len(df)
    total_days = (df["date"].max() - df["date"].min()).days if not df.empty else 0
    profiler.checkpoint("Подготовка глобальных значений завершена")
    return total_msg, total_words, total_days, author_stats
# ---------------- КЭШИРОВАНИЕ ТЯЖЕЛЫХ ГРАФИКОВ (ОПТИМИЗИРОВАННАЯ) ----------------
@st.cache_data(show_spinner="Генерация облака слов...")
def get_heavy_analytics(df):
    """
    Генерирует облако и N-граммы.
    ОПТИМИЗАЦИЯ: Агрессивное сэмплирование для N-грамм (20k вместо 1M).
    """
    wc = None
    ngrams_list = []
    
    if "text" in df.columns:
        # 1. ОБЛАКО СЛОВ
        # Берем только длинные слова, чтобы ускорить склейку
        # Если строк > 200k, берем сэмпл для облака тоже (визуально разницы нет)
        if len(df) > 200000:
            text_data = df["text"].dropna().sample(200000).astype(str)
        else:
            text_data = df["text"].dropna().astype(str)
            
        text_combined = " ".join([t for t in text_data if len(t) > 3])
        text_clean = re.sub(r'[^а-яёa-z\s]', '', text_combined.lower())
        
        if text_clean:
            try:
                mask = np.array(Image.open("heart_mask.png")) if os.path.exists("heart_mask.png") else None
                # Max words ограничиваем, чтобы быстрее рендерилось
                wc = WordCloud(
                    width=600, height=400, 
                    background_color="white", 
                    colormap="Reds",
                    mask=mask,
                    max_words=100, 
                    stopwords=STOP_WORDS
                ).generate(text_clean)
            except Exception:
                pass

        # 2. N-ГРАММЫ (ТУТ БЫЛ ТОРМОЗ)
        # Берем 20 000 случайных сообщений. Этого достаточно для поиска частых фраз.
        # Это снизит время с 5 сек до 0.3 сек.
        sample_size = min(20000, len(df))
        sample_text_series = df["text"].dropna().sample(sample_size).astype(str)
        sample_text = " ".join(sample_text_series)
        
        # Быстрая очистка
        words = re.sub(r'[^а-яёa-z\s]', '', sample_text.lower()).split()
        words = [w for w in words if w not in STOP_WORDS and len(w) > 2]
        
        if words:
            bi_grams = zip(words, words[1:])
            counts = Counter(bi_grams)
            for bigram, count in counts.most_common(10):
                # Фильтруем мусор
                if count > 1:
                    ngrams_list.append((f"{bigram[0]} {bigram[1]}", count))

    return wc, ngrams_list
# ---------------- КЭШИРОВАНИЕ ЗАЛА СЛАВЫ (С ПОДДЕРЖКОЙ ДАТЫ) ----------------
@st.cache_data(show_spinner="Подсчет профилей личности и словаря...")
def get_hall_of_fame_data(df, selected_authors, start_date=None):
    """
    Рассчитывает метрики.
    start_date: дата начала отношений. Если передана, метрики нежности считаются от неё.
    """
    if "hour" not in df.columns:
        df["hour"] = df["date"].dt.hour

    # --- БЛОК 1: ОБЩАЯ ИСТОРИЯ (Считаем по всему времени) ---
    
    # 1. ГЛАВНЫЙ БОЛТУН
    msg_counts = df["from"].value_counts().to_dict()

    # 2. САМЫЙ БЫСТРЫЙ
    df_sorted = df.sort_values("date")
    time_diffs = df_sorted["date"].diff().dt.total_seconds() / 60
    author_changed = df_sorted["from"] != df_sorted["from"].shift()
    
    replies_df = pd.DataFrame({'from': df_sorted['from'], 'diff': time_diffs})
    replies_df = replies_df[author_changed & (replies_df['diff'] < 720)]
    reply_speed = replies_df.groupby("from")['diff'].mean().to_dict()

    # 3. ЛЕВ ТОЛСТОЙ
    len_mean = df.groupby("from")["len"].mean().to_dict()

    # 4. ИНИЦИАТОР
    initiators_mask = time_diffs > 360
    initiators = df_sorted[initiators_mask]["from"].value_counts().to_dict()

    # 5. ПОЧЕМУЧКА
    questions_count = df[df["text"].str.contains(r"\?", na=False)]["from"].value_counts().to_dict()



    # 7. МЕДИА И ССЫЛКИ
    links_count = df[df["text"].str.contains("http", na=False)]["from"].value_counts().to_dict()
    
    if "media_type" in df.columns:
        media_count = df[df["media_type"].notna() | (df["media_type"] != "")] ["from"].value_counts().to_dict()
    elif "file" in df.columns:
        media_count = df[df["file"].notna()]["from"].value_counts().to_dict()
    else:
        media_count = {}

    # 8. ЖАВОРОНОК И СОВА
    lark_count = df[(df["hour"] >= 6) & (df["hour"] <= 10)]["from"].value_counts().to_dict()
    owl_count = df[(df["hour"] >= 0) & (df["hour"] <= 4)]["from"].value_counts().to_dict()

    # --- БЛОК 2: ПЕРИОД ОТНОШЕНИЙ (Нежность, Эмодзи, Извинения) ---
    # Если дата задана, фильтруем df. Если нет - берем весь.
    if start_date:
        # Преобразуем start_date в datetime64 для сравнения
        ts_start = pd.to_datetime(start_date)
        df_period = df[df['date'] >= ts_start]
    else:
        df_period = df
        
    # Считаем количество сообщений ЗА ПЕРИОД (для расчета процентов)
    msg_counts_period = df_period["from"].value_counts().to_dict()
    print(msg_counts_period)
    # 9. МИЛАШКА (Слова любви)
    cute_mask = df_period["text"].str.contains(r"красив|любим|лучш|солн|умн|мил|родн|зай|кот|прекрас|обожаю|нежн|скуча|любл|ахуен|целу|муа|секс|принц|слад|золот|лучш|я бол|сладк|хочу тебя|лучш|мув", case=False, na=False)
    cute_count = df_period[cute_mask]["from"].value_counts().to_dict()

    # 10. ЭМОДЗИ (По периоду)
    emoji_counts = {}
    # 11. СЛОВАРНЫЙ ЗАПАС (Оставляем по всей истории, это интеллект)
    vocab_counts = {}
    # 12. ИЗВИНЕНИЯ (По периоду)
    apology_mask = df_period["text"].str.contains(r"прости|извини|sorry|виноват|стыд", case=False, na=False)
    apology_count = df_period[apology_mask]["from"].value_counts().to_dict()

    # 13. --- НОВАЯ МЕТРИКА: ПОДДЕРЖКА (The Therapist) ---
    support_regex = r"всё будет|пережива|справи|спокой|забей|норм|держись|понимаю|не бойся|всё хорошо|поддерживаю|как ты"
    support_mask = df_period["text"].str.contains(support_regex, case=False, na=False)
    support_count = df_period[support_mask]["from"].value_counts().to_dict()
    
    #6. Эмоциональность
    hype_regex = r"ого|вау|круто|жесть|супер|класс|офигеть|шок|!!|капец|ужас|пиздец|ахуеть"
    hype_mask = df_period["text"].str.contains(hype_regex, case=False, na=False)
    hype_count = df_period[hype_mask]["from"].value_counts().to_dict()

    try:
        for author in selected_authors:
            # Словарь считаем по всей истории (df)
            full_text = " ".join(df[df["from"] == author]["text"].dropna().tolist())
            vocab_counts[author] = len(set(full_text.split()))

            # Эмодзи считаем по периоду (df_period)
            period_text = df_period[df_period["from"] == author]["text"].dropna()
            if 'extract_emojis' in globals():
                emoji_counts[author] = period_text.apply(lambda x: len(extract_emojis(x))).sum()
            else:
                emoji_counts[author] = 0

    except:
        emoji_counts = {a: 0 for a in selected_authors}
        vocab_counts = {a: 0 for a in selected_authors}

    return {
        # Общие (Зал Славы)
        "msg_counts": msg_counts,
        "reply_speed": reply_speed,
        "len_mean": len_mean,
        "initiators": initiators,
        "questions": questions_count,
        "hype": hype_count,           # ЗАМЕНА (было laughter)
        "support": support_count,     # НОВОЕ
        "media": media_count,
        "links": links_count,
        "lark": lark_count,
        "owl": owl_count,
        
        # Период (Нежность)
        "msg_counts_period": msg_counts_period, 
        "cute": cute_count,
        "emoji": emoji_counts,
        "apology": apology_count,
        "hype_period": hype_count,       # ЗАМЕНА (было laughter_period)
        "support_period": support_count, # НОВОЕ
        
        # Интеллект
        "vocab": vocab_counts
    }
# ---------------- КЭШИРОВАНИЕ ГРАФИКОВ (ВЕРСИЯ С ТАЙМЛАЙНОМ) ----------------
@st.cache_data(show_spinner="Анализ истории, стикеров и важных событий...")
def get_charts_data(df):
    """
    Рассчитывает данные для графиков.
    ОПТИМИЗАЦИЯ: Создает контекст стикеров через shift(), убирая циклы поиска.
    """
    # 1. ДАННЫЕ ДЛЯ ИСТОРИИ
    daily_counts = df.set_index('date').resample('D').size().reset_index(name='count')
    
    # 2. ДАННЫЕ ДЛЯ БАЛАНСА
    user_counts = df['from'].value_counts()
    
    # 3. ДАННЫЕ ДЛЯ СТИКЕРОВ
    stickers_df = pd.DataFrame()
    sticker_contexts = {} # Новый объект для готовых словарей
    if "file" in df.columns:
        # ВЕКТОРИЗАЦИЯ КОНТЕКСТА (Самое важное ускорение)
        # Создаем временную колонку с текстом предыдущего сообщения (сдвиг вниз на 1)
        # Это происходит в ядре C++, поэтому мгновенно, в отличие от Python-циклов
        df_context = df.copy() # Работаем с копией, чтобы не ломать основной df
        df_context['prev_text'] = df_context['text'].shift(1)
        
        # Фильтруем стикеры
        mask = (df_context["file"].str.contains(r'\.webp|\.tgs', na=False)) | \
               (df_context["type"] == "sticker") | \
               (df_context["media_type"] == "sticker")
        
        # Берем стикеры сразу с приклеенным контекстом
        stickers_df = df_context[mask].copy()
        
        # Оставляем только нужное
        cols_needed = ['file', 'from', 'date', 'prev_text'] # prev_text уже тут!
        stickers_df = stickers_df[[c for c in cols_needed if c in stickers_df.columns]]
        # --- ГЛАВНАЯ ОПТИМИЗАЦИЯ (ПРЕ-ВЫЧИСЛЕНИЕ) ---
        # Мы заранее собираем списки контекстов в словари.
        # Это тяжелая операция, но теперь она выполняется 1 раз в кэше.
        
        # 1. Словарь для режима "Все вместе": Key=File -> Value=[Contexts]
        # Используем dropna(), чтобы не хранить мусор
        dict_all = stickers_df.groupby('file')['prev_text'].apply(lambda x: x.dropna().tolist()).to_dict()
        
        # 2. Словарь для конкретных авторов: Key=(File, Author) -> Value=[Contexts]
        dict_auth = stickers_df.groupby(['file', 'from'])['prev_text'].apply(lambda x: x.dropna().tolist()).to_dict()
        
        sticker_contexts = {
            "all": dict_all,
            "auth": dict_auth
        }
    # 4. ДАННЫЕ ДЛЯ ДНЕЙ НЕДЕЛИ
    day_counts = df["date"].dt.day_name().value_counts()

    # 5. ДАННЫЕ ДЛЯ ТЕПЛОВОЙ КАРТЫ
    days_mapped = df["date"].dt.day_name().map(
        {'Monday': 'Пн', 'Tuesday': 'Вт', 'Wednesday': 'Ср', 'Thursday': 'Чт', 
         'Friday': 'Пт', 'Saturday': 'Сб', 'Sunday': 'Вс'}
    )
    hours = df["date"].dt.hour
    hm_source = pd.DataFrame({'day_name_ru': days_mapped, 'hour': hours})
    hm_data = hm_source.groupby(["day_name_ru", "hour"]).size().reset_index(name='count')

    # 6. ДАННЫЕ ДЛЯ ТАЙМЛАЙНА
    events = []
    
    # Вспомогательная функция внутри кэша
    def check_event(mask, title, icon):
        try:
            # Ищем первое совпадение
            # head(1) значительно ускоряет работу по сравнению с полной фильтрацией
            matches = df[mask].head(1) 
            if not matches.empty:
                first = matches.iloc[0]
                events.append({
                    "date": first['date'],
                    "title": title,
                    "text": first['text'],
                    "author": first['from'],
                    "icon": icon
                })
        except:
            pass

    # --- ОПРЕДЕЛЕНИЕ СОБЫТИЙ (ЛОГИКА ИЗ ТАБА) ---
    # 1. ГЛАВНЫЕ СЛОВА
    if "text" in df.columns:
        check_event(df['text'].str.contains("люблю тебя", case=False, na=False), "Первое 'Люблю тебя'", "❤️")
        check_event(df['text'].str.contains("обожаю", case=False, na=False), "Первое 'Обожаю'", "🥰")
        check_event(df['text'].str.contains("скучаю", case=False, na=False), "Первое 'Скучаю'", "🥺")

        # 2. МИЛЫЕ ПРОЗВИЩА
        check_event(df['text'].str.contains("Солнце", case=False, na=False), "Первое 'Солнышко'", "☀️")
        check_event(df['text'].str.contains("лис", case=False, na=False), "Первый 'Лис'", "🐱")
        check_event(df['text'].str.contains("Милый", case=False, na=False), "Первый 'Милый'", "🐰")
        check_event(df['text'].str.contains("принцесса", case=False, na=False), "Первое упоминание титула", "👑")
        check_event(df['text'].str.contains("красив|прекрас", case=False, na=False), "Первый комплимент", "😍")

        # 3. ДЕЙСТВИЯ И ВСТРЕЧИ
        check_event(df['text'].str.contains("можно и погулять", case=False, na=False), "Первое предложение встречи", "🌹")
        check_event(df['text'].str.contains("фильм|сериал", case=False, na=False), "Первое обсуждение кино", "🎬")
        check_event(df['text'].str.contains("кушать", case=False, na=False), "Первый разговор о еде", "🍕")
        check_event(df['text'].str.contains("спат|сон|кровать", case=False, na=False), "Первое 'Пора спать'", "😴")
        check_event(df['text'].str.contains("гулять|прогулка", case=False, na=False), "Первая прогулка", "🌳")

        # 4. ЭМОЦИИ И РИТУАЛЫ
        check_event(df['text'].str.contains("доброе утро", case=False, na=False), "Первое 'Доброе утро'", "☕")
        check_event(df['text'].str.contains("спокойной ночи|сладких снов", case=False, na=False), "Первая 'Спокойной ночи'", "🌙")
        check_event(df['text'].str.contains("прости|извини", case=False, na=False), "Первое извинение", "🤝")
        check_event(df['text'].str.contains("ахах|лол|ору|rfl", case=False, na=False), "Первый смех", "😂")
        check_event(df['text'].str.contains("обещаю", case=False, na=False), "Первое обещание", "🤞")
        check_event(df['text'].str.contains("спасибо|благодарю", case=False, na=False), "Первая благодарность", "🙏")
        check_event(df['text'].str.contains("дайсон", case=False, na=False), "Первый 'Дайсон'", "😏")

        # 5. ТЕХНИЧЕСКОЕ
        check_event(df['text'].str.contains("http", case=False, na=False), "Первая ссылка", "🔗")

    # Медиа события
    if "media_type" in df.columns:
        check_event(df['media_type'] == 'sticker', "Первый стикер", "🎭")
        check_event(df['media_type'].isin(['photo', 'video_file']), "Первое фото/видео", "📸")
        check_event(df['media_type'] == 'voice_message', "Первое голосовое", "🎤")
        check_event(df['media_type'] == 'video_message', "Первый кружочек", "🔵")

    # Сортировка по времени
    events.sort(key=lambda x: x.get('date', pd.Timestamp.min))
    profiler.checkpoint("Тяжёлые вычисления завершены")
    # ВОЗВРАЩАЕМ 7 ЭЛЕМЕНТОВ (sticker_contexts на 6-м месте)
    return daily_counts, user_counts, stickers_df, day_counts, hm_data, sticker_contexts, events
# Прогресс бар (оставляем как было, просто для контекста места вставки)
col_prog1, col_prog2 = st.columns([4, 1])
with col_prog1:
    st.caption(f"🚀 Путь к году (осталось {365 - diff.days} дн.)")
    progress = min(max(diff.days / 365, 0.0), 1.0)
    st.progress(progress)
with col_prog2:
    st.caption(f"**{int(progress*100)}%**")
profiler.checkpoint("Подготовка завершена")
st.markdown("---")

tabs = st.tabs([
    "🏆 Зал Славы", 
    "📈 Активность", 
    "⚖️ Баланс & Нежность",
    "🔮 Ванга",
    "⏳ История (Первые)",
    "🔎 Поиск", 
    "🎭 Стикеры",
    "☁️ Слова",
    "🎮 Викторина"
])
profiler.checkpoint("Создание табов")
# ================== ТАБ 1: ГЛАВНАЯ (ОПТИМИЗИРОВАННАЯ) ==================
with tabs[0]:
    # 1. Общая статистика (берем из get_global_metrics, она уже у нас есть)
    total_msg, total_words, total_days, author_stats = get_global_metrics(df)
    
    st.subheader("📊 Общая статистика")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("💌 Всего сообщений", f"{total_msg:,}".replace(",", " "))
    c2.metric("📅 Дней вместе", total_days)
    
    # Общий вес символов (быстро через sum)
    total_chars = df['len'].sum() if "len" in df.columns else 0
    c3.metric("📝 Тысяч символов", f"{total_chars/1000:.1f}k")
    
    # Слово "люблю" (быстрый поиск без regex или с simple regex)
    love_count = df["text"].str.contains("люблю", case=False, na=False).sum()
    c4.metric("❤️ Слов 'Люблю'", love_count)

    st.markdown("### 🏆 Наш Зал Славы")
    st.caption("Нажмите на кнопку под карточкой, чтобы узнать детали")

    # 2. Получаем метрики для карточек из КЭША
    # Это самая важная строка - она заменяет 10 секунд вычислений на 0.01 сек
    hof = get_hall_of_fame_data(df, selected)

    # Ряд 1
    r1c1, r1c2, r1c3, r1c4 = st.columns(4)
    with r1c1: draw_winner_card("Главный болтун", hof["msg_counts"], "🦜", description="У кого больше всего отправленных сообщений.")
    with r1c2: draw_winner_card("Самый быстрый", hof["reply_speed"], "🚀", "мин", reverse=True, description="Среднее время ответа на сообщение (чем меньше, тем лучше).")
    with r1c3: draw_winner_card("Лев Толстой", hof["len_mean"], "✍️", "симв.", description="Средняя длина одного сообщения в символах.")
    with r1c4: draw_winner_card("Инициатор", hof["initiators"], "💡", description="Кто чаще пишет первым после перерыва в общении (> 6 часов).")
    
    st.write("") 
    
    # Ряд 2
    r2c1, r2c2, r2c3, r2c4 = st.columns(4)
    with r2c1: draw_winner_card("Почемучка", hof["questions"], "🤔", description="Количество сообщений с вопросительным знаком.")
    with r2c2: draw_winner_card("Реакционер", hof["hype"], "🔥", description="Генератор энергии: 'ОГО!', 'ЖЕСТЬ', 'КРУТО'.")
    with r2c3: draw_winner_card("Медиа-магнат", hof["media"], "🎬", description="Количество отправленных фото, видео и голосовых.")
    with r2c4: draw_winner_card("Король ссылок", hof["links"], "🌐", description="Сколько ссылок (http...) было отправлено.")

    st.write("")

    # Ряд 3
    r3c1, r3c2, r3c3, r3c4 = st.columns(4)
    with r3c1: draw_winner_card("Жаворонок", hof["lark"], "☕️", description="Сообщения, отправленные утром (с 6:00 до 10:00).")
    with r3c2: draw_winner_card("Сова", hof["owl"], "🌙", description="Сообщения, отправленные глубокой ночью (с 00:00 до 04:00).")
    with r3c3: draw_winner_card("Эмодзи-мастер", hof["emoji"], "😜", description="Общее количество использованных смайликов.")
    with r3c4: draw_winner_card("Милашка", hof["cute"], "🥰", description="Использование слов любви, нежности и комплиментов.")
    
    profiler.checkpoint("Отрисовка главной завершена")
# ================== ТАБ 2: СТАТИСТИКА (FIXED) ==================
with tabs[1]:
    # ИСПРАВЛЕНИЕ: Добавили *_, чтобы собрать лишние значения (timeline_events) и не вызывать ошибку
    daily, user_counts, _, day_counts, hm_data, *_ = get_charts_data(df)
    
    col_act1, col_act2 = st.columns([2, 1])
    
    # График 1: Динамика (Area Chart)
    with col_act1:
        st.subheader("📈 Динамика сообщений")
        try:
            fig_daily = px.area(daily, x='date', y='count', 
                                title='Наша история по дням', 
                                labels={'date':'Дата', 'count':'Сообщений'})
            fig_daily.update_traces(line_color='#FF69B4', fill='tozeroy')
            fig_daily.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_daily, use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка графика динамики: {e}")
        
    # График 2: Дни недели (Bar Chart)
    with col_act2:
        st.subheader("📅 Любимый день")
        try:
            fig_bar = px.bar(day_counts, 
                            x=day_counts.index, 
                            y=day_counts.values,
                            color_discrete_sequence=['#FFB6C1'])
            fig_bar.update_layout(showlegend=False, 
                                xaxis_title=None, 
                                yaxis_title=None,
                                margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)
        except Exception as e:
            st.error(f"Ошибка графика дней: {e}")

    # График 3: Тепловая карта (Heatmap)
    st.subheader("🕒 Карта нашей активности (Часы)")
    try:
        days_order = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
        
        fig_hm = px.density_heatmap(
            hm_data, x="hour", y="day_name_ru", z="count", 
            color_continuous_scale="RdPu",
            labels={"hour": "Час", "day_name_ru": "День", "count": "Сообщений"},
            category_orders={"day_name_ru": days_order},
            title="Когда нам жарче всего общаться? 🔥"
        )
        fig_hm.update_layout(xaxis_dtick=1, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig_hm, use_container_width=True)
    except Exception as e:
        st.error(f"Ошибка тепловой карты: {e}")

    profiler.checkpoint("Отрисовка аналитики завершена")
# ================== ТАБ 3: БАЛАНС & НЕЖНОСТЬ (FIXED DATE) ==================
with tabs[2]: # Проверьте индекс (обычно tabs[2] или tabs[3])
    st.subheader("⚖️ Профили Личности")
    
    # 1. Используем глобальную константу REL_START_DATE
    # Передаем её в функцию. 
    # get_hall_of_fame_data вернет общую статистику в 'metrics_raw' 
    # и статистику за период отношений в ключах 'cute', 'apology' и т.д.
    hof = get_hall_of_fame_data(df, selected, start_date=REL_START_DATE)
    
    metrics_raw = {}
    for auth in selected:
        # Общие характеристики (считаем по всей переписке для точности профиля)
        msg_count = hof["msg_counts"].get(auth, 0)
        avg_len = hof["len_mean"].get(auth, 0)
        vocab = hof["vocab"].get(auth, 0)
        
        # Эмодзи и Скорость тоже берем общие (или можно заменить на period, если хотите)
        # Для радара лучше брать общие паттерны поведения
        e_count = hof["emoji"].get(auth, 0) # Здесь emoji вернутся за период (см. функцию), это ок
        
        # Важно: делим на кол-во сообщений ЗА ПЕРИОД, если метрика за период
        msg_count_period = hof["msg_counts_period"].get(auth, 1)
        if msg_count_period == 0: msg_count_period = 1
        
        emoji_ratio = (e_count / msg_count_period) 
        
        speed_val = hof["reply_speed"].get(auth, 60)
        if pd.isna(speed_val): speed_val = 60
        
        metrics_raw[auth] = {
            "Болтливость": msg_count,      # Общая
            "Многословность": avg_len,     # Общая
            "Словарный запас": vocab,      # Общий
            "Эмоциональность": emoji_ratio,# За период (так как emoji считаем по start_date)
            "Скорость ответа": speed_val   # Общая
        }

    # --- Radar Chart (Отрисовка) ---
    if metrics_raw:
        categories = ["Болтливость", "Многословность", "Словарный запас", "Эмоциональность", "Скорость ответа"]
        max_vals = {cat: 0 for cat in categories}
        for auth in metrics_raw:
            for cat in categories:
                if cat != "Скорость ответа":
                    max_vals[cat] = max(max_vals[cat], metrics_raw[auth][cat])
        
        fig_radar = go.Figure()
        colors = {"Принц": "#636EFA", "Принцесса": "#FF69B4"} 
        
        for auth in metrics_raw:
            values = []
            # Нормализация
            values.append((metrics_raw[auth]["Болтливость"] / max_vals["Болтливость"]) * 100 if max_vals["Болтливость"] else 0)
            values.append((metrics_raw[auth]["Многословность"] / max_vals["Многословность"]) * 100 if max_vals["Многословность"] else 0)
            values.append((metrics_raw[auth]["Словарный запас"] / max_vals["Словарный запас"]) * 100 if max_vals["Словарный запас"] else 0)
            values.append((metrics_raw[auth]["Эмоциональность"] / max_vals["Эмоциональность"]) * 100 if max_vals["Эмоциональность"] else 0)
            s = metrics_raw[auth]["Скорость ответа"]
            speed_score = max(0, 100 - s)
            values.append(speed_score)
            values.append(values[0])
            
            fig_radar.add_trace(go.Scatterpolar(
                r=values, theta=categories + [categories[0]], fill='toself', name=auth, line_color=colors.get(auth, None)
            ))

        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
            showlegend=True, title="Кто в чем круче?", height=500,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    st.markdown("---")

    # 3. Уровень нежности (СТРОГО ОТ ДАТЫ ОТНОШЕНИЙ)
    st.subheader("🧸 Уровень нежности")
    # Красиво форматируем дату для заголовка
    date_lbl = REL_START_DATE.strftime('%d.%m.%Y')
    st.caption(f"Статистика считается с момента начала отношений: {date_lbl}")
    
    plot_data = []
    tenderness_scores = {}
    
    for auth in selected:
        # Извлекаем метрики, используя новые ключи из return функции
        # Обрати внимание: hof["..."] должны совпадать с ключами в return
        msg_counts_period = hof["msg_counts_period"].get(auth, 0)

        cute_val = hof["cute"].get(auth, 0)
        hype_val = hof["hype_period"].get(auth, 0)      # Эмоции (за период)
        support_val = hof["support_period"].get(auth, 0) # Поддержка (за период)
        apology_val = hof["apology"].get(auth, 0)
        
        # Добавляем данные для столбцов
        plot_data.append({"User": auth, "Type": "Милота 🥰", "Count": cute_val})
        plot_data.append({"User": auth, "Type": "Эмоциональность🤩", "Count": hype_val})
        plot_data.append({"User": auth, "Type": "Поддержка 💕", "Count": support_val})
        plot_data.append({"User": auth, "Type": "Извинения 🙏", "Count": apology_val})
        
        # Расчет индекса (Очки: Милота + Поддержка*1.5 + Эмоции*0.5 - Извинения*0.5)
        score = (cute_val + (support_val * 3) + (hype_val * 0.2) - (apology_val * 0.2))/(msg_counts_period/100)
        tenderness_scores[auth] = score

    plot_df = pd.DataFrame(plot_data)
    
    c_bal1, c_bal2 = st.columns([2, 1])
    with c_bal1:
        if not plot_df.empty:
            fig_bal = px.bar(plot_df, x="User", y="Count", color="Type", barmode="group",
                                color_discrete_map={"Милота 🥰": "#FF69B4", "Эмоциональность🤩": "#FFD700", "Извинения 🙏": "#A9A9A9","Поддержка 💕": "#FF00E6"},
                                title="О чем мы говорим чаще?")
            fig_bal.update_layout(plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_bal, use_container_width=True)
        
    with c_bal2:
        st.markdown("### 🌡️ Индекс Любви")
        st.caption("Процент сообщений с ласковыми словами")
        for auth, score in tenderness_scores.items():
            st.metric(f"{auth}", f"{score:.1f}%", delta="Супер!" if score > 3 else "Норм")
            
    profiler.checkpoint("Отрисовка баланса завершена")
# ================== ТАБ 4: ВАНГА ==================
with tabs[3]:
    st.subheader("🔮 Нейросеть отношений")
    st.markdown("Введи слово, а я попробую продолжить фразу так, как это сделали бы мы:")
    
    col_pred1, col_pred2 = st.columns([1, 2])
    with col_pred1:
        seed = st.text_input("Начало фразы:", value="Люблю")
        length = st.slider("Сколько слов добавить?", 3, 20, 8)
        do_predict = st.button("✨ Предсказать",width='stretch')
    with col_pred2:
        if do_predict and seed:
            last_word = seed.split()[-1]
            prediction = predict_phrase(markov_model, last_word, length)
            full = seed.rsplit(' ', 1)[0] + " " + prediction if len(seed.split()) > 1 else prediction
            st.markdown(f"""<div class="prediction-box">✨ {full}...</div>""", unsafe_allow_html=True)
profiler.checkpoint("Отрисовка ванги завершена")
# ================== ТАБ 4: ИСТОРИЯ (TIMELINE) [FIXED] ==================
with tabs[4]:
    st.subheader("📜 Наша Хронология")
    
    # ИСПРАВЛЕНИЕ: Распаковываем все 7 переменных, чтобы events попал куда надо
    # Порядок в get_charts_data: daily, user, stickers, days, hm, contexts, events
    daily, user_counts, stickers_df, day_counts, hm_data, sticker_contexts, events = get_charts_data(df)
    
    if not events:
        st.info("История пока пуста. Попробуй выбрать больше участников или проверить файл данных.")
    else:
        # Генерация HTML
        timeline_html = '<div class="timeline-container">'
        
        for evt in events:
            # Теперь evt — это словарь, и evt['date'] сработает корректно
            try:
                date_str = evt['date'].strftime('%d %B %Y')
                time_str = evt['date'].strftime('%H:%M')
                
                # Экранируем кавычки и обрезаем текст
                clean_msg = str(evt['text']).replace('"', '&quot;')
                if len(clean_msg) > 100: clean_msg = clean_msg[:100] + "..."
                if len(clean_msg) < 2: clean_msg = "<i>(Вложение)</i>"
                
                timeline_html += f"""
    <div class="timeline-item">
    <div class="timeline-dot"></div>
    <span class="timeline-date">{date_str} <span style="font-weight:400; opacity:0.7">в {time_str}</span></span>
    <div class="timeline-card">
    <div class="timeline-icon">{evt['icon']}</div>
    <div class="timeline-content">
    <div class="timeline-title">{evt['title']}</div>
    <div class="timeline-text">"{clean_msg}"</div>
    <div class="timeline-author">— {evt['author']}</div>
    </div>
    </div>
    </div>"""
            except Exception as e:
                # На случай сбоя в конкретном событии, чтобы не ломать весь таб
                continue
            
        timeline_html += '</div>'
        st.markdown(timeline_html, unsafe_allow_html=True)
    profiler.checkpoint("Отрисовка истории завершена")
# ================== ТАБ 6: ПОИСК ==================
# ================== ТАБ 6 (или 5): ПОИСК ==================
with tabs[5]:
    st.subheader("🔎 Поиск воспоминаний")
    search_query = st.text_input("Что ищем?", placeholder="Например: люблю, море, пицца")
    
    if search_query:
        # Поиск по тексту
        results = df[df["text"].str.contains(search_query, case=False, na=False)]
        
        st.success(f"Найдено сообщений: **{len(results)}**")
        
        if len(results) > 0:
            # График частоты упоминаний
            res_daily = results.groupby(results["date"].dt.date).size().reset_index(name='count')
            fig_search = px.bar(res_daily, x='date', y='count', color_discrete_sequence=['#FF69B4'])
            st.plotly_chart(fig_search, width='stretch')
            
            st.markdown("##### Последние находки:")
            
            # --- ИЗМЕНЕНИЕ: ПЕРЕВОРАЧИВАЕМ РЕЗУЛЬТАТЫ ---
            # .iloc[::-1] разворачивает DataFrame задом наперед (сначала новые)
            newest_results = results.iloc[::-1]
            
            # Выводим первые 5 из ПЕРЕВЕРНУТОГО списка
            for i in range(min(20, len(newest_results))):
                msg = newest_results.iloc[i]
                # Добавил год в дату, чтобы было понятнее, когда это было
                st.markdown(f"**{msg['date'].strftime('%d.%m.%Y')} {msg['from']}:** {msg['text']}")
profiler.checkpoint("Отрисовка поиска завершена")
# ================== ТАБ 7: СТИКЕРЫ (GROUPBY ОПТИМИЗАЦИЯ) ==================
# ================== ТАБ 7: СТИКЕРЫ (INSTANT RENDER) ==================
with tabs[6]:
    st.subheader("🎭 Любимые стикеры")
    
    # Распаковываем данные (обратите внимание на sticker_contexts)
    daily, user_counts, stickers_df_raw, day_counts, hm_data, sticker_contexts, *rest = get_charts_data(df)

    if not stickers_df_raw.empty:
        # --- ФИЛЬТР ---
        f_col1, f_col2 = st.columns([1, 3])
        with f_col1:
            filter_options = ["Все вместе"] + user_counts.index.tolist()
            sticker_author = st.radio("Чьи стикеры смотрим?", filter_options, index=0)

        # Фильтрация только для подсчета топа (это быстро)
        if sticker_author != "Все вместе":
            st_df = stickers_df_raw[stickers_df_raw["from"] == sticker_author]
        else:
            st_df = stickers_df_raw

        popular_files = st_df["file"].value_counts()
        # ТРЕБОВАНИЕ: > 10 раз
        popular_files = popular_files[popular_files > 10] 
        
        if popular_files.empty:
            st.info(f"Нет стикеров с частотой > 10.")
        else:
            cols = st.columns(3)
            
            for idx, (file_path, count) in enumerate(popular_files.items()):
                col = cols[idx % 3]
                with col:
                    with st.container(border=True):
                        try:
                            # Медиа
                            if os.path.exists(file_path):
                                if file_path.endswith(".webm"):
                                    st.video(file_path, autoplay=True, loop=True, muted=True, start_time=0)
                                else:
                                    st.image(file_path)
                            else:
                                if os.path.exists(os.path.basename(file_path)):
                                    st.image(os.path.basename(file_path))
                                else:
                                    st.markdown("🖼️ *файл не найден*")
                        except: pass
                        
                        rank_emoji = "🥇 " if idx==0 else "🥈 " if idx==1 else "🥉 " if idx==2 else ""
                        st.markdown(f"<h4 style='text-align:center; color: #FF69B4; margin:5px;'>{rank_emoji}{count}</h4>", unsafe_allow_html=True)
                        
                        # --- КОНТЕКСТ (МГНОВЕННЫЙ) ---
                        # Берем готовый список из кэша
                        raw_contexts = []
                        
                        if sticker_author == "Все вместе":
                            # Берем из словаря 'all'
                            raw_contexts = sticker_contexts.get("all", {}).get(file_path, [])
                        else:
                            # Берем из словаря 'auth' по ключу (Файл, Автор)
                            raw_contexts = sticker_contexts.get("auth", {}).get((file_path, sticker_author), [])
                        
                        # Фильтрация коротких сообщений (быстрая операция в памяти)
                        valid_contexts = [str(c) for c in raw_contexts if len(str(c)) > 2]

                        if valid_contexts:
                            st.markdown("<div style='font-size:0.8em; color:gray; margin-top:5px;'>Обычно в ответ на:</div>", unsafe_allow_html=True)
                            
                            # ТРЕБОВАНИЕ: Топ-5 контекстов
                            most_common = Counter(valid_contexts).most_common(5)
                            
                            for ctx, freq in most_common:
                                clean_ctx = ctx
                                st.markdown(f"""
                                <div style='background:#f0f2f6; padding:4px; border-radius:4px; font-size:0.8em; margin-bottom:2px; border-left: 3px solid #FFB6C1;'>
                                    📩 {clean_ctx} <span style='color:#aaa;'>({freq})</span>
                                </div>
                                """, unsafe_allow_html=True)
    else:
        st.warning("Стикеры не найдены.")
        
    profiler.checkpoint("Отрисовка стикеров завершена")

# ================== ТАБ 8: СЛОВА ==================
with tabs[7]:
    Create_word_Cloud()
    profiler.checkpoint("Отрисовка облака слов завершена")
    st.markdown("#### 🥈 Частые фразы")
    ngrams = get_ngrams(df["text"], 2)
    profiler.checkpoint("Отрисовка нграм завершена")
    cols_ng = st.columns(5)
    for i, (phrase, count) in enumerate(ngrams[:5]):
        cols_ng[i].metric(phrase.capitalize(), count)

# ================== ТАБ 9: ВИКТОРИНА ==================
with tabs[8]:
    render_quiz_tab(df, selected)
profiler.checkpoint("Отрисовка вкладок завершена")
# Футер
st.markdown("---")
st.markdown("<div style='text-align: center; color: #aaa; font-size: 14px;'>Создано с ❤️ навсегда</div>", unsafe_allow_html=True)
profiler.finish()