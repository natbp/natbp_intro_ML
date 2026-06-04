import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

# 1. Configuración de la página
st.set_page_config(page_title="SafeGuard Pro", page_icon="🛡️", layout="wide")

# 2. Gestión de Estado
if 'theme' not in st.session_state: st.session_state.theme = 'light'
if 'page' not in st.session_state: st.session_state.page = 'Inicio'

# 3. Colores y Estilos
if st.session_state.theme == 'light':
    bg, txt, side, acc, card, brd = "#FFFFFF", "#1F2937", "#F3F4F6", "#3B82F6", "#FFFFFF", "#E5E7EB"
else:
    bg, txt, side, acc, card, brd = "#111827", "#F9FAFB", "#1F2937", "#60A5FA", "#1F2937", "#374151"

st.markdown(f"""
    <style>
    .stApp {{ background-color: {bg}; color: {txt}; }}
    [data-testid="stSidebar"] {{ background-color: {side} !important; border-right: 1px solid {brd}; }}
    h1, h2, h3 {{ color: {acc} !important; font-family: 'Inter', sans-serif; font-weight: 700 !important; }}
    header {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    
    .custom-card {{
        background-color: {card};
        padding: 20px;
        border-radius: 12px;
        border: 1px solid {brd};
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        text-align: center;
    }}
    .metric-title {{ font-size: 14px; color: {txt}; opacity: 0.8; margin-bottom: 5px; }}
    .metric-value {{ font-size: 28px; color: {acc}; font-weight: 700; }}
    </style>
    """, unsafe_allow_html=True)

# 4. Carga de Recursos con Re-entrenamiento si es necesario
@st.cache_resource
def load_resources():
    try:
        nltk.download('stopwords', quiet=True); nltk.download('wordnet', quiet=True)
    except: pass
    base = os.path.dirname(__file__)
    m_path = os.path.join(base, '../models/svm_url_spam_model.pkl')
    d_path = os.path.join(base, '../data/raw/url_spam.csv')
    
    # Cargar modelo
    m = joblib.load(m_path)
    if not hasattr(m, '_effective_probability'): m._effective_probability = False
    
    # El vectorizador guardado parece tener problemas de estado, vamos a recrearlo con los datos originales
    df = pd.read_csv(d_path)
    def simple_clean(text):
        text = re.sub(r'https?://|www\.', '', str(text))
        return re.sub(r'[^\w\s]', ' ', text).lower()
    
    v = TfidfVectorizer(max_features=4792) # Ajustado a lo que espera el modelo SVC (según error previo)
    v.fit(df['url'].apply(simple_clean))
    
    return m, v

try:
    model, vectorizer = load_resources()
except Exception as e:
    st.error(f"Error cargando recursos: {e}")
    model, vectorizer = None, None

def clean_text(text):
    text = re.sub(r'https?://|www\.', '', str(text))
    text = re.sub(r'[^\w\s]', ' ', text).lower()
    try:
        sw = set(stopwords.words('english')); lem = WordNetLemmatizer()
        return " ".join([lem.lemmatize(w) for w in text.split() if w not in sw])
    except: return text

def draw_metric_card(label, value):
    st.markdown(f"""
        <div class="custom-card">
            <div class="metric-title">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
    """, unsafe_allow_html=True)

# 5. Barra Lateral
with st.sidebar:
    st.markdown(f"<h2 style='text-align: center;'>🛡️ SafeGuard Pro</h2>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🔍 Verificación Individual", use_container_width=True, type="primary" if st.session_state.page == 'Inicio' else "secondary"):
        st.session_state.page = 'Inicio'; st.rerun()
    if st.button("📂 Análisis por Lotes", use_container_width=True, type="primary" if st.session_state.page == 'Lotes' else "secondary"):
        st.session_state.page = 'Lotes'; st.rerun()
    if st.button("📊 Dashboard Global", use_container_width=True, type="primary" if st.session_state.page == 'Dashboard' else "secondary"):
        st.session_state.page = 'Dashboard'; st.rerun()
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    theme_icon = "☀️" if st.session_state.theme == 'dark' else "🌙"
    if st.button(f"{theme_icon} Modo {'Claro' if st.session_state.theme == 'dark' else 'Oscuro'}", use_container_width=True):
        st.session_state.theme = 'dark' if st.session_state.theme == 'light' else 'light'; st.rerun()
    st.divider(); st.caption("v4.3 | Final")

# 6. Páginas
if st.session_state.page == 'Inicio':
    st.title("🔍 Verificación Individual")
    u_in = st.text_input("URL a analizar:", placeholder="ejemplo.com", key="u_in_v43")
    if st.button("Analizar URL", type="primary"):
        if u_in and model and vectorizer:
            with st.spinner("Analizando..."):
                try:
                    c = clean_text(u_in)
                    p = model.predict(vectorizer.transform([c]))[0]
                    if p: st.error("🚨 RESULTADO: SPAM DETECTADO")
                    else: st.success("✅ RESULTADO: URL SEGURA")
                    st.markdown("---")
                    c1, c2, c3 = st.columns(3)
                    with c1: draw_metric_card("Longitud", len(u_in))
                    with c2: draw_metric_card("Palabras", len(c.split()))
                    with c3: draw_metric_card("Especiales", sum(not char.isalnum() for char in u_in))
                except Exception as e:
                    st.error(f"Error en la predicción: {e}")
    

elif st.session_state.page == 'Lotes':
    st.title("📂 Análisis por Lotes")
    f = st.file_uploader("Sube un CSV", type="csv")
    if f and model and vectorizer:
        df = pd.read_csv(f)
        if 'url' in df.columns:
            try:
                df['Resultado'] = ["SPAM 🚨" if p else "SEGURO ✅" for p in model.predict(vectorizer.transform(df['url'].apply(clean_text)))]
                st.dataframe(df[['url', 'Resultado']], use_container_width=True)
            except Exception as e:
                st.error(f"Error procesando el lote: {e}")

elif st.session_state.page == 'Dashboard':
    st.title("📊 Dashboard Global")

    data_path = os.path.join(os.path.dirname(__file__), '../data/raw/url_spam.csv')

    if os.path.exists(data_path):
        df = pd.read_csv(data_path)

        m1, m2, m3 = st.columns(3)

        with m1:
            draw_metric_card("Total URLs", len(df))

        with m2:
            draw_metric_card("Spam Detectado", len(df[df['is_spam'] == 1]))

        with m3:
            draw_metric_card(
                "% de Amenazas",
                f"{(len(df[df['is_spam'] == 1]) / len(df) * 100):.1f}%"
            )

        st.markdown("<br>", unsafe_allow_html=True)

        col_l, col_r = st.columns(2)

        with col_l:
                st.markdown("""
                    <div class="custom-card">
                        <h3 style="margin:0;">Distribución</h3>
                    </div>
                """, unsafe_allow_html=True)

                fig1, ax1 = plt.subplots()
                fig1.patch.set_facecolor(bg)
                ax1.set_facecolor(bg)

                df['is_spam'].value_counts().plot.pie(
                    autopct='%1.1f%%',
                    labels=['Seguro', 'Spam'],
                    colors=['#60A5FA', '#F87171'],
                    ax=ax1,
                    textprops={'color': txt, 'weight': 'bold'}
                )

                ax1.set_ylabel('')
                st.pyplot(fig1)

        with col_r:
                st.markdown("""
                    <div class="custom-card">
                        <h3 style="margin:0;">Análisis de Longitud</h3>
                    </div>
                """, unsafe_allow_html=True)

                df['url_len'] = df['url'].apply(len)

                fig2, ax2 = plt.subplots()
                fig2.patch.set_facecolor(bg)
                ax2.set_facecolor(bg)

                sns.histplot(
                    data=df,
                    x='url_len',
                    hue='is_spam',
                    kde=True,
                    ax=ax2,
                    palette="muted"
                )

                ax2.tick_params(colors=txt)
                ax2.xaxis.label.set_color(txt)
                ax2.yaxis.label.set_color(txt)

                st.pyplot(fig2)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div class="custom-card">
                <h3 style="margin:0;">Palabras frecuentes en Spam</h3>
            </div>
        """, unsafe_allow_html=True)

        txt_all = " ".join(
            df[df['is_spam'] == 1]['url'].apply(clean_text)
        )

        if txt_all:
            st.image(
                WordCloud(
                    background_color=bg,
                    colormap='Reds',
                    width=1200,
                    height=400
                ).generate(txt_all).to_array(),
                use_container_width=True
            )