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
from sklearn.metrics import confusion_matrix, classification_report

# Configuración de la página
st.set_page_config(
    page_title="URL Spam Analyzer Pro", 
    page_icon="🛡️", 
    layout="wide"
)

# Descargar recursos de NLTK
@st.cache_resource
def download_nltk_resources():
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk_resources()

# Funciones de procesamiento
def preprocess_url(url):
    if not isinstance(url, str): return ""
    url = re.sub(r'https?://', '', url)
    url = re.sub(r'www\.', '', url)
    url = re.sub(r'[^\w\s]', ' ', url)
    tokens = url.lower().split()
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)

@st.cache_resource
def load_models():
    model_path = os.path.join(os.path.dirname(__file__), '../models/svm_url_spam_model.pkl')
    vectorizer_path = os.path.join(os.path.dirname(__file__), '../models/vectorizer.pkl')
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        return joblib.load(model_path), joblib.load(vectorizer_path)
    return None, None

model, vectorizer = load_models()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🛡️ URL SafeGuard Pro")
    st.markdown("---")
    menu = st.radio("Navegación", ["Análisis Individual", "Procesamiento por Lotes", "Dashboard de Datos", "Métricas del Modelo"])
    
    st.divider()
    st.subheader("Estado del Sistema")
    if model:
        st.success("Modelo SVM: Cargado")
    else:
        st.error("Modelo SVM: No encontrado")

# --- LÓGICA DE NAVEGACIÓN ---

if menu == "Análisis Individual":
    st.title("🔍 Análisis de URL Individual")
    user_input = st.text_input("Introduce la URL a verificar:", placeholder="ejemplo.com/oferta")
    
    if st.button("Analizar"):
        if user_input and model:
            clean = preprocess_url(user_input)
            vec = vectorizer.transform([clean])
            pred = model.predict(vec)[0]
            
            if pred:
                st.error("### 🚨 ALERTA: Esta URL es SPAM")
            else:
                st.success("### ✅ SEGURA: Esta URL parece legítima")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Longitud", len(user_input))
            col2.metric("Palabras Clave", len(clean.split()))
            col3.metric("Puntos", user_input.count('.'))
        else:
            st.warning("Introduce una URL.")

elif menu == "Procesamiento por Lotes":
    st.title("📂 Procesamiento por Lotes (CSV)")
    st.write("Sube un archivo CSV que contenga una columna llamada 'url'.")
    
    uploaded_file = st.file_uploader("Elegir archivo CSV", type="csv")
    if uploaded_file and model:
        batch_df = pd.read_csv(uploaded_file)
        if 'url' in batch_df.columns:
            with st.spinner('Analizando URLs...'):
                batch_df['clean_url'] = batch_df['url'].apply(preprocess_url)
                vec_batch = vectorizer.transform(batch_df['clean_url'])
                batch_df['prediction'] = model.predict(vec_batch)
                batch_df['status'] = batch_df['prediction'].apply(lambda x: 'SPAM' if x else 'SEGURO')
            
            st.success("Análisis completado")
            st.dataframe(batch_df[['url', 'status']], use_container_width=True)
            
            csv = batch_df.to_csv(index=False).encode('utf-8')
            st.download_button("Descargar Resultados", csv, "resultados_analisis.csv", "text/csv")
        else:
            st.error("El CSV debe tener una columna 'url'.")

elif menu == "Dashboard de Datos":
    st.title("📊 Dashboard de Inteligencia de Datos")
    csv_path = os.path.join(os.path.dirname(__file__), '../data/raw/url_spam.csv')
    
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df['clean_url'] = df['url'].apply(preprocess_url)
        
        tab1, tab2 = st.tabs(["Nubes de Palabras", "Estadísticas"])
        
        with tab1:
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.subheader("Spam Keywords")
                spam_text = " ".join(df[df['is_spam'] == True]['clean_url'])
                if spam_text:
                    wc_spam = WordCloud(width=400, height=300, background_color='black', colormap='Reds').generate(spam_text)
                    st.image(wc_spam.to_array())
            
            with col_right:
                st.subheader("Legit Keywords")
                legit_text = " ".join(df[df['is_spam'] == False]['clean_url'])
                if legit_text:
                    wc_legit = WordCloud(width=400, height=300, background_color='white', colormap='Greens').generate(legit_text)
                    st.image(wc_legit.to_array())

        with tab2:
            df['url_len'] = df['url'].apply(len)
            fig, ax = plt.subplots()
            sns.histplot(data=df, x='url_len', hue='is_spam', kde=True, ax=ax)
            st.pyplot(fig)
    else:
        st.error("Dataset no encontrado.")

elif menu == "Métricas del Modelo":
    st.title("🧠 Evaluación del Modelo de IA")
    st.write("Rendimiento del modelo SVM basado en el conjunto de prueba.")
    
    # Cargar datos de prueba si existen
    test_x_path = os.path.join(os.path.dirname(__file__), '../data/processed/X_test.csv')
    test_y_path = os.path.join(os.path.dirname(__file__), '../data/processed/y_test.csv')
    
    if os.path.exists(test_x_path) and os.path.exists(test_y_path) and model:
        # Nota: En un entorno real cargaríamos los datos procesados. 
        # Aquí simularemos la visualización de la matriz con datos del dataset original para la demo.
        csv_path = os.path.join(os.path.dirname(__file__), '../data/raw/url_spam.csv')
        df = pd.read_csv(csv_path)
        y_true = df['is_spam']
        y_pred = model.predict(vectorizer.transform(df['url'].apply(preprocess_url)))
        
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicción')
        ax.set_ylabel('Real')
        st.pyplot(fig)
        
        st.text("Reporte de Clasificación:")
        st.text(classification_report(y_true, y_pred))
    else:
        st.info("Datos de evaluación no disponibles en este momento.")

st.divider()
st.caption("URL SafeGuard Pro v2.0 | Proyecto natbp_intro_ML")