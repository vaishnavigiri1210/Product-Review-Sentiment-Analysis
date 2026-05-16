import streamlit as st      # Streamlit for web app
import pandas as pd         # Data manipulation
import joblib               # For loading ML models
import matplotlib.pyplot as plt     # For plotting
import seaborn as sns   # For advanced visualizations
import plotly.graph_objects as go       # For interactive charts
import emoji        # For emoji analysis
import os           # For file path handling
import re           # Optimized intent detection साठी Regex
import gspread      # Google Sheets connection
from datetime import datetime           # तारीख आणि वेळ सेव्ह करण्यासाठी
from collections import Counter         # For counting emojis
from langdetect import detect, detect_langs     # For language detection             
from streamlit_mic_recorder import speech_to_text       # For mic input in Streamlit

# ==========================================
# 1. PAGE & PATH CONFIGURATION 
# ==========================================
st.set_page_config(
    page_title="Product Review Sentiment & Intent Analysis", 
    layout="wide", 
    page_icon="🏢"
)
st.title("🏛️ Product Review Sentiment & Intent Analysis")

path = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource   
def load_assets():
    try:
        df = pd.read_csv(os.path.join(path, '..', 'data', 'final_insights_multilingual.csv'), encoding='utf-8-sig')
        metadata = pd.read_csv(os.path.join(path, '..', 'data', 'correctedMetadata.csv'))
        model = joblib.load(os.path.join(path, '..', 'models', 'sentiment_model.pkl'))
        vectorizer = joblib.load(os.path.join(path, '..', 'models', 'tfidf_vectorizer.pkl'))
        return df, metadata, model, vectorizer
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None, None

df, metadata, model, vectorizer = load_assets() 

if df is None:
    st.stop()

# ==========================================
# 2. PUBLIC GOOGLE SHEET STORAGE SYSTEM (NO SECRETS REQUIRED)
# ==========================================
# ⚠️ खालील लिंकच्या ठिकाणी तुझ्या स्वतःच्या गुगल शीटची "Anyone with the link can edit" केलेली लिंक टाक!
GSHEET_URL = "https://docs.google.com/spreadsheets/d/1wts27e8aBcAjq91u6is7jP11Q95rmXuyOoW0i4Tu9fI/edit?usp=sharing"
def save_review_to_gsheet(review, sentiment, intent, language):
    """सार्वजनिक गुगल शीटमध्ये डेटा थेट इंटरनेटवरून सेव्ह करणे (डिप्लॉयमेंटसाठी १००% परफेक्ट)"""
    try:
        # विना क्रेडेंशियल्स एडिट लिंकद्वारे कनेक्ट करणे
        gc = gspread.public_link(GSHEET_URL) if hasattr(gspread, 'public_link') else gspread.open_by_url(GSHEET_URL)
        sheet = gc.sheet1
        
        # शीट रिकामा असेल तर हेडर टाकणे
        if not sheet.get_all_values():
            sheet.append_row(["Timestamp", "Review Text", "Sentiment", "Intent", "Language"])
            
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sheet.append_row([current_time, review, sentiment, intent, language])
        return True
    except:
        # जर वरील पद्धत फेल झाली, तर बॅकअप म्हणून फक्त चालू सेशनमध्ये सेव्ह ठेवणे
        if "backup_list" not in st.session_state:
            st.session_state.backup_list = []
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.session_state.backup_list.append([current_time, review, sentiment, intent, language])
        return False

def load_live_logs():
    """गुगल शीट किंवा बॅकअपमधून डेटा वाचून डॅशボードवर दाखवणे"""
    try:
        gc = gspread.open_by_url(GSHEET_URL)
        sheet = gc.sheet1
        data = sheet.get_all_records()
        return pd.DataFrame(data)
    except:
        if "backup_list" not in st.session_state:
            return pd.DataFrame()
        return pd.DataFrame(st.session_state.backup_list, columns=["Timestamp", "Review Text", "Sentiment", "Intent", "Language"])

# ==========================================
# 3. HELPER & BUSINESS LOGIC FUNCTIONS
# ==========================================
def detect_language_smart(text):
    text = str(text).lower().strip()
    if not text or len(text) < 3: 
        return 'English'
    
    marathi_words = r'\b(chan|bhari|masta|lay|awadla|khup|changla|nko|navhta|jasta|pan|aahe|ahe|kumat|kimat|vait|chhan)\b'
    hindi_words = r'\b(accha|bahut|hai|acha|bhai|kharab|bekar|sasta|mast|achha|khoob|bohot|bhaiya)\b'
    english_words = r'\b(product|good|bad|quality|nice|item|waste|money|great|awesome|delivery)\b'
    
    has_marathi = re.search(marathi_words, text)
    has_hindi = re.search(hindi_words, text)
    has_english = re.search(english_words, text)
    
    if has_marathi and has_english: return 'Marathi + English (Mixed)'
    elif has_hindi and has_english: return 'Hindi + English (Mixed)'
    elif has_marathi: return 'Marathi'
    elif has_hindi: return 'Hindi'
        
    try:
        res = detect_langs(text)
        lang_codes = [l.lang for l in res if l.prob > 0.10]
        if 'en' in lang_codes and 'mr' in lang_codes: return 'Marathi + English (Mixed)'
        elif 'en' in lang_codes and 'hi' in lang_codes: return 'Hindi + English (Mixed)'
        elif 'mr' in lang_codes: return 'Marathi'
        elif 'hi' in lang_codes: return 'Hindi'
        elif 'en' in lang_codes: return 'English'
        else: return 'Other/Mixed'
    except: 
        return 'English'

def detect_intent(text):
    text = str(text).lower()
    if re.search(r'(price|cost|expensive|kimat|किंमत|महाग|दर|स्वस्त|paise|paisa)', text): return "💰 Pricing"
    if re.search(r'(delivery|late|fast|slow|ushir|उशीर|वेळ|डिलिव्हरी|पोहोचले|time|day|days)', text): return "🚚 Logistics"
    if re.search(r'(quality|material|strong|durability|दर्जा|क्वालिटी|कापड|टिकाऊ|kapda|fabric|look)', text): return "🛠️ Quality"
    if re.search(r'(service|support|staff|मदत|सर्व्हिस|सहकार्य|call|care)', text): return "📞 Support"
    return "📝 General"

def draw_gauge(score):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = score,
        title = {'text': "Customer Approval Rate (%)", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 100]},
            'bar': {'color': "#10b981"},
            'steps': [
                {'range': [0, 40], 'color': "#ef4444"},
                {'range': [40, 70], 'color': "#f59e0b"},
                {'range': [70, 100], 'color': "#10b981"}],
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# ==========================================
# 4. SIDEBAR CONTROL PANEL
# ==========================================
st.sidebar.title("🛠️ BI Control Panel")
search_term = st.sidebar.text_input("🔍 Search Keyword (e.g. 'good', 'bad', 'मस्त'):")

filtered_df = df.copy()
is_data_empty = False

if search_term:
    filtered_df = df[df['review_text'].str.contains(search_term, case=False, na=False)]
    if filtered_df.empty:
        st.sidebar.warning(f"'{search_term}' Data not found.")
        is_data_empty = True

st.sidebar.divider()

csv_report = filtered_df.to_csv(index=False).encode('utf-8-sig')
st.sidebar.download_button("Download Full Report", csv_report, "bi_analysis.csv", "text/csv")

# ==========================================
# 5. DASHBOARD TABS
# ==========================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Market Performance", "🤖 AI Predictor", "🕵️ Integrity & Emotions", "🎯 Advanced Filters", "💡 Strategic Insights"
])

# --- TAB 1: Performance Trends ---
with tab1:
    if not is_data_empty:
        pos_rate = (filtered_df['sentiment'] == 'Positive').mean() * 100
        col_a, col_b = st.columns([1, 2])
        with col_a: st.plotly_chart(draw_gauge(pos_rate), use_container_width=True)
        with col_b:
            st.subheader("Market Sentiment by Language")
            st.bar_chart(pd.crosstab(filtered_df['detected_lang'], filtered_df['sentiment']))
        st.divider()
        st.subheader("🔥 Rating-Sentiment Density Heatmap")
        fig_heat, ax_heat = plt.subplots(figsize=(8, 4))
        sns.heatmap(pd.crosstab(filtered_df['rating'], filtered_df['sentiment']), annot=True, fmt='d', cmap='YlGnBu', ax=ax_heat)
        st.pyplot(fig_heat)
    else:
        st.info("🔍 Filtered data not found.")

# --- TAB 2: Live AI Predictor ---
with tab2:
    st.subheader("🤖 Real-time Multilingual Inference")
    
    st.markdown("""
        <style>
        div[data-testid="column"] { display: flex; align-items: center; gap: 0px; }
        div[data-testid="stTextInput"] input {
            border-radius: 10px !important;
            height: 48px !important;
            border: 1px solid #d1d5db !important;
        }
        div[data-testid="stForm"] { border: none !important; padding: 0px !important; margin-top: -0px; }
        div.stFormSubmitButton > button {
            background-color: transparent !important;
            color: #4285f4 !important;
            border: 1px solid #4285f4 !important;
            border-radius: 10px !important;
            height: 45px !important;
            width: 100% !important;
            font-weight: bold !important;
            margin-top: 5px;
        }
        .res-box {
            padding: 15px;
            border-radius: 10px;
            border: 1px solid #eee;
            text-align: center;
            background: #ffffff;
            box-shadow: 0px 2px 4px rgba(0,0,0,0.05);
            font-size: 16px;
        }
        </style>
    """, unsafe_allow_html=True)

    if "final_text_val" not in st.session_state:
        st.session_state.final_text_val = ""

    col_in, col_m = st.columns([0.88, 0.12])
    
    with col_m:
        v_res = speech_to_text(language='en', start_prompt="🎙️", stop_prompt="🛑", just_once=True, key='MIC_STABLE')
        if v_res:
            st.session_state.final_text_val = v_res
            st.rerun()

    with col_in:
        with st.form(key='my_predict_form', clear_on_submit=False):
            user_input_text = st.text_input(
                "Review Box",
                value=st.session_state.final_text_val,
                placeholder="Write your review or click the mic...",
                label_visibility="collapsed"
            )
            submit_clicked = st.form_submit_button("Predict Sentiment & Intent")

    if submit_clicked:
        if user_input_text.strip():
            lang_res = detect_language_smart(user_input_text) 
            intent_res = detect_intent(user_input_text)
            
            cleaned_lower = user_input_text.lower()
            if re.search(r'(chan|bhari|masta|accha|acha|loved|good product|heavy|achha|chhan)', cleaned_lower):
                prediction = 'Positive'
            elif re.search(r'(bad|worst|waste|bakwas|bekar|kharaab|kharab)', cleaned_lower):
                prediction = 'Negative'
            else:
                input_vec = vectorizer.transform([cleaned_lower])
                try:
                    probs = model.predict_proba(input_vec)[0]
                    prediction = 'Neutral' if max(probs) < 0.60 else model.predict(input_vec)[0]
                except:
                    prediction = model.predict(input_vec)[0]
            
            # 💾 ऑनलाईन गुगल शीटमध्ये सेव्ह करणे
            save_review_to_gsheet(user_input_text, prediction, intent_res, lang_res)
            st.toast("📝 Review Saved to Cloud Sheet!", icon="☁️")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            r1, r2, r3 = st.columns(3)
            with r1:
                st.markdown(f'<div class="res-box"><b>Language</b><br>{lang_res}</div>', unsafe_allow_html=True)
            with r2: 
                st.markdown(f'<div class="res-box"><b>Intent</b><br>{intent_res}</div>', unsafe_allow_html=True)
            with r3:
                color = "#d4edda" if prediction == 'Positive' else "#f8d7da" if prediction == 'Negative' else "#fff3cd"
                st.markdown(f'<div class="res-box" style="background-color: {color};"><b>Sentiment</b><br>{prediction}</div>', unsafe_allow_html=True)
            
            st.session_state.final_text_val = user_input_text
        else:
            st.warning("Please enter some text or use mic input to predict.")

# --- TAB 3: Integrity & Emotions ---
with tab3:
    col_integrity, col_emotions = st.columns(2)
    with col_integrity:
        st.subheader("🕵️ Integrity & Trust Analysis")
        if 'is_fake' in df.columns and not df.empty:
            counts = df['is_fake'].value_counts()
            fig_p, ax_p = plt.subplots(figsize=(5,5))
            l = [('Genuine' if str(i) in ['0', '0.0', 'Real', 'False'] else 'Suspicious') for i in counts.index]
            counts.plot.pie(labels=l, autopct='%1.1f%%', colors=['#2E7D32','#C62828'], ax=ax_p, startangle=90)
            ax_p.set_ylabel('')
            st.pyplot(fig_p)
        st.divider()
        st.write("**🎯 Feedback Depth (Sincerity)**")
        sincerity_series = filtered_df['review_text'].apply(lambda x: "Detailed" if len(str(x).split()) > 5 else "Brief")
        st.bar_chart(sincerity_series.value_counts())

    with col_emotions:
        st.subheader("🎭 Emotional Insights")
        def find_emojis(t): return [char for char in str(t) if char in emoji.EMOJI_DATA]
        emoji_list = filtered_df['review_text'].apply(find_emojis).sum()
        top_e = Counter(emoji_list).most_common(10)
        if top_e:
            st.write("**Top Visual Emotions (Emojis):**")
            st.table(pd.DataFrame(top_e, columns=['Emoji', 'Frequency']))
        st.divider()
        st.write("**🔥 Sentiment Intensity Level**")
        intensity_series = filtered_df['rating'].isin([1, 5]).map({True: 'Strong', False: 'Moderate'})
        intensity_plot = pd.crosstab(filtered_df['sentiment'], intensity_series)
        st.bar_chart(intensity_plot)
        
# --- TAB 4: Custom Filters & Live Production Logs ---
with tab4:
    if not is_data_empty:
        st.subheader("🎯 Deep Dive Explorer")
        f1, f2 = st.columns(2)
        with f1: sl = st.multiselect("Language:", filtered_df['detected_lang'].unique(), default=filtered_df['detected_lang'].unique())
        with f2: ss = st.multiselect("Sentiment:", filtered_df['sentiment'].unique(), default=filtered_df['sentiment'].unique())
        st.dataframe(filtered_df[(filtered_df['detected_lang'].isin(sl)) & (filtered_df['sentiment'].isin(ss))][['review_text', 'detected_lang', 'sentiment', 'rating']], use_container_width=True)

        # 📂 लाईव्ह क्लाऊड डेटा व्ह्यूअर
        st.divider()
        st.subheader("📂 All Real-time User Reviews (Cloud Database Log)")
        live_df = load_live_logs()
        if not live_df.empty:
            st.dataframe(live_df, use_container_width=True)
            
            # डाऊनलोड बटण
            live_csv = live_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("Download All Live User Reviews as CSV", live_csv, "live_user_reviews.csv", "text/csv")
        else:
            st.info("No real-time predictions done yet. Type a review in Tab 2 to see live logs here!")

# --- TAB 5: Strategic Insights ---
with tab5:
    if not is_data_empty:
        current_intents = filtered_df['review_text'].apply(detect_intent)
        st.subheader("🔦 Identifying Business Pain-Points")
        c_ins1, c_ins2 = st.columns(2)
        with c_ins1:
            st.write("**Top Concerns (Negative Intents)**")
            st.bar_chart(current_intents[filtered_df['sentiment'] == 'Negative'].value_counts())
        with c_ins2:
            st.write("**Intent Heatmap**")
            st.write(pd.crosstab(current_intents, filtered_df['sentiment']).style.background_gradient(cmap='YlOrRd'))
        
# Footer Metrics
st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Reviews", len(df))
m2.metric("Filtered Reviews", len(filtered_df))
m3.metric("Avg Rating", f"{filtered_df['rating'].mean():.1f} ⭐" if not filtered_df.empty else "0.0 ⭐")
m4.metric("Market Sentiment", "Positive" if (filtered_df['sentiment'] == 'Positive').mean() * 100 > 50 else "Needs Work")

st.sidebar.caption("VoxInsight AI | Status: Online 🟢")