import streamlit as st       # Streamlit for web app
import pandas as pd          # Data manipulation
import joblib                # For loading ML models
import matplotlib.pyplot as plt     # For plotting
import seaborn as sns    # For advanced visualizations
import plotly.graph_objects as go       # For interactive charts here gauge chart
import emoji         # For emoji analysis
import os           # For file path handling
import re           # Optimized intent, language detection and keyword matching for Regular Expressions
import requests     # sending data to and fetching from API-based Google Sheet system over the internet
from datetime import datetime           # current data and time handling
from collections import Counter         # For counting emojis
from langdetect import detect_langs     # For language detection 
from streamlit_mic_recorder import speech_to_text       # For mic input in Streamlit

# 1. PAGE & PATH CONFIGURATION 
st.set_page_config(
    page_title="Product Review Sentiment & Intent Analysis", 
    layout="wide", 
    page_icon="🏢"
)
st.title("🏛️ Product Review Sentiment & Intent Analysis")

path = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource   # run this function only once and remember the result
def load_assets():
    try:
        df = pd.read_csv(os.path.join(path, '..', 'data', 'final_insights_multilingual.csv'), encoding='utf-8-sig')
        model = joblib.load(os.path.join(path, '..', 'models', 'sentiment_model.pkl'))
        vectorizer = joblib.load(os.path.join(path, '..', 'models', 'tfidf_vectorizer.pkl'))
        return df, model, vectorizer
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None, None

df, model, vectorizer = load_assets() 

if df is None:
    st.stop()   # stops the execution of the app if data loading fails

# 2. API-BASED GOOGLE SHEET SYSTEM
API_URL = "https://script.google.com/macros/s/AKfycbwccWuyEQyiOunRKit9WMvgr8cYsu6k-KqRTbP9s8dmsharWo4fPqTAPiMGBe0xWDJnsQ/exec"

def save_review_to_gsheet(review, sentiment, intent, language):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S") 
    payload = {
        "timestamp": current_time,
        "review": review,
        "sentiment": sentiment,
        "intent": intent,
        "language": language
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=10)
        if response.status_code == 200:
            return True
        raise Exception("API Error")
    except:
        if "backup_list" not in st.session_state:
            st.session_state.backup_list = []
        st.session_state.backup_list.append([current_time, review, sentiment, intent, language])
        return False

def load_live_logs():
    try:
        response = requests.get(API_URL, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data:
                return pd.DataFrame(data)
    except:
        pass
    
    if "backup_list" not in st.session_state or not st.session_state.backup_list:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.backup_list, columns=["Timestamp", "Review Text", "Sentiment", "Intent", "Language"])

# 3. ADVANCED HELPER & BUSINESS LOGIC FUNCTIONS

def detect_language_smart(text):
    raw_text = str(text).strip()
    text_lower = raw_text.lower()
    
    if not text_lower or len(text_lower) < 3: 
        # Check if the short string is purely an emoji
        if all(char in emoji.EMOJI_DATA for char in raw_text):
            return 'Emoji-Only'
        return 'English'
    
    # Check for Devanagari Script (Native Marathi / Hindi characters)
    if re.search(r'[\u0900-\u097F]', raw_text):
        # Specific high-frequency native Marathi markers
        marathi_native = r'(आहे|नाही|छान|मस्त|भारी|खूप|तयार|करून|बघा|नका|काही|केलं|मिळालं|होता)'
        if re.search(marathi_native, raw_text):
            return 'Marathi'
        return 'Hindi'

    # Transliterated (Latin Alphabet) Keywords via Exact Word Boundaries
    marathi_words = r'\b(chan|bhari|masta|lay|awadla|khup|changla|nko|navhta|jasta|pan|aahe|ahe|kumat|kimat|vait|chhan|nakki|k खरेदी)\b'
    hindi_words = r'\b(accha|bahut|hai|acha|bhai|kharab|bekar|sasta|mast|achha|khoob|bohot|bhaiya|nahi|hai|hota|gaya|mil)\b'
    english_words = r'\b(product|good|bad|quality|nice|item|waste|money|great|awesome|delivery|perfect|worst|terrible)\b'
    
    has_marathi = re.search(marathi_words, text_lower)
    has_hindi = re.search(hindi_words, text_lower)
    has_english = re.search(english_words, text_lower)
    
    if has_marathi and has_english: return 'Marathi + English (Mixed)'
    elif has_hindi and has_english: return 'Hindi + English (Mixed)'
    elif has_marathi: return 'Marathi'
    elif has_hindi: return 'Hindi'
    
    try:
        res = detect_langs(text_lower)
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
    if re.search(r'(price|cost|expensive|kimat|किंमत|महाग|दर|स्वस्त|paise|paisa|बर्बादी|waste of money|पैसे वाया)', text): return "💰 Pricing"
    if re.search(r'(delivery|late|fast|slow|ushir|उशीर|वेळ|डिलिव्हरी|पोहोचले|time|day|days|delays|delayed|service|सर्व्हिस)', text): return "🚚 Logistics"
    if re.search(r'(quality|material|strong|durability|दर्जा|क्वालिटी|कापड|टिकाऊ|kapda|fabric|look|broken|तुटलं|damage|damaged|defective)', text): return "🛠️ Quality"
    if re.search(r'(support|staff|मदत|सहकार्य|call|care|customer service|respond)', text): return "📞 Support"
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

# 4. SIDEBAR CONTROL PANEL
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

# 5. DASHBOARD TABS
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Market Performance", "🤖 AI Predictor", "🕵️ Integrity & Emotions", "🎯 Advanced Filters", "💡 Strategic Insights"
])

# TAB 1: Performance Trends
with tab1:
    if not is_data_empty:
        pos_rate = (filtered_df['sentiment'] == 'Positive').mean() * 100
        col_a, col_b = st.columns([1, 2])
        with col_a: 
            st.plotly_chart(draw_gauge(pos_rate), use_container_width=True)
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

# TAB 2: Live AI Predictor
with tab2:
    st.subheader("🤖 Real-time Multilingual Inference")

    if "final_text_val" not in st.session_state:
        st.session_state.final_text_val = ""

    col_in, col_m = st.columns([0.88, 0.12])

    with col_m:
        v_res = speech_to_text(
            language='en-IN',
            start_prompt="🎙️",
            stop_prompt="🛑",
            just_once=True,
            key='MIC_STABLE'
        )
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
            cleaned_lower = user_input_text.lower().strip()

            # --- EMOJI TRACKING SYSTEM ---
            strong_positive_emojis = ["😍", "👍", "❤️", "🔥", "😊", "✨", "😋", "🚀", "💰", "🥰", "👌"]
            strong_negative_emojis = ["😡", "👎", "🤮", "💩", "❌", "🗑️", "😷", "💀", "🚫", "😫", "💢", "😞", "💔"]
            
            extracted_emojis = [char for char in user_input_text if char in emoji.EMOJI_DATA]
            pos_emoji_count = sum(1 for e in extracted_emojis if e in strong_positive_emojis)
            neg_emoji_count = sum(1 for e in extracted_emojis if e in strong_negative_emojis)

            # --- MULTILINGUAL WORD BOUNDARY DICTIONARIES ---
            positive_words = [
                "good","great","excellent","amazing","awesome","love","best","fantastic","perfect","nice",
                "worth","छान","मस्त","भारी","उत्तम","चांगला","आवडलं","अच्छा","शानदार","बेस्ट है",
                "बहुत बढ़िया","पैसे वसूल","worth every rupee","highly recommended"
            ]
       
            negative_words = [
                "bad","worst","poor","terrible","waste","broken","useless","not worth","delay",
                "खराब","वाईट","बेकार","निराश","पैसे वाया","घटिया","पैसे बर्बाद","could be better", 
                "average","damaged","did not respond","poor support"
            ]

            negative_phrases = [
                "not good", "not worth", "poor quality", "waste of money", 
                "very bad", "bad product", "खराब", "वाईट", "बेकार"
            ]

            prediction = None
            
            # First check explicit negative combinations
            if any(phrase in cleaned_lower for phrase in negative_phrases):
                prediction = "Negative"
            else:
                # Count matches using true word boundaries (\b) so parts of words aren't misread
                pos_count = sum(1 for word in positive_words if re.search(r'\b' + re.escape(word) + r'\b', cleaned_lower))
                neg_count = sum(1 for word in negative_words if re.search(r'\b' + re.escape(word) + r'\b', cleaned_lower))

                # Combine word results with raw emoji counts
                pos_count += pos_emoji_count
                neg_count += neg_emoji_count

                if pos_count > neg_count:
                    prediction = "Positive"
                elif neg_count > pos_count:
                    prediction = "Negative"
                elif len(extracted_emojis) > 0 and pos_emoji_count == 0 and neg_emoji_count == 0:
                    prediction = "Neutral"

            # Fallback to Machine Learning Model if rules tie
            if prediction is None:
                # Pad emojis with whitespace so the vectorizer catches them explicitly
                padded_text = "".join(f" {ch} " if ch in emoji.EMOJI_DATA else ch for ch in cleaned_lower)
                input_vec = vectorizer.transform([padded_text])
                
                try:
                    probs = model.predict_proba(input_vec)[0]
                    if max(probs) < 0.55:
                        prediction = "Neutral"
                    else:
                        prediction = model.predict(input_vec)[0]
                except:
                    prediction = model.predict(input_vec)[0]

            # Sync results online
            success = save_review_to_gsheet(user_input_text, prediction, intent_res, lang_res)

            if success:
                st.success("✅ Review Saved Successfully to Cloud Database!")
            else:
                st.warning("⚠️ Saved Locally to Session State Backup (Network/API Timeout)")

            r1, r2 = st.columns(2)
            with r1: st.info(f"Intent Domain: {intent_res}")
            with r2: st.success(f"Model Prediction: {prediction}")
            st.session_state.final_text_val = ""
        else:
            st.warning("Please enter some text or use mic input.")

# TAB 3: Integrity & Emotions
with tab3:
    col_integrity, col_emotions = st.columns(2)
    with col_integrity:
        st.subheader("🕵️ Integrity & Trust Analysis")
        if 'is_fake' in df.columns and not df.empty:
            counts = df['is_fake'].value_counts()
            fig_p, ax_p = plt.subplots(figsize=(5,5))
            l = [('Genuine' if str(i) in ['0', '0.0', 'Real', 'False'] else 'Suspicious') for i in counts.index]
            counts.plot.pie(labels=l, autopct='%1.1f%%', colors = ['#2E7D32'] if len(counts)==1 else ['#2E7D32','#C62828'], ax=ax_p, startangle=90)
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
        
# TAB 4: Custom Filters & Live Production Logs
with tab4:
    if not is_data_empty:
        st.subheader("🎯 Deep Dive Explorer")
        f1, f2 = st.columns(2)
        with f1: sl = st.multiselect("Language:", filtered_df['detected_lang'].unique(), default=filtered_df['detected_lang'].unique())
        with f2: ss = st.multiselect("Sentiment:", filtered_df['sentiment'].unique(), default=filtered_df['sentiment'].unique())
        st.dataframe(filtered_df[(filtered_df['detected_lang'].isin(sl)) & (filtered_df['sentiment'].isin(ss))][['review_text', 'detected_lang', 'sentiment', 'rating']], use_container_width=True)

        st.divider()
        st.subheader("📂 All Real-time User Reviews (Cloud Database Log)")
        live_df = load_live_logs()
        if not live_df.empty:
            st.dataframe(live_df, use_container_width=True)
            live_csv = live_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button("Download All Live User Reviews as CSV", live_csv, "live_user_reviews.csv", "text/csv")
        else:
            st.info("No real-time predictions done yet. Type a review in Tab 2 to see live logs here!")

# TAB 5: Strategic Insights
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
avg_rating = filtered_df['rating'].fillna(0).mean()
m3.metric("Avg Rating", f"{avg_rating:.1f} ⭐" if not filtered_df.empty else "0.0 ⭐")
m4.metric("Market Sentiment", "Positive" if (filtered_df['sentiment'] == 'Positive').mean() * 100 > 50 else "Needs Work")

st.sidebar.caption("Predict, Review, Intent | Status: Online 🟢")