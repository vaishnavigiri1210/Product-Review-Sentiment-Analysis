import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import emoji
import os
from collections import Counter
from langdetect import detect, detect_langs
import streamlit as st
from streamlit_mic_recorder import speech_to_text

# 1. Page Configuration
st.set_page_config(page_title="VoxInsight AI: Enterprise Dashboard", layout="wide", page_icon="🏢")
st.title("🏛️ VoxInsight AI: Multilingual Sentiment & Business Intelligence")

# 2. Asset Loading
path = os.path.dirname(__file__)

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

# --- 3. Helper & Business Logic Functions ---
def detect_language_smart(text):
    text = str(text).strip()
    if not text or len(text) < 3: return 'English'
    try:
        res = detect_langs(text)
        lang_codes = [l.lang for l in res if l.prob > 0.15]
        if 'mr' in lang_codes: return 'Marathi'
        elif 'hi' in lang_codes: return 'Hindi'
        else: return 'English'
    except: return 'English'

def detect_intent(text):
    text = str(text).lower()
    if any(word in text for word in ['price', 'cost', 'expensive', 'किंमत', 'महाग', 'दर', 'स्वस्त']):
        return "💰 Pricing"
    elif any(word in text for word in ['delivery', 'late', 'fast', 'slow', 'उशीर', 'वेळ', 'डिलिव्हरी', 'पोहोचले']):
        return "🚚 Logistics"
    elif any(word in text for word in ['quality', 'material', 'strong', 'durability', 'दर्जा', 'क्वालिटी', 'कापड', 'टिकाऊ']):
        return "🛠️ Quality"
    elif any(word in text for word in ['service', 'support', 'staff', 'मदत', 'सर्व्हिस', 'सहकार्य']):
        return "📞 Support"
    else:
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

# 4. Sidebar Control Panel
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

# 5. Dashboard Tabs
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

# --- TAB 2: Live AI Predictor (The Bulletproof Version with Neat UI) ---
with tab2:
    st.subheader("🤖 Real-time Multilingual Inference")
    
    # advanced CSS for better UI (especially for the mic button and form)
    st.markdown("""
        <style>
        /* column alignment */
        div[data-testid="column"] { 
            display: flex; 
            align-items: center; 
            gap: 0px; 
        }

        /* text input */
        div[data-testid="stTextInput"] input {
            border-radius: 10px !important;
            height: 48px !important;
            border: 1px solid #d1d5db !important;
        }

        /* form border and padding */
        div[data-testid="stForm"] {
            border: none !important;
            padding: 0px !important;
            margin-top: -0px; /* spaceing between mic and form */
        }

        /* Predict Button */
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

        /* result box styling */
        .res-box {
            padding: 12px;
            border-radius: 10px;
            border: 1px solid #eee;
            text-align: center;
            background: #ffffff;
            box-shadow: 0px 2px 4px rgba(0,0,0,0.05);
        }
        </style>
    """, unsafe_allow_html=True)

    # 2. Initialize session state for final text value
    if "final_text_val" not in st.session_state:
        st.session_state.final_text_val = ""

    # 3. Mic Input + Text Input in the same row
    col_in, col_m = st.columns([0.88, 0.12])
    
    with col_m:
        v_res = speech_to_text(
            language='en', 
            start_prompt="🎙️", 
            stop_prompt="🛑", 
            just_once=True, 
            key='MIC_STABLE'
        )
        if v_res:
            st.session_state.final_text_val = v_res
            st.rerun()

    # 4. Form for Text Input (with session state to retain mic input)
    with col_in:
        with st.form(key='my_predict_form', clear_on_submit=False):
            user_input_text = st.text_input(
                "Review Box",
                value=st.session_state.final_text_val,
                placeholder="Write your review or click the mic...",
                label_visibility="collapsed"
            )
            submit_clicked = st.form_submit_button("Predict Sentiment & Intent")

    # 5. Prediction Logic with a neat 3-column result display
    if submit_clicked:
        if user_input_text.strip():
            lang_res = detect_language_smart(user_input_text)
            intent_res = detect_intent(user_input_text)
            input_vec = vectorizer.transform([user_input_text.lower()])
            
            try:
                probs = model.predict_proba(input_vec)[0]
                prediction = 'Neutral' if max(probs) < 0.60 else model.predict(input_vec)[0]
            except:
                prediction = model.predict(input_vec)[0]
            
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
            st.warning("please enter some text or use mic input to predict.")

# --- TAB 3: Integrity & Emotions (Business Focused) ---
with tab3:
    col_integrity, col_emotions = st.columns(2)
    
    with col_integrity:
        st.subheader("🕵️ Integrity & Trust Analysis")
        # 1. Fake vs Real Pie Chart
        if 'is_fake' in df.columns and not df.empty:
            counts = df['is_fake'].value_counts()
            fig_p, ax_p = plt.subplots(figsize=(5,5))
            # labels based on index: 0 for Real, 1 for Fake
            l = [('Genuine' if i == 0 else 'Suspicious') for i in counts.index]
            counts.plot.pie(labels=l, autopct='%1.1f%%', colors=['#2E7D32','#C62828'], ax=ax_p, startangle=90)
            ax_p.set_ylabel('')
            st.pyplot(fig_p)
        
        st.divider()
        # 2. Sincerity Score
        st.write("**🎯 Feedback Depth (Sincerity)**")
        def check_sincerity(text):
            words = len(str(text).split())
            return "Detailed" if words > 5 else "Brief"
        
        df['sincerity'] = df['review_text'].apply(check_sincerity)
        sincerity_stats = df['sincerity'].value_counts()
        st.bar_chart(sincerity_stats)
        st.caption("Detailed reviews often indicate more engaged customers, while brief ones may suggest superficial feedback.")

    with col_emotions:
        st.subheader("🎭 Emotional Insights")
        # 1. Emoji Analysis
        def find_emojis(t): return [char for char in str(t) if char in emoji.EMOJI_DATA]
        emoji_list = df['review_text'].apply(find_emojis).sum()
        top_e = Counter(emoji_list).most_common(10)
        
        if top_e:
            st.write("**Top Visual Emotions (Emojis):**")
            st.table(pd.DataFrame(top_e, columns=['Emoji', 'Frequency']))
        
        st.divider()
        # 2. Sentiment Intensity
        st.write("**🔥 Sentiment Intensity Level**")
        # intensity on the basis of rating: 1 & 5 are strong, 2-4 are moderate
        def get_intensity(row):
            if row['rating'] in [1, 5]: return "Strong"
            else: return "Moderate"
            
        df['intensity'] = df.apply(get_intensity, axis=1)
        intensity_plot = pd.crosstab(df['sentiment'], df['intensity'])
        st.bar_chart(intensity_plot)
        st.caption("Strong Intensity means customers have very strong opinions about your product.")
        
# --- TAB 4: Custom Filters ---
with tab4:
    if not is_data_empty:
        st.subheader("🎯 Deep Dive Explorer")
        f1, f2 = st.columns(2)
        with f1: sl = st.multiselect("Language:", filtered_df['detected_lang'].unique(), default=filtered_df['detected_lang'].unique())
        with f2: ss = st.multiselect("Sentiment:", filtered_df['sentiment'].unique(), default=filtered_df['sentiment'].unique())
        st.dataframe(filtered_df[(filtered_df['detected_lang'].isin(sl)) & (filtered_df['sentiment'].isin(ss))][['review_text', 'detected_lang', 'sentiment', 'rating']], use_container_width=True)

# --- TAB 5: Strategic Insights ---
with tab5:
    if not is_data_empty:
        filtered_df['intent'] = filtered_df['review_text'].apply(detect_intent)
        st.subheader("🔦 Identifying Business Pain-Points")
        c_ins1, c_ins2 = st.columns(2)
        with c_ins1:
            st.write("**Top Concerns (Negative Intents)**")
            st.bar_chart(filtered_df[filtered_df['sentiment'] == 'Negative']['intent'].value_counts())
        with c_ins2:
            st.write("**Intent Heatmap**")
            st.write(pd.crosstab(filtered_df['intent'], filtered_df['sentiment']).style.background_gradient(cmap='YlOrRd'))
        
        st.divider()
        st.subheader("💡 AI Recommendation Engine")
        top_issue = filtered_df['intent'].value_counts().idxmax()
        avg_r = filtered_df['rating'].mean()
        
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.info(f"**AI Insight:** Most discussion is about **{top_issue}**. Average Rating: **{avg_r:.1f}/5**")
            if "Logistics" in top_issue and avg_r < 3.5:
                st.error("🚨 Fix delivery delays to improve satisfaction.")
            elif "Pricing" in top_issue:
                st.warning("⚠️ Consider seasonal discounts to tackle price sensitivity.")
            else:
                st.success("✅ Focus on scaling marketing for current best-sellers.")
        
        with col_r2:
            st.write("**📊 Competitor Benchmark**")
            bench_data = pd.DataFrame({'Metric': ['Quality', 'Price', 'Service'], 'You': [avg_r, 3.8, 4.2], 'Market': [4.0, 3.5, 3.9]})
            st.line_chart(bench_data.set_index('Metric'))

# Footer Metrics
st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Reviews", len(df))
m2.metric("Filtered Reviews", len(filtered_df))
m3.metric("Avg Rating", f"{filtered_df['rating'].mean():.1f} ⭐")
m4.metric("Market Sentiment", "Positive" if (filtered_df['sentiment'] == 'Positive').mean() * 100 > 50 else "Needs Work")

st.sidebar.caption("VoxInsight AI | Status: Online 🟢")