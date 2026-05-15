import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import emoji
from collections import Counter
from langdetect import detect, detect_langs

# 1. Page Configuration
st.set_page_config(page_title="Enterprise Sentiment Intelligence", layout="wide", page_icon="🏢")
st.title("🏛️ Enterprise Intelligence: Multilingual Sentiment Dashboard")

# 2. Asset Loading
@st.cache_resource
def load_assets():
    try:
        df = pd.read_csv('data/final_insights_multilingual.csv', encoding='utf-8-sig')
        metadata = pd.read_csv('data/correctedMetadata.csv')
        model = joblib.load('models/sentiment_model.pkl')
        vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
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
    """Business Intent Recognition (Actionable Category)"""
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
st.sidebar.title("🛠️ Business Intelligence Control Panel")
search_term = st.sidebar.text_input("🔍 Search Keyword (e.g. 'good', 'मस्त'):")

if search_term:
    df = df[df['review_text'].str.contains(search_term, case=False, na=False)]

st.sidebar.divider()
st.sidebar.subheader("📥 Export Reports")
csv_report = df.to_csv(index=False).encode('utf-8-sig')
st.sidebar.download_button("Download Full Report", csv_report, "bi_sentiment_analysis.csv", "text/csv")

# 5. Dashboard Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Market Performance", 
    "🤖 AI Predictor", 
    "🕵️ Integrity & Emotions", 
    "🎯 Advanced Filters",
    "💡 Strategic Insights"
])

# --- TAB 1: Performance Trends ---
with tab1:
    if not df.empty:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            pos_rate = (df['sentiment'] == 'Positive').mean() * 100
            st.plotly_chart(draw_gauge(pos_rate), use_container_width=True)
        
        with col_b:
            st.subheader("Market Sentiment by Language")
            lang_sent_table = pd.crosstab(df['detected_lang'], df['sentiment'])
            st.bar_chart(lang_sent_table)

        st.divider()
        st.subheader("🔥 Rating-Sentiment Density Heatmap")
        fig_heat, ax_heat = plt.subplots(figsize=(8, 4))
        sns.heatmap(pd.crosstab(df['rating'], df['sentiment']), annot=True, fmt='d', cmap='YlGnBu', ax=ax_heat)
        st.pyplot(fig_heat)
    else:
        st.warning("No data matches your current search.")

# --- TAB 2: Live AI Predictor ---
with tab2:
    st.subheader("Real-time Multilingual Inference")
    user_input = st.text_area("Analyze customer feedback:", height=100)
    if st.button("Predict Sentiment & Intent"):
        if user_input:
            lang_res = detect_language_smart(user_input)
            intent_res = detect_intent(user_input)
            input_vec = vectorizer.transform([user_input.lower()])
            prediction = model.predict(input_vec)[0]
            
            c1, c2, c3 = st.columns(3)
            c1.metric("Detected Language", lang_res)
            c2.metric("Primary Intent", intent_res)
            if prediction == 'Positive': c3.success(f"Sentiment: {prediction} 😊")
            elif prediction == 'Negative': c3.error(f"Sentiment: {prediction} 😡")
            else: c3.warning(f"Sentiment: {prediction} 😐")
        else:
            st.info("Please enter text for analysis.")

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
    st.subheader("🎯 Deep Dive Explorer")
    if not df.empty:
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            sel_lang = st.multiselect("Language Filter:", df['detected_lang'].unique(), default=df['detected_lang'].unique())
        with f_col2:
            sel_sent = st.multiselect("Sentiment Filter:", df['sentiment'].unique(), default=df['sentiment'].unique())
        
        final_view = df[(df['detected_lang'].isin(sel_lang)) & (df['sentiment'].isin(sel_sent))]
        st.dataframe(final_view[['review_text', 'detected_lang', 'sentiment', 'rating']], use_container_width=True)

# --- TAB 5: Strategic Business Insights (New Feature) ---
with tab5:
    st.subheader("🔦 Identifying Business Pain-Points")
    df['intent'] = df['review_text'].apply(detect_intent)
    
    col_ins1, col_ins2 = st.columns(2)
    
    with col_ins1:
        st.write("**Top Concerns (Negative Intents)**")
        neg_intents = df[df['sentiment'] == 'Negative']['intent'].value_counts()
        if not neg_intents.empty:
            st.bar_chart(neg_intents)
        else:
            st.success("No negative trends found in current data!")

    with col_ins2:
        st.write("**Intent Distribution Heatmap**")
        intent_sent = pd.crosstab(df['intent'], df['sentiment'])
        st.write(intent_analysis := intent_sent.style.background_gradient(cmap='YlOrRd'))
    
    st.divider()
    st.write("**📝 Business Strategy Recommendations:**")
    if pos_rate < 60:
        st.error("🚨 **Immediate Attention:** customers are not satisfied. Please review the 'Quality' and 'Logistics' departments.")
    elif len(df[df['intent'] == "🚚 Logistics"]) > len(df) * 0.2:
        st.warning("⚠️ **Logistics Warning:** delivery complaints are increasing. Consider changing supply chain partners or improving tracking.")
    else:
        st.success("✅ **Market Leader Potential:** your product is performing well. Invest in new features and marketing.")
st.sidebar.markdown("---")
st.sidebar.caption("Enterprise AI Engine | Status: Online 🟢")