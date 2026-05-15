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
        # Path for Streamlit Cloud 
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
st.sidebar.title("🛠️ BI Control Panel")
search_term = st.sidebar.text_input("🔍 Search Keyword (e.g. 'good', 'bad','मस्त','अच्छा'):")

# --- Empty States Logic (Filtering Data) ---
filtered_df = df.copy()
is_data_empty = False

if search_term:
    filtered_df = df[df['review_text'].str.contains(search_term, case=False, na=False)]
    if filtered_df.empty:
        st.sidebar.warning(f"'{search_term}' Data not found.")
        is_data_empty = True

# --- Sidebar About Section ---
st.sidebar.divider()
with st.sidebar.expander("ℹ️ About This Project"):
    st.write("""
    **Product Sentiment BI Tool**
    ही एक प्रगत AI सिस्टीम आहे जी ग्राहकांच्या रिव्ह्यूचे विश्लेषण करते.
    
    * **Features:** Multilingual support, Intent detection, Data integrity.
    * **Developer:** Vaishnavi Giri
    * **Tech:** NLP, Python, Streamlit
    """)

st.sidebar.divider()
st.sidebar.subheader("📥 Export Reports")
csv_report = filtered_df.to_csv(index=False).encode('utf-8-sig')
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
    if not is_data_empty:
        col_a, col_b = st.columns([1, 2])
        with col_a:
            pos_rate = (filtered_df['sentiment'] == 'Positive').mean() * 100
            st.plotly_chart(draw_gauge(pos_rate), use_container_width=True)
        
        with col_b:
            st.subheader("Market Sentiment by Language")
            lang_sent_table = pd.crosstab(filtered_df['detected_lang'], filtered_df['sentiment'])
            st.bar_chart(lang_sent_table)

        st.divider()
        st.subheader("🔥 Rating-Sentiment Density Heatmap")
        fig_heat, ax_heat = plt.subplots(figsize=(8, 4))
        sns.heatmap(pd.crosstab(filtered_df['rating'], filtered_df['sentiment']), annot=True, fmt='d', cmap='YlGnBu', ax=ax_heat)
        st.pyplot(fig_heat)
    else:
        st.info("🔍 Filtered data not found. Please try a different keyword.")

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

# --- TAB 3: Integrity & Emotions ---
with tab3:
    if not is_data_empty:
        col_integrity, col_emotions = st.columns(2)
        
        with col_integrity:
            st.subheader("🕵️ Integrity & Trust Analysis")
            if 'is_fake' in filtered_df.columns:
                counts = filtered_df['is_fake'].value_counts()
                fig_p, ax_p = plt.subplots(figsize=(5,5))
                l = [('Genuine' if i == 0 else 'Suspicious') for i in counts.index]
                counts.plot.pie(labels=l, autopct='%1.1f%%', colors=['#2E7D32','#C62828'], ax=ax_p, startangle=90)
                ax_p.set_ylabel('')
                st.pyplot(fig_p)
            
            st.divider()
            st.write("**🎯 Feedback Depth (Sincerity)**")
            def check_sincerity(text):
                words = len(str(text).split())
                return "Detailed" if words > 5 else "Brief"
            
            filtered_df['sincerity'] = filtered_df['review_text'].apply(check_sincerity)
            sincerity_stats = filtered_df['sincerity'].value_counts()
            st.bar_chart(sincerity_stats)

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
            def get_intensity(row):
                return "Strong" if row['rating'] in [1, 5] else "Moderate"
                
            filtered_df['intensity'] = filtered_df.apply(get_intensity, axis=1)
            intensity_plot = pd.crosstab(filtered_df['sentiment'], filtered_df['intensity'])
            st.bar_chart(intensity_plot)
    else:
        st.info("Data not available for emotional analysis.")

# --- TAB 4: Custom Filters ---
with tab4:
    st.subheader("🎯 Deep Dive Explorer")
    if not is_data_empty:
        f_col1, f_col2 = st.columns(2)
        with f_col1:
            sel_lang = st.multiselect("Language Filter:", filtered_df['detected_lang'].unique(), default=filtered_df['detected_lang'].unique())
        with f_col2:
            sel_sent = st.multiselect("Sentiment Filter:", filtered_df['sentiment'].unique(), default=filtered_df['sentiment'].unique())
        
        final_view = filtered_df[(filtered_df['detected_lang'].isin(sel_lang)) & (filtered_df['sentiment'].isin(sel_sent))]
        st.dataframe(final_view[['review_text', 'detected_lang', 'sentiment', 'rating']], use_container_width=True)

# --- TAB 5: Strategic Insights ---
with tab5:
    if not is_data_empty:
        st.subheader("🔦 Identifying Business Pain-Points")
        filtered_df['intent'] = filtered_df['review_text'].apply(detect_intent)
        
        col_ins1, col_ins2 = st.columns(2)
        
        with col_ins1:
            st.write("**Top Concerns (Negative Intents)**")
            neg_intents = filtered_df[filtered_df['sentiment'] == 'Negative']['intent'].value_counts()
            if not neg_intents.empty:
                st.bar_chart(neg_intents)
            else:
                st.success("No negative trends found in current data!")

        with col_ins2:
            st.write("**Intent Distribution Heatmap**")
            intent_sent = pd.crosstab(filtered_df['intent'], filtered_df['sentiment'])
            st.write(intent_sent.style.background_gradient(cmap='YlOrRd'))
        
        st.divider()
        st.write("**📝 Business Strategy Recommendations:**")
        cur_pos_rate = (filtered_df['sentiment'] == 'Positive').mean() * 100
        if cur_pos_rate < 60:
            st.error("🚨 **Immediate Attention:** customers are not satisfied. 'Quality' and 'Logistics' departments need attention.")
        elif len(filtered_df[filtered_df['intent'] == "🚚 Logistics"]) > len(filtered_df) * 0.2:
            st.warning("⚠️ **Logistics Warning:** Delivery complaints are increasing. Improve the supply chain.")
        else:
            st.success("✅ **Market Leader Potential:** Your product is performing well.")
    else:
        st.info("Data not available for strategic analysis.")

st.sidebar.markdown("---")
st.sidebar.caption("Enterprise AI Engine | Status: Online 🟢")