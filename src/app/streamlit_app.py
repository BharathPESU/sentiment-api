# src/app/streamlit_app.py
import os
import time
import requests
import pandas as pd
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="Customer Review Sentiment Analyzer",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

API_URL = os.getenv("API_URL", "http://localhost:8000")
MODEL_DIR = os.getenv("MODEL_DIR", "models/sentiment")

# Custom CSS for polished styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #64748B;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .sentiment-positive {
        color: #16A34A;
        font-size: 2rem;
        font-weight: 700;
    }
    .sentiment-negative {
        color: #DC2626;
        font-size: 2rem;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Helper function to perform inference (FastAPI with fallback to direct model loading)
@st.cache_resource
def get_local_classifier():
    """Load the model directly into memory as a fallback if the API server isn't running."""
    if os.path.exists(MODEL_DIR):
        try:
            from transformers import pipeline
            return pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR)
        except Exception:
            return None
    return None

def analyze_sentiment(text: str):
    """Call FastAPI backend or fallback to local transformer model."""
    start_time = time.time()
    # 1. Try FastAPI backend first
    try:
        resp = requests.post(
            f"{API_URL.rstrip('/')}/predict",
            json={"text": text},
            timeout=4
        )
        if resp.status_code == 200:
            data = resp.json()
            latency = (time.time() - start_time) * 1000
            return {
                "label": data["label"].upper(),
                "score": float(data["score"]),
                "source": "FastAPI Service",
                "latency_ms": round(latency, 1)
            }
    except Exception:
        pass

    # 2. Fallback to direct model pipeline
    local_pipe = get_local_classifier()
    if local_pipe is not None:
        result = local_pipe(text, truncation=True)[0]
        latency = (time.time() - start_time) * 1000
        return {
            "label": result["label"].upper(),
            "score": float(result["score"]),
            "source": "Local DistilBERT Model",
            "latency_ms": round(latency, 1)
        }

    return {"error": "Unable to connect to FastAPI and local model not found in " + MODEL_DIR}

# Sidebar configuration & diagnostics
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/speech-bubble-with-dots.png", width=110)
    st.title("Sentiment Engine")
    st.markdown("Fine-tuned **DistilBERT** transformer trained on consumer review benchmarks.")
    st.divider()

    st.subheader("⚙️ Backend Status")
    # Quick health probe
    api_online = False
    try:
        h_resp = requests.get(f"{API_URL.rstrip('/')}/health", timeout=2)
        if h_resp.status_code == 200 and h_resp.json().get("model_loaded"):
            api_online = True
    except Exception:
        api_online = False

    if api_online:
        st.success(f"FastAPI API: Connected (`{API_URL}`)")
    else:
        st.info("FastAPI API: Offline (Using Direct In-Memory Model)")

    st.divider()
    st.markdown("### 📊 Capabilities")
    st.markdown("- **Single Review Deep-Dive**\n- **Batch CSV Analysis**\n- **Confidence & Latency Metrics**")

# Main Header
st.markdown('<div class="main-header">💬 Customer Review Sentiment Analyzer</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Classify incoming customer feedback into positive or negative sentiment in real-time.</div>', unsafe_allow_html=True)

# Tabs
tab_single, tab_batch, tab_about = st.tabs(["🔍 Analyze Single Review", "📂 Batch Analysis (CSV)", "ℹ️ Architecture & Metrics"])

# ================= TAB 1: Single Review =================
with tab_single:
    col_input, col_result = st.columns([1.1, 0.9], gap="large")

    with col_input:
        st.subheader("Enter Customer Review")

        # Sample reviews
        st.caption("Quick sample reviews:")
        sample_cols = st.columns(3)
        sample_text = ""
        if sample_cols[0].button("🌟 Stellar Experience", use_container_width=True):
            st.session_state["review_input"] = "The product arrived two days early and exceeded all my expectations! Outstanding craftsmanship and customer support."
        if sample_cols[1].button("😡 Dissatisfied Buyer", use_container_width=True):
            st.session_state["review_input"] = "Completely broken on arrival. Cheap materials, horrible packaging, and no response from warranty support."
        if sample_cols[2].button("⚖️ Mixed / Neutral", use_container_width=True):
            st.session_state["review_input"] = "The build quality is satisfactory for the price point, but the battery life is slightly below what was advertised."

        user_review = st.text_area(
            "Review Content",
            value=st.session_state.get("review_input", "The customer service representative resolved my billing issue in under three minutes. Truly remarkable service!"),
            height=160,
            placeholder="Type or paste any customer review here..."
        )

        char_count = len(user_review)
        word_count = len(user_review.split())
        st.caption(f"Length: **{char_count}** characters | **{word_count}** words")

        analyze_btn = st.button("🚀 Analyze Sentiment", type="primary", use_container_width=True)

    with col_result:
        st.subheader("Prediction Result")
        if analyze_btn or user_review:
            if not user_review.strip():
                st.warning("Please enter some review text to analyze.")
            else:
                with st.spinner("Classifying sentiment..."):
                    res = analyze_sentiment(user_review)

                if "error" in res:
                    st.error(res["error"])
                else:
                    is_pos = res["label"] == "POSITIVE"
                    score_pct = int(res["score"] * 100)

                    # Highlight container
                    res_container = st.container(border=True)
                    with res_container:
                        r_col1, r_col2 = st.columns([1, 1])
                        with r_col1:
                            st.caption("PREDICTED SENTIMENT")
                            if is_pos:
                                st.markdown('<div class="sentiment-positive">😊 POSITIVE</div>', unsafe_allow_html=True)
                            else:
                                st.markdown('<div class="sentiment-negative">🙁 NEGATIVE</div>', unsafe_allow_html=True)
                        with r_col2:
                            st.caption("CONFIDENCE SCORE")
                            st.metric(label="", value=f"{score_pct}%", delta=f"{res['latency_ms']} ms")

                        st.progress(res["score"])

                        # Confidence Breakdown
                        pos_score = res["score"] if is_pos else (1.0 - res["score"])
                        neg_score = (1.0 - res["score"]) if is_pos else res["score"]

                        st.markdown("**Probability Breakdown**")
                        c1, c2 = st.columns(2)
                        c1.metric("Positive Probability", f"{pos_score * 100:.2f}%")
                        c2.metric("Negative Probability", f"{neg_score * 100:.2f}%")

                        st.caption(f"Inference Provider: **{res['source']}**")

# ================= TAB 2: Batch Analysis =================
with tab_batch:
    st.subheader("Batch Customer Feedback Processing")
    st.markdown("Upload a CSV file containing customer reviews to classify sentiment at scale.")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    demo_data_btn = st.button("Load Sample Reviews Dataset")
    if demo_data_btn or uploaded_file is not None:
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.DataFrame({
                "review_id": [f"REV-{100+i}" for i in range(6)],
                "text": [
                    "Absolutely love this! Works flawlessly and battery lasts days.",
                    "Disappointing purchase. Stopped working after 3 days.",
                    "Great value for money. Very sturdy and well made.",
                    "Delivery took forever and package was crushed. Terrible.",
                    "Best customer support team ever. Highly recommended!",
                    "Mediocre performance, but acceptable considering the cheap price."
                ]
            })

        st.write("Preview of Input Data:")
        st.dataframe(df.head(6), use_container_width=True)

        # Detect text column
        text_cols = [c for c in df.columns if df[c].dtype == object]
        chosen_col = st.selectbox("Select Review Text Column", text_cols, index=0 if text_cols else None)

        if st.button("Run Batch Inference", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            labels, scores = [], []

            for idx, row in df.iterrows():
                text_val = str(row[chosen_col])
                res = analyze_sentiment(text_val)
                labels.append(res.get("label", "ERROR"))
                scores.append(res.get("score", 0.0))
                progress_bar.progress((idx + 1) / len(df))
                status_text.text(f"Processed {idx + 1} / {len(df)} reviews...")

            df["predicted_sentiment"] = labels
            df["confidence"] = scores
            status_text.success("Batch classification complete!")

            # Summary Metrics
            pos_count = (df["predicted_sentiment"] == "POSITIVE").sum()
            neg_count = (df["predicted_sentiment"] == "NEGATIVE").sum()

            m1, m2, m3 = st.columns(3)
            m1.metric("Total Reviews", len(df))
            m2.metric("Positive Sentiment", f"{pos_count} ({pos_count/len(df)*100:.1f}%)")
            m3.metric("Negative Sentiment", f"{neg_count} ({neg_count/len(df)*100:.1f}%)")

            # Chart
            sentiment_counts = pd.DataFrame({
                "Sentiment": ["Positive", "Negative"],
                "Count": [pos_count, neg_count]
            })
            st.bar_chart(sentiment_counts.set_index("Sentiment"))

            st.dataframe(df, use_container_width=True)

            # Export
            csv_data = df.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Download Results CSV",
                data=csv_data,
                file_name="sentiment_analysis_results.csv",
                mime="text/csv"
            )

# ================= TAB 3: Architecture & Metrics =================
with tab_about:
    st.subheader("System Architecture")
    st.markdown("""
    This application is powered by an end-to-end MLOps pipeline:
    - **Backbone Model**: Hugging Face DistilBERT (`distilbert-base-uncased`) fine-tuned on binary sentiment classification.
    - **Inference Service**: High-performance asynchronous FastAPI server running under Uvicorn with Prometheus metrics.
    - **Model Versioning**: Managed via DVC with S3 remote storage (`bharath-mlops-practice`).
    - **Quality Gate & Orchestration**: Automated Prefect flow retraining triggered by Kolmogorov-Smirnov drift detection.
    """)

