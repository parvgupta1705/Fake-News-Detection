import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from src.pipeline import load_or_train_model, predict_article, get_explanations, credibility_score, DATA_DIR

st.set_page_config(page_title="TruthLens | Fake News Detection", page_icon="📰", layout="wide")

st.markdown("""
<style>
.main-header {font-size:2.35rem;font-weight:800;letter-spacing:-1px}
.sub-header {color:#7c8798;font-size:1rem}
.result-card {padding:1rem 1.2rem;border:1px solid rgba(128,128,128,.25);border-radius:14px}
.small-muted {font-size:.85rem;color:#7c8798}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📰 TruthLens</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Based Fake News & Misinformation Detection System · TF-IDF + Classical ML + Explainability</div>', unsafe_allow_html=True)
st.write("")

@st.cache_resource
def get_bundle():
    return load_or_train_model()

with st.spinner("Loading text-processing pipeline and classifiers…"):
    bundle = get_bundle()

demo_mode = bundle.get("demo_mode", False)
if demo_mode:
    st.info("Demo mode is active because no research dataset was found. Predictions are illustrative. Add the labelled dataset and run `python train.py` to train/evaluate on real data.")
else:
    st.success(f"Trained model loaded · Selected classifier: {bundle['best_model_name']}")

with st.sidebar:
    st.header("Project controls")
    st.write("**Selected model:**", bundle["best_model_name"])
    st.write("**Feature extraction:**", "TF-IDF (unigrams + bigrams)")
    st.write("**Feature limit:**", "15,000")
    st.divider()
    st.caption("This tool estimates textual patterns. It does not verify claims against authoritative sources and must not be treated as a definitive fact-check.")

tab_detect, tab_eval, tab_about = st.tabs(["🔎 Detect news", "📊 Model evaluation", "ℹ️ About the project"])

with tab_detect:
    left, right = st.columns([1.25, 0.75], gap="large")
    with left:
        headline = st.text_input("News headline", placeholder="Paste the headline here…")
        article = st.text_area("Article text", height=240, placeholder="Paste the article body here…")
        c1, c2 = st.columns(2)
        with c1:
            source = st.text_input("Publisher / source (optional)", placeholder="e.g. Example News")
        with c2:
            source_reputation = st.selectbox("Known source reputation (optional)", ["Unknown", "Established / verified", "Unverified / unfamiliar"])
        submitted = st.button("Analyze article", type="primary", use_container_width=True)
        if st.button("Load sample article", use_container_width=True):
            st.session_state["sample_loaded"] = True
            st.rerun()
        if st.session_state.get("sample_loaded", False):
            headline = "Officials announce a new public transport plan after consultation"
            article = ("City officials announced a proposed public transport plan on Tuesday. "
                       "The proposal will be reviewed by the local council next month, according to the published meeting notes. "
                       "Residents can submit comments through the council's public consultation process.")
            st.session_state["sample_loaded"] = False
            st.info("Sample loaded. Click Analyze article to run it.")
    with right:
        st.markdown("#### How to read the result")
        st.write("- **REAL-like / FAKE-like** describes patterns learned from the training corpus.")
        st.write("- **Model score** is a model-derived score, not the probability that a claim is true.")
        st.write("- **Influential terms** show text features that contributed to the classifier's decision.")
        st.write("- **Source score** is a heuristic and is not an independent source verification.")

    if submitted:
        text = (headline.strip() + " " + article.strip()).strip()
        if len(text.split()) < 5:
            st.warning("Please enter a headline and/or article with at least five words.")
        else:
            result = predict_article(bundle, text)
            pred = result["label"]
            st.divider()
            col1, col2, col3 = st.columns(3)
            col1.metric("Prediction", pred)
            col2.metric("Model score", f"{result['score']:.1f}/100", help="Relative model score; not a calibrated probability of truth.")
            col3.metric("Source credibility heuristic", f"{credibility_score(text, source, source_reputation):.0f}/100")
            st.caption("⚠️ Automated text classification can be wrong. Verify consequential claims with multiple reliable sources.")
            st.markdown("#### Why did the model predict this?")
            terms = get_explanations(bundle, text, pred)
            if terms:
                st.write("Terms with the strongest influence in this model:")
                st.write(", ".join(f"`{t}`" for t in terms))
            else:
                st.write("No clear individual terms could be extracted for this prediction.")
            with st.expander("Compare all five classifiers"):
                rows = []
                for name, model in bundle["models"].items():
                    p = model.predict([text])[0]
                    rows.append({"Classifier": name, "Prediction": p})
                st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            with st.expander("Show preprocessing preview"):
                st.code(" ".join(text.lower().split()), language="text")
                st.caption("The TF-IDF vectorizer handles tokenization and feature weighting during inference.")

with tab_eval:
    metrics = bundle.get("metrics", {})
    if demo_mode:
        st.warning("Metrics are intentionally not presented as research performance in demo mode. Train on the labelled corpus to generate a held-out evaluation.")
    elif metrics:
        metric_df = pd.DataFrame(metrics).T.reset_index().rename(columns={"index":"Model"})
        st.dataframe(metric_df, use_container_width=True, hide_index=True)
        if "Linear SVM" in metrics:
            st.markdown("#### Linear SVM confusion matrix")
            cm = bundle.get("confusion_matrix")
            if cm is not None:
                fig, ax = plt.subplots()
                ax.imshow(cm)
                ax.set_xticks([0,1], labels=["FAKE","REAL"])
                ax.set_yticks([0,1], labels=["FAKE","REAL"])
                ax.set_xlabel("Predicted label")
                ax.set_ylabel("True label")
                for i in range(2):
                    for j in range(2):
                        ax.text(j, i, str(cm[i][j]), ha="center", va="center")
                st.pyplot(fig)
                plt.close(fig)
        st.caption("Metrics are generated from the dataset and split used by this local run; they may differ from the paper.")
    else:
        st.info("No evaluation metrics are available yet. Add the research dataset and run `python train.py`.")

with tab_about:
    st.markdown("""
### Research alignment
This implementation follows the paper's lightweight classical-ML approach:
- Text input combines title/headline and article body.
- TF-IDF vectorization with unigram and bigram features, up to 15,000 features.
- Five classifiers: Multinomial Naive Bayes, Logistic Regression, Linear SVM, Passive-Aggressive and Random Forest.
- Model comparison using accuracy, precision, recall and F1-score.
- A post-hoc influential-term explanation based on linear-model coefficients where available.
- A lightweight source/linguistic credibility heuristic.

### Important limitations
The system learns correlations in labelled text, not factual truth. It may misclassify satire, opinion pieces, breaking news, and articles outside the training domain. The source score is a transparent heuristic, not a verified reputation database. Use human fact-checking for important decisions.
""")
    st.markdown("**Dataset described in the paper:** 6,335 articles, with 3,171 REAL and 3,164 FAKE; an 80/20 train/test split. See the paper's dataset and methodology sections.")
