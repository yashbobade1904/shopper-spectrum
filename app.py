"""
Shopper Spectrum — Streamlit App
Customer Segmentation (RFM + KMeans) and Product Recommendation (Item-based CF)

Run with:  streamlit run app.py
Expects a sibling "models/" folder containing:
  kmeans_model.pkl, scaler.pkl, cluster_label_map.pkl,
  product_names.pkl, item_similarity.npz
"""

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import scipy.sparse as sp

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛍️",
    layout="wide",
)

# ----------------------------------------------------------------------------
# Cached model loading
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    missing = []
    paths = {
        "kmeans": os.path.join(MODELS_DIR, "kmeans_model.pkl"),
        "scaler": os.path.join(MODELS_DIR, "scaler.pkl"),
        "label_map": os.path.join(MODELS_DIR, "cluster_label_map.pkl"),
        "product_names": os.path.join(MODELS_DIR, "product_names.pkl"),
        "similarity": os.path.join(MODELS_DIR, "item_similarity.npz"),
    }
    for name, p in paths.items():
        if not os.path.exists(p):
            missing.append(p)
    if missing:
        return None, missing

    with open(paths["kmeans"], "rb") as f:
        kmeans = pickle.load(f)
    with open(paths["scaler"], "rb") as f:
        scaler = pickle.load(f)
    with open(paths["label_map"], "rb") as f:
        label_map = pickle.load(f)
    with open(paths["product_names"], "rb") as f:
        product_names = pickle.load(f)
    similarity = sp.load_npz(paths["similarity"])

    artifacts = {
        "kmeans": kmeans,
        "scaler": scaler,
        "label_map": label_map,
        "product_names": product_names,
        "similarity": similarity,
    }
    return artifacts, []


artifacts, missing_files = load_artifacts()

# ----------------------------------------------------------------------------
# Segment descriptions shown to the user
# ----------------------------------------------------------------------------
SEGMENT_INFO = {
    "High-Value": {
        "color": "#2E7D32",
        "desc": "Frequent, big-spending, recent buyers. Your most valuable customers — "
                "prioritize retention with loyalty perks and early access.",
    },
    "Regular": {
        "color": "#1565C0",
        "desc": "Steady mid-frequency, mid-spend customers. The largest pool of revenue "
                "to grow further through upsell and cross-sell.",
    },
    "Occasional": {
        "color": "#F9A825",
        "desc": "Bought recently but rarely, for small amounts. Good candidates for "
                "engagement campaigns to build a buying habit.",
    },
    "At-Risk": {
        "color": "#C62828",
        "desc": "Haven't purchased in a long time and historically bought rarely. "
                "Target with win-back campaigns before they're lost for good.",
    },
}

# ----------------------------------------------------------------------------
# Sidebar navigation
# ----------------------------------------------------------------------------
st.sidebar.title("🛍️ Shopper Spectrum")
page = st.sidebar.radio(
    "Choose a module",
    ["Product Recommendations", "Customer Segmentation"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Built on RFM analysis + K-Means clustering for segmentation, and "
    "item-based collaborative filtering for recommendations."
)

if missing_files:
    st.error(
        "Model files are missing. Make sure a `models/` folder sits next to "
        "`app.py` containing:\n\n" + "\n".join(f"- `{os.path.basename(p)}`" for p in missing_files)
    )
    st.stop()

# ----------------------------------------------------------------------------
# Page 1: Product Recommendations
# ----------------------------------------------------------------------------
if page == "Product Recommendations":
    st.title("🔗 Product Recommendations")
    st.write(
        "Enter a product name to find the top similar products, based on what "
        "customers tend to buy together."
    )

    product_names = artifacts["product_names"]
    similarity = artifacts["similarity"]

    query = st.text_input(
        "Product name",
        placeholder="e.g. WHITE HANGING HEART T-LIGHT HOLDER",
    )
    n_results = st.slider("Number of recommendations", min_value=1, max_value=10, value=5)

    def get_similar_products(product_name, n=5):
        if product_name not in product_names:
            matches = [p for p in product_names if product_name.upper() in p.upper()]
            if not matches:
                return None, [], []
            candidates = matches[:8]
        else:
            candidates = [product_name]
        matched = candidates[0]
        idx = product_names.index(matched)
        sims = similarity[idx].toarray().ravel()
        sims[idx] = -1
        top_idx = np.argsort(sims)[::-1][:n]
        results = [(product_names[i], float(sims[i])) for i in top_idx]
        return matched, results, candidates

    if st.button("Get Recommendations", type="primary") and query.strip():
        matched, recs, candidates = get_similar_products(query.strip(), n_results)
        if matched is None:
            st.warning("No matching product found. Try a shorter or different keyword.")
        else:
            if len(candidates) > 1:
                st.caption(
                    f"Multiple products matched — showing results for: **{matched}** "
                    f"({len(candidates)} other matches found, try a more specific name to pick a different one)"
                )
            else:
                st.caption(f"Showing results for: **{matched}**")

            st.subheader("Top similar products")
            for i, (name, score) in enumerate(recs):
                with st.container(border=True):
                    st.markdown(f"**{i+1}. {name}**")
                    st.progress(min(max(score, 0.0), 1.0), text=f"Similarity: {score:.3f}")
    elif query.strip() == "":
        st.info("👆 Enter a product name above and click **Get Recommendations**.")

# ----------------------------------------------------------------------------
# Page 2: Customer Segmentation
# ----------------------------------------------------------------------------
else:
    st.title("🎯 Customer Segmentation")
    st.write(
        "Enter a customer's Recency, Frequency, and Monetary values to predict "
        "which segment they belong to."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        recency = st.number_input(
            "Recency (days since last purchase)", min_value=0, max_value=2000, value=30, step=1
        )
    with col2:
        frequency = st.number_input(
            "Frequency (number of purchases)", min_value=1, max_value=2000, value=3, step=1
        )
    with col3:
        monetary = st.number_input(
            "Monetary (total amount spent)", min_value=0.0, max_value=1_000_000.0, value=500.0, step=10.0
        )

    if st.button("Predict Segment", type="primary"):
        kmeans = artifacts["kmeans"]
        scaler = artifacts["scaler"]
        label_map = artifacts["label_map"]

        log_features = pd.DataFrame(
            [[np.log1p(recency), np.log1p(frequency), np.log1p(monetary)]],
            columns=["Recency_log", "Frequency_log", "Monetary_log"],
        )
        scaled = scaler.transform(log_features)
        cluster = int(kmeans.predict(scaled)[0])
        segment = label_map.get(cluster, f"Cluster {cluster}")

        info = SEGMENT_INFO.get(segment, {"color": "#555", "desc": ""})

        st.markdown(
            f"""
            <div style="padding: 1.2rem 1.5rem; border-radius: 10px; border-left: 6px solid {info['color']};
                        background-color: rgba(0,0,0,0.03); margin-top: 1rem;">
                <h3 style="margin:0; color:{info['color']};">Predicted Segment: {segment}</h3>
                <p style="margin-top:0.5rem;">{info['desc']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("Segment Reference")
    ref_df = pd.DataFrame(
        [
            {"Segment": "High-Value", "Typical Recency": "~12 days", "Typical Frequency": "~14 orders", "Typical Monetary": "~£8,000+"},
            {"Segment": "Regular", "Typical Recency": "~71 days", "Typical Frequency": "~4 orders", "Typical Monetary": "~£1,800"},
            {"Segment": "Occasional", "Typical Recency": "~18 days", "Typical Frequency": "~2 orders", "Typical Monetary": "~£550"},
            {"Segment": "At-Risk", "Typical Recency": "~183 days", "Typical Frequency": "~1.3 orders", "Typical Monetary": "~£340"},
        ]
    )
    st.dataframe(ref_df, use_container_width=True, hide_index=True)
