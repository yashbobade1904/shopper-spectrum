# Shopper Spectrum

Customer segmentation (RFM + K-Means) and product recommendation (item-based
collaborative filtering) for e-commerce transaction data, with a Streamlit
app for interactive use.

## Folder contents

```
Shopper_Spectrum_Analysis.ipynb   # Full analysis: cleaning, EDA, RFM, clustering, recommender
app.py                            # Streamlit app (2 modules: recommendations + segmentation)
requirements.txt                  # Python dependencies
models/                           # Trained model artifacts (used by app.py)
    kmeans_model.pkl
    scaler.pkl
    cluster_label_map.pkl
    product_names.pkl
    item_similarity.npz
    rfm_segmented.pkl
```

## Setup

```bash
pip install -r requirements.txt
```

## 1. Re-running the analysis notebook

The notebook already contains full output (charts, tables, printed results)
from a real run on the dataset. To re-run it yourself:

1. Place your `online_retail.csv` file in this same folder.
2. Open `Shopper_Spectrum_Analysis.ipynb` in Jupyter and run all cells.
3. This regenerates everything in `models/`.

## 2. Launching the Streamlit app

```bash
streamlit run app.py
```

This opens a browser tab with two modules in the sidebar:

- **Product Recommendations** — type a product name, get the top-N most
  similar products based on co-purchase patterns.
- **Customer Segmentation** — enter Recency / Frequency / Monetary values
  for a customer, get their predicted segment (High-Value, Regular,
  Occasional, or At-Risk) with a plain-language description.

The app reads directly from the `models/` folder, so as long as that folder
sits next to `app.py`, it works out of the box — no need to re-run the
notebook first.

## Segments at a glance

| Segment | Recency | Frequency | Monetary | Meaning |
|---|---|---|---|---|
| High-Value | Low | High | High | Best customers — protect & reward |
| Regular | Medium | Medium | Medium | Steady — grow via upsell/cross-sell |
| Occasional | Low | Low | Low | Recent but infrequent — build habit |
| At-Risk | High | Low | Low | Dormant — win-back campaigns |

## Notes

- Clustering uses log-transformed RFM features (the raw values are heavily
  right-skewed) standardized with `StandardScaler`, then K-Means with k=4.
- k=4 was chosen over the silhouette-optimal k=2 because it maps onto four
  business-actionable segments while still scoring reasonably (~0.34)
  on silhouette — see Section 6 of the notebook for the full tradeoff.
- The recommender is item-based collaborative filtering: cosine similarity
  between products based on customer purchase-quantity vectors.
