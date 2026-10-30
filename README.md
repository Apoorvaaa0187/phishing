# PhishShield — URL-only rebuild

This rebuild separates **live URL screening** from **offline model evaluation** while using one shared URL-only feature extractor in both paths.

## Dataset (recommended)

Use the **PhiUSIIL Phishing URL (Website) Dataset** from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset

Download `PhiUSIIL_Phishing_URL_Dataset.csv` and place it in this project folder beside `app.py`.

Important dataset caveat: PhiUSIIL contains 54 features, including features derived from webpage/source-code analysis. This project deliberately does **not** train on those precomputed webpage features. It uses only the raw `URL` column and the `Label` column, then derives the same 25 lexical/structural URL features for training and live scans. UCI's label convention is `1 = legitimate`, `0 = phishing`; this project converts it to `1 = phishing`, `0 = legitimate`.

Dataset facts from UCI: 235,795 instances; 134,850 legitimate and 100,945 phishing URLs; no missing values reported. The FILENAME field is ignored.

## Windows setup

1. Extract the ZIP.
2. Download the CSV from UCI and put it beside `app.py`.
3. Double-click `run_windows.bat`, or run:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python app.py
```

4. Open http://127.0.0.1:8000

First training can take several minutes. The app caches trained models and metrics in `artifacts/model_bundle.joblib`. It checks the dataset SHA-256 and application version before reusing the cache. Click **Retrain & evaluate** to force a fresh evaluation.

## Evaluation methodology

- Deduplicate by URL.
- Convert labels to phishing-positive (`1`).
- Derive all features from the URL string only.
- Use an 80/20 domain-group holdout split where possible, keeping URLs from the same approximate base domain together to reduce leakage. A stratified random split is used only if the group split cannot provide both classes.
- Compare Logistic Regression, Random Forest, Extra Trees, and HistGradientBoosting on the same test set.
- Calculate accuracy, precision, recall, F1, ROC-AUC, PR-AUC, ROC points, and confusion matrices from held-out test predictions.

The KNN model and old cached metrics were removed to avoid a very expensive nearest-neighbour evaluation on the full dataset. All four models use the same held-out test rows.

## Live detection

`POST /api/predict` applies the same `extract_url_features()` function used to prepare the training matrix. It does not fabricate missing features or use page-derived features. The scanner does not visit the submitted website; it is a URL-pattern classifier and cannot guarantee safety.

## API

- `GET /` — dashboard
- `GET /api/status` — model readiness
- `GET /api/metrics` — current held-out evaluation
- `POST /api/evaluate` — retrain and recompute all metrics
- `POST /api/predict` — scan one URL

## Important limitations

- URL-only detection is weaker than a system that can safely inspect DNS, TLS, redirects, page content, and reputation feeds.
- The domain grouping is a lightweight approximation (last two hostname labels), not a public-suffix-list implementation.
- A random/domain-group holdout is not a true future-time test. For a stronger research evaluation, use a time-based split and an external, independently collected test set.
- A high ROC-AUC or accuracy is not proof of production-grade protection. Report the split method, class balance, data source, and limitations in your project report.
