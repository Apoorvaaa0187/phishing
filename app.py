from __future__ import annotations
import hashlib, ipaddress, json, math, os, re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, average_precision_score)
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / 'PhiUSIIL_Phishing_URL_Dataset.csv'
BUNDLE = ROOT / 'artifacts' / 'model_bundle.joblib'
ARTIFACTS_DIR = BUNDLE.parent
RANDOM_STATE = 42
APP_VERSION = '2.0-url-only'
app = Flask(__name__)

FEATURE_NAMES = [
    'url_length','hostname_length','path_length','query_length','dot_count',
    'subdomain_count','hyphen_count','at_count','question_count','equal_count',
    'underscore_count','percent_count','digit_count','digit_ratio','special_char_count',
    'has_ip_address','uses_https','has_suspicious_tld','has_punycode','uses_shortener',
    'has_port','path_depth','hostname_entropy','url_entropy','suspicious_word_count'
]
SUSPICIOUS_TLDS = {'zip','mov','top','xyz','click','work','country','gq','tk','ml','cf','ga','rest','fit'}
SHORTENERS = {'bit.ly','tinyurl.com','t.co','goo.gl','ow.ly','is.gd','buff.ly','cutt.ly','rb.gy','rebrand.ly'}
SUSPICIOUS_WORDS = {'login','verify','account','update','secure','signin','banking','confirm','password','wallet','support','invoice','gift','free','bonus'}


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    n = len(text)
    return float(-sum((c/n) * math.log2(c/n) for c in counts.values()))


def extract_url_features(raw_url: str) -> dict[str, float]:
    value = (raw_url or '').strip()
    if not value:
        raise ValueError('Please enter a URL.')
    candidate = value if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', value) else 'http://' + value
    parsed = urlparse(candidate)
    host = (parsed.hostname or '').lower().strip('.')
    if not host:
        raise ValueError('Please enter a valid URL with a hostname, for example https://example.com.')
    path = parsed.path or ''
    query = parsed.query or ''
    try:
        ipaddress.ip_address(host.strip('[]'))
        has_ip = 1
    except ValueError:
        has_ip = 0
    labels = host.split('.')
    tld = labels[-1] if len(labels) > 1 else ''
    digits = sum(ch.isdigit() for ch in value)
    special = sum(not ch.isalnum() for ch in value)
    host_entropy = _entropy(host)
    url_entropy = _entropy(value)
    words = re.findall(r'[a-zA-Z]+', (host + ' ' + path + ' ' + query).lower())
    try:
        port = parsed.port
    except ValueError:
        port = None
    f = {
        'url_length': len(value), 'hostname_length': len(host), 'path_length': len(path),
        'query_length': len(query), 'dot_count': value.count('.'),
        'subdomain_count': max(0, len(labels)-2), 'hyphen_count': value.count('-'),
        'at_count': value.count('@'), 'question_count': value.count('?'),
        'equal_count': value.count('='), 'underscore_count': value.count('_'),
        'percent_count': value.count('%'), 'digit_count': digits,
        'digit_ratio': digits / max(1, len(value)), 'special_char_count': special,
        'has_ip_address': has_ip, 'uses_https': int(parsed.scheme.lower() == 'https'),
        'has_suspicious_tld': int(tld in SUSPICIOUS_TLDS),
        'has_punycode': int('xn--' in host), 'uses_shortener': int(host in SHORTENERS),
        'has_port': int(port is not None), 'path_depth': len([p for p in path.split('/') if p]),
        'hostname_entropy': host_entropy, 'url_entropy': url_entropy,
        'suspicious_word_count': sum(w in SUSPICIOUS_WORDS for w in words),
    }
    return {k: float(f[k]) for k in FEATURE_NAMES}


def normalize_label(series: pd.Series) -> pd.Series:
    """PhiUSIIL convention: Label=1 legitimate, Label=0 phishing. Return 1=phishing."""
    numeric = pd.to_numeric(series, errors='coerce')
    if numeric.notna().all() and set(numeric.unique()).issubset({0, 1}):
        return (numeric == 0).astype(int)
    text = series.astype(str).str.strip().str.lower()
    mapping = {'phishing': 1, 'malicious': 1, 'bad': 1, '1': 1,
               'legitimate': 0, 'benign': 0, 'safe': 0, 'good': 0, '0': 0}
    mapped = text.map(mapping)
    if mapped.isna().any():
        raise ValueError('Could not interpret labels. Expected PhiUSIIL Label (1=legitimate, 0=phishing).')
    return mapped.astype(int)


def domain_group(raw_url: str) -> str:
    candidate = raw_url if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', str(raw_url)) else 'http://' + str(raw_url)
    host = (urlparse(candidate).hostname or '').lower()
    parts = host.split('.')
    return '.'.join(parts[-2:]) if len(parts) >= 2 else host


def load_training_data():
    if not DATASET.exists():
        raise FileNotFoundError(
            f'Dataset not found: {DATASET.name}. Download the PhiUSIIL CSV and place it beside app.py.'
        )
    df = pd.read_csv(DATASET)
    url_col = next((c for c in df.columns if c.lower() == 'url'), None)
    label_col = next((c for c in df.columns if c.lower() == 'label'), None)
    if not url_col or not label_col:
        raise ValueError('Dataset must contain columns named URL and Label.')
    data = df[[url_col, label_col]].dropna().copy()
    data[url_col] = data[url_col].astype(str).str.strip()
    data = data[data[url_col].str.len() > 0]
    data['target'] = normalize_label(data[label_col])
    data = data.drop_duplicates(subset=[url_col]).reset_index(drop=True)
    X = pd.DataFrame([extract_url_features(u) for u in data[url_col]], columns=FEATURE_NAMES)
    y = data['target'].astype(int)
    groups = data[url_col].map(domain_group)
    return X, y, groups, len(df), len(data)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    if hasattr(model, 'predict_proba'):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        scores = model.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, scores)
    # Keep chart payload compact while retaining curve shape.
    indices = np.linspace(0, len(fpr)-1, min(160, len(fpr))).astype(int)
    cm = confusion_matrix(y_test, pred, labels=[0, 1]).tolist()
    metrics = {
        'accuracy': float(accuracy_score(y_test, pred)),
        'precision': float(precision_score(y_test, pred, zero_division=0)),
        'recall': float(recall_score(y_test, pred, zero_division=0)),
        'f1': float(f1_score(y_test, pred, zero_division=0)),
        'roc_auc': float(roc_auc_score(y_test, scores)),
        'pr_auc': float(average_precision_score(y_test, scores)),
        'confusion_matrix': cm,
        'roc_curve': {'fpr': fpr[indices].tolist(), 'tpr': tpr[indices].tolist()},
    }
    return model, metrics


def train_and_evaluate():
    X, y, groups, original_rows, usable_rows = load_training_data()
    # Split by registrable-ish domain so URLs from the same site are less likely to leak across sets.
    try:
        splitter = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
        train_idx, test_idx = next(splitter.split(X, y, groups))
        if y.iloc[train_idx].nunique() < 2 or y.iloc[test_idx].nunique() < 2:
            raise ValueError('Group split produced a single-class partition.')
        split_method = '80/20 domain-group holdout (same base domain kept in one split)'
    except Exception:
        train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.20,
                                                random_state=RANDOM_STATE, stratify=y)
        split_method = '80/20 stratified random holdout (domain grouping fallback)'
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    models = {
        'Logistic Regression': make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, class_weight='balanced')),
        'HistGradientBoosting': HistGradientBoostingClassifier(max_iter=140, learning_rate=0.08, max_leaf_nodes=31, random_state=RANDOM_STATE),
        'Random Forest': RandomForestClassifier(n_estimators=220, min_samples_leaf=2, class_weight='balanced_subsample', random_state=RANDOM_STATE, n_jobs=-1),
        'Extra Trees': ExtraTreesClassifier(n_estimators=220, min_samples_leaf=2, class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1),
    }
    results, trained = {}, {}
    for name, model in models.items():
        fit_X, fit_y = X_train, y_train
        trained[name], results[name] = evaluate_model(name, model, fit_X, X_test, fit_y, y_test)
        results[name]['training_rows'] = int(len(fit_X))
    dataset_hash = hashlib.sha256(DATASET.read_bytes()).hexdigest()
    payload = {
        'app_version': APP_VERSION, 'dataset_sha256': dataset_hash,
        'dataset_name': DATASET.name, 'original_rows': int(original_rows),
        'usable_unique_urls': int(usable_rows), 'train_rows': int(len(X_train)),
        'test_rows': int(len(X_test)), 'phishing_rows': int(y.sum()),
        'legitimate_rows': int((y == 0).sum()), 'split_method': split_method,
        'positive_class': 'Phishing (1)', 'feature_names': FEATURE_NAMES,
        'metrics': results,
        'best_model': max(results, key=lambda k: results[k]['f1']),
    }
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    joblib.dump({'models': trained, 'metrics': payload, 'dataset_sha256': dataset_hash}, BUNDLE)
    return trained, payload


STATE = {'models': {}, 'metrics': None, 'error': None}



def ensure_trained(force=False):
    # If models are already loaded in memory, reuse them.
    # Avoid reloading models and hashing the dataset for every URL scan.
    if not force and STATE['metrics'] is not None and STATE['models']:
        return

    if not force and BUNDLE.exists():
        try:
            bundle = joblib.load(BUNDLE)
            current_hash = hashlib.sha256(DATASET.read_bytes()).hexdigest()

            if (
                bundle.get('dataset_sha256') == current_hash
                and bundle['metrics'].get('app_version') == APP_VERSION
            ):
                STATE.update(
                    models=bundle['models'],
                    metrics=bundle['metrics'],
                    error=None
                )
                return
        except Exception:
            pass

    models, metrics = train_and_evaluate()
    STATE.update(models=models, metrics=metrics, error=None)



@app.get('/')
def home():
    return render_template('index.html')


@app.get('/api/status')
def status():
    try:
        if STATE['metrics'] is None:
            ensure_trained()
        return jsonify({'ready': True, 'metrics': STATE['metrics']})
    except Exception as exc:
        return jsonify({'ready': False, 'error': str(exc)}), 503


@app.get('/api/metrics')
def metrics():
    try:
        ensure_trained()
        return jsonify(STATE['metrics'])
    except Exception as exc:
        return jsonify({'error': str(exc)}), 503


@app.post('/api/evaluate')
def evaluate():
    try:
        ensure_trained(force=True)
        return jsonify({'message': 'Evaluation completed from the current dataset.', 'metrics': STATE['metrics']})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


@app.post('/api/predict')
def predict():
    try:
        ensure_trained()
        body = request.get_json(silent=True) or {}
        raw_url = str(body.get('url', '')).strip()
        features = extract_url_features(raw_url)
        selected = body.get('model') or STATE['metrics']['best_model']
        if selected not in STATE['models']:
            selected = STATE['metrics']['best_model']
        row = pd.DataFrame([features], columns=FEATURE_NAMES)
        model = STATE['models'][selected]
        prediction = int(model.predict(row)[0])
        score = float(model.predict_proba(row)[0, 1]) if hasattr(model, 'predict_proba') else None
        return jsonify({'url': raw_url, 'prediction': 'Phishing' if prediction == 1 else 'Likely legitimate',
                        'is_phishing': bool(prediction), 'phishing_score': score,
                        'model': selected, 'features': features,
                        'warning': 'This is a URL-pattern screening result, not a guarantee of website safety.'})
    except Exception as exc:
        return jsonify({'error': str(exc)}), 400


if __name__ == '__main__':
    print('PhishShield starting. First evaluation may take several minutes.')
    app.run(host='127.0.0.1', port=int(os.environ.get('PORT', 8000)), debug=False)
