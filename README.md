# TruthLens — AI-Based Fake News & Misinformation Detection

A runnable research-project implementation inspired by the paper **AI-Based Fake News and Misinformation Detection System**. It uses a lightweight TF-IDF + classical machine-learning pipeline with five benchmark classifiers, a Streamlit interface, model evaluation, influential-term explanations, and a clearly labelled heuristic credibility score.

> **Important:** This repository is an implementation of the paper's described methodology, not a guarantee of reproducing its reported results. The paper reports 94.08% accuracy and 0.941 F1 for Linear SVM on its 6,335-article dataset. Your results will depend on the exact dataset, preprocessing, label mapping, split, and hyperparameters. Do not claim the paper's metrics as your run's results unless independently reproduced.

## Features

- Headline + article text input
- TF-IDF features with unigram/bigram range and up to 15,000 features
- Five classifiers: Multinomial Naive Bayes, Logistic Regression, Linear SVM, Passive-Aggressive, Random Forest
- Held-out evaluation: accuracy, precision, recall, F1
- Linear-model influential-term explanation
- Confusion matrix for Linear SVM
- Lightweight source/linguistic credibility heuristic
- Demo mode so the UI runs before adding the research dataset
- Modular code and a reproducible random seed

## Quick start (Windows)

Open PowerShell or a terminal in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

If PowerShell blocks activation, run the install and app commands using the environment's Python directly:

```powershell
py -m pip install -r requirements.txt
py -m streamlit run app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`).

## Quick start (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Train on the paper's dataset

The paper describes 6,335 articles, with 3,171 REAL and 3,164 FAKE labels. Obtain the dataset from the source cited in the paper, respecting its license. Place the CSVs in `data/`:

```text
data/
├── True.csv
└── Fake.csv
```

This format is supported for the commonly used Kaggle-style corpus. Then run:

```bash
python train.py
```

The script combines title and body, removes empty/duplicate entries, creates a stratified 80/20 train/test split, fits the TF-IDF vectorizer on the training portion only, evaluates all five models, prints the metrics, and saves the trained bundle under `models/news_models.joblib`. The model artifact is ignored by Git by default; regenerate it after cloning by running `python train.py`.

A single CSV is also supported as `data/news.csv`, `data/dataset.csv`, or `data/fake_or_real_news.csv`, with:
- a label column: `label`, `class`, `target`, or `is_fake`
- a text column: `text`, `body`, `article`, or `content`, and optionally `title` or `headline`

Label values accepted include REAL/FAKE and common True/False or 0/1 conventions. Check your dataset's label convention before training.

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. In Streamlit Community Cloud, choose **Create app** and select the repository, branch, and `app.py`.
3. Deploy. The app starts in demo mode if no trained model or dataset is present.
4. For real research inference, arrange for the training dataset to be available in the deployment environment and run `python train.py` as part of your deployment workflow, or commit a permitted model artifact if its size and license allow it.

## Repository structure

```text
fake_news_detection_system/
├── app.py                 # Streamlit web application
├── train.py               # Train/evaluate and save model bundle
├── requirements.txt
├── src/
│   ├── __init__.py
│   └── pipeline.py        # Dataset loading, models, prediction, explanations
├── data/
│   └── README.md          # Dataset placement instructions
├── models/                # Generated model bundle (not committed)
├── .gitignore
└── README.md
```

## How the system works

1. **Input:** user supplies headline and article body.
2. **Text representation:** TF-IDF vectorizer extracts unigram and bigram features, with English stop-word removal and a 15,000-feature cap.
3. **Classification:** five classifiers are trained independently and evaluated on the same held-out test set.
4. **Model selection:** the model with the highest held-out F1 for the FAKE class is selected for the app.
5. **Explanation:** for linear models, the app ranks present terms by their signed contribution to the selected class.
6. **Credibility heuristic:** optional source label and simple linguistic signals produce a separate heuristic score. This is not a real-time source verification service.

## Evaluation and research integrity

- The app deliberately does not display the paper's reported accuracy as if it were produced by this code.
- Demo mode uses a tiny illustrative corpus only to make the interface testable; its predictions and metrics are not research-valid.
- TF-IDF models learn patterns correlated with labels. They do not independently establish whether a real-world claim is true.
- The credibility score is a heuristic and is not backed by a source reputation database.
- If reproducing the paper, document the exact dataset version, split, preprocessing and hyperparameters in your report.
- The paper's confusion-matrix narrative and metric table should be checked against one another before claiming exact reproduction.

## Suggested GitHub description

`Lightweight fake-news text classification using TF-IDF, five classical ML models, explainability, evaluation metrics, and a Streamlit web interface.`

## License

Add a license file only after deciding how you want to license your original code. Dataset licensing is separate; do not redistribute data unless permitted.
