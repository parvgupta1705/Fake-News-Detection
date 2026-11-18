from pathlib import Path
import re
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import LinearSVC

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "news_models.joblib"

# Small illustrative seed corpus only, for making the app runnable before the research dataset is added.
DEMO_ROWS = [
("Officials release public report after council meeting", "The council published its annual report on Monday and said the document would be discussed at a public meeting next week.", "REAL"),
("Health department issues seasonal vaccination guidance", "The health department published updated guidance and advised residents to consult qualified medical professionals for individual advice.", "REAL"),
("Central bank announces policy review following scheduled meeting", "The central bank said its committee reviewed economic indicators and would publish the full minutes on its official website.", "REAL"),
("University opens applications for next academic session", "The university announced that applications will open next month. Applicants can review eligibility and deadlines on the admissions portal.", "REAL"),
("Rail service restored after maintenance work", "Rail officials said services resumed after overnight maintenance. Some delays remained and passengers were advised to check live updates.", "REAL"),
("Local authority approves road repair budget", "The local authority approved a repair budget after a debate during its monthly meeting, according to the meeting record.", "REAL"),
("Researchers publish results of a small laboratory study", "The authors said the study had limitations and that further independent research would be needed to confirm the findings.", "REAL"),
("Company reports quarterly revenue in investor filing", "The company released its quarterly filing with revenue figures and a discussion of operating costs and future risks.", "REAL"),
("Court schedules hearing in ongoing civil case", "The court listed the matter for a hearing next month. The filing contains allegations that have not been decided by the court.", "REAL"),
("Weather agency issues heavy rain advisory", "The weather agency issued an advisory for several districts and recommended following local emergency instructions.", "REAL"),
("Public library announces extended weekend hours", "The library said it will extend opening hours during the examination period and published the updated schedule online.", "REAL"),
("Election office publishes voter registration deadlines", "The election office published registration dates and instructions for checking voter details through its official service.", "REAL"),
("Experts urge caution while interpreting new survey", "Researchers noted that the survey sample was limited and said the results should not be treated as proof of causation.", "REAL"),
("Hospital opens new outpatient clinic", "Hospital administrators said the clinic will begin accepting appointments on Monday and listed services on the hospital website.", "REAL"),
("Transport agency releases annual safety statistics", "The agency published annual figures and explained that the report covers incidents recorded during the previous calendar year.", "REAL"),
("New study finds association but not proof of cause", "The study found an association in its sample, while the authors cautioned that the observational design cannot establish causality.", "REAL"),
("BREAKING: Secret miracle drink cures every disease overnight", "A hidden formula doctors refuse to reveal supposedly cures every disease overnight. Share this immediately before authorities erase the evidence.", "FAKE"),
("Government secretly gives every citizen free cash today", "Everyone is guaranteed a huge cash payment today. Click the link now and forward this message before the offer disappears forever.", "FAKE"),
("Scientists confirm the moon is made of cheese", "A viral post claims scientists confirmed the moon is made of cheese, but provides no study, institution, or verifiable evidence.", "FAKE"),
("Shocking video proves all elections are controlled by one person", "The post claims every election is controlled by one person and urges readers to share it, but cites no records or credible evidence.", "FAKE"),
("Doctors hate this one trick that reverses aging instantly", "An anonymous website promises instant age reversal and says doctors are hiding the secret. No clinical evidence or named researchers are provided.", "FAKE"),
("URGENT: Banks will delete all accounts at midnight", "A forwarded message warns that all bank accounts will be deleted at midnight unless users share their passwords through an unknown link.", "FAKE"),
("Celebrity secretly elected president of another country", "A viral headline claims a celebrity secretly became president, but offers no official announcement or reliable reporting to support it.", "FAKE"),
("Miracle device creates unlimited electricity for free", "The article claims a small device creates unlimited energy without fuel and says buyers must pay immediately to reserve one.", "FAKE"),
("Every food item is poisoned according to anonymous insider", "An anonymous source claims all food is poisoned and urges people to panic, without providing evidence, data, or identifiable sources.", "FAKE"),
("Secret law makes criticism illegal starting tomorrow", "The post claims a secret law takes effect tomorrow and threatens readers, but links to no legislation or official notice.", "FAKE"),
("Viral message claims school exams are cancelled nationwide", "A forwarded message says all exams are cancelled nationwide but gives no issuing authority, date, or official circular.", "FAKE"),
("This simple magnet makes fuel last forever", "A sales page claims a magnet permanently doubles fuel efficiency and asks readers to buy quickly, without independent test results.", "FAKE"),
("Leaked document proves every celebrity is an alien", "A sensational post claims a leaked document proves celebrities are aliens but does not provide an authentic document or verifiable source.", "FAKE"),
("Share now: one phone setting reveals everyone's private messages", "The article promises access to anyone's private messages through a secret setting and asks users to install an unknown application.", "FAKE"),
("Secret cure suppressed by all hospitals", "The post claims a single untested remedy cures serious illnesses and that all hospitals are hiding it, without clinical evidence.", "FAKE"),
("Viral claim says city water will turn people invisible", "A social media post predicts that drinking city water will make people invisible and provides no scientific evidence or credible source.", "FAKE"),
("Anonymous account claims a major bank has vanished", "A post says a major bank disappeared overnight and urges panic withdrawals, but cites no bank statement or regulator notice.", "FAKE"),
("Guaranteed investment multiplies savings one hundred times", "The article promises guaranteed returns of one hundred times in a week and demands an upfront transfer to an anonymous wallet.", "FAKE"),
]

def normalize_label(value):
    s = str(value).strip().lower()
    if s in {"fake", "false", "1", "misinformation"}: return "FAKE"
    if s in {"real", "true", "0", "reliable"}: return "REAL"
    return str(value).strip().upper()

def find_dataset():
    # Supported formats: True.csv + Fake.csv (Kaggle), or a single labelled CSV.
    true_path, fake_path = DATA_DIR / "True.csv", DATA_DIR / "Fake.csv"
    if true_path.exists() and fake_path.exists():
        real = pd.read_csv(true_path)
        fake = pd.read_csv(fake_path)
        real["label"] = "REAL"
        fake["label"] = "FAKE"
        df = pd.concat([real, fake], ignore_index=True)
        return prepare_frame(df)
    for p in [DATA_DIR / "news.csv", DATA_DIR / "dataset.csv", DATA_DIR / "fake_or_real_news.csv"]:
        if p.exists():
            return prepare_frame(pd.read_csv(p))
    return None

def prepare_frame(df):
    lower = {str(c).lower(): c for c in df.columns}
    label_col = next((lower[k] for k in ["label", "class", "target", "is_fake"] if k in lower), None)
    if not label_col:
        raise ValueError("Dataset needs a label/class/target column with REAL/FAKE labels.")
    title_col = next((lower[k] for k in ["title", "headline"] if k in lower), None)
    body_col = next((lower[k] for k in ["text", "body", "article", "content"] if k in lower), None)
    if not title_col and not body_col:
        raise ValueError("Dataset needs a title/headline and/or text/body/content column.")
    title = df[title_col].fillna("").astype(str) if title_col else ""
    body = df[body_col].fillna("").astype(str) if body_col else ""
    text = (title + " " + body).str.strip()
    labels = df[label_col].map(normalize_label)
    out = pd.DataFrame({"text": text, "label": labels})
    out = out[out["label"].isin(["REAL", "FAKE"])]
    out = out[out["text"].str.len() > 10].drop_duplicates("text")
    if out["label"].nunique() != 2:
        raise ValueError("After label normalization, dataset must contain both REAL and FAKE examples.")
    return out.reset_index(drop=True)

def build_models():
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(C=5, max_iter=2000, random_state=42),
        "Linear SVM": LinearSVC(C=1.0, random_state=42),
        "Passive-Aggressive": PassiveAggressiveClassifier(max_iter=1000, random_state=42, tol=1e-3),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=1),
    }

def fit_bundle(df, demo_mode=False):
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    vectorizer = TfidfVectorizer(
        max_features=15000, ngram_range=(1, 2), stop_words="english",
        lowercase=True, strip_accents="unicode", sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    models, metrics = {}, {}
    for name, model in build_models().items():
        model.fit(X_train_vec, y_train)
        pred = model.predict(X_test_vec)
        models[name] = model
        metrics[name] = {
            "Accuracy": round(accuracy_score(y_test, pred), 4),
            "Precision (FAKE)": round(precision_score(y_test, pred, pos_label="FAKE", zero_division=0), 4),
            "Recall (FAKE)": round(recall_score(y_test, pred, pos_label="FAKE", zero_division=0), 4),
            "F1 (FAKE)": round(f1_score(y_test, pred, pos_label="FAKE", zero_division=0), 4),
        }
    best_name = max(metrics, key=lambda n: metrics[n]["F1 (FAKE)"])
    svm_pred = models["Linear SVM"].predict(X_test_vec)
    cm = confusion_matrix(y_test, svm_pred, labels=["FAKE", "REAL"]).tolist()
    return {
        "vectorizer": vectorizer, "models": models, "best_model_name": best_name,
        "metrics": metrics, "confusion_matrix": cm, "demo_mode": demo_mode,
        "dataset_size": len(df), "test_size": len(y_test)
    }

def load_or_train_model():
    MODEL_DIR.mkdir(exist_ok=True)
    if MODEL_PATH.exists():
        try:
            return joblib.load(MODEL_PATH)
        except Exception:
            pass
    df = find_dataset()
    demo = False
    if df is None:
        df = pd.DataFrame(DEMO_ROWS, columns=["title", "body", "label"])
        df["text"] = df["title"] + " " + df["body"]
        df = df[["text", "label"]]
        demo = True
    bundle = fit_bundle(df, demo_mode=demo)
    if not demo:
        joblib.dump(bundle, MODEL_PATH)
    return bundle

def predict_article(bundle, text):
    vec = bundle["vectorizer"].transform([text])
    model = bundle["models"][bundle["best_model_name"]]
    label = model.predict(vec)[0]
    score = 50.0
    if hasattr(model, "decision_function"):
        raw = float(model.decision_function(vec)[0])
        # Display a bounded relative score, not a calibrated probability.
        score = 100 / (1 + __import__("math").exp(-max(-8, min(8, raw))))
        if label == "FAKE": score = 100 - score
    elif hasattr(model, "predict_proba"):
        probs = model.predict_proba(vec)[0]
        classes = list(model.classes_)
        score = float(probs[classes.index(label)] * 100)
    return {"label": label, "score": score}

def get_explanations(bundle, text, predicted_label, top_n=10):
    model = bundle["models"][bundle["best_model_name"]]
    vectorizer = bundle["vectorizer"]
    try:
        names = vectorizer.get_feature_names_out()
        vec = vectorizer.transform([text])
        if hasattr(model, "coef_"):
            weights = model.coef_[0]
            contributions = vec.toarray()[0] * weights
            # sklearn classes are usually FAKE, REAL sorted; coefficient positive means classes_[1].
            classes = list(model.classes_)
            direction = 1 if predicted_label == classes[1] else -1
            ranked = sorted(range(len(contributions)), key=lambda i: contributions[i] * direction, reverse=True)
            terms = []
            for i in ranked:
                if contributions[i] * direction <= 0: continue
                term = names[i]
                if term not in terms: terms.append(term)
                if len(terms) >= top_n: break
            return terms
        return [names[i] for i in vec.nonzero()[1][:top_n]]
    except Exception:
        return []

def credibility_score(text, source="", reputation="Unknown"):
    # Explicitly heuristic; not a factual verification or live reputation lookup.
    score = 50.0
    if reputation == "Established / verified": score += 20
    elif reputation == "Unverified / unfamiliar": score -= 10
    words = text.split()
    if len(words) >= 80: score += 5
    if len(words) < 20: score -= 5
    if text.count("!") >= 3: score -= 8
    if sum(1 for w in words if w.isupper() and len(w) > 2) >= 4: score -= 7
    if re.search(r"\b(guaranteed|miracle|secret cure|share immediately|they don't want you to know)\b", text, re.I):
        score -= 10
    if source.strip(): score += 3
    return max(0, min(100, score))
