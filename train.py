from src.pipeline import find_dataset, fit_bundle, MODEL_DIR, MODEL_PATH, DEMO_ROWS
import pandas as pd
import joblib

def main():
    df = find_dataset()
    if df is None:
        raise SystemExit(
            "No research dataset found. Put True.csv and Fake.csv in data/ (recommended), "
            "or put a labelled CSV named news.csv there. See README.md."
        )
    print(f"Loaded {len(df):,} unique labelled articles.")
    print("Label counts:")
    print(df["label"].value_counts().to_string())
    bundle = fit_bundle(df, demo_mode=False)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print("\nHeld-out test results (80/20 split):")
    print(pd.DataFrame(bundle["metrics"]).T.to_string())
    print(f"\nSelected model: {bundle['best_model_name']}")
    print(f"Saved model bundle to: {MODEL_PATH}")

if __name__ == "__main__":
    main()
