# Dataset setup

The paper describes 6,335 labelled articles (3,171 REAL and 3,164 FAKE), based on the Fake or Real News corpus referenced in the paper.

For training, place your permitted dataset files in this folder:
- `True.csv` — real articles
- `Fake.csv` — fake articles

The app also accepts a single labelled CSV named `news.csv`, `dataset.csv`, or `fake_or_real_news.csv`. It must contain a label/class/target column with REAL/FAKE (or True/False) values and a title/headline and/or text/body/content column.

Dataset files are excluded from Git by default because datasets may be large and have separate licensing/redistribution terms. Follow the dataset provider's license and usage requirements.
