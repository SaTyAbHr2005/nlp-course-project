# Smart Reading Assistant

Classifies English text as **Easy / Medium / Hard** and, if it is not Easy, rewrites it in simpler language.

- **Classifier:** ELECTRA-large hybrid (transformer `[CLS]` embedding + 10 `textstat` readability features → MLP)
- **Simplifier:** fine-tuned T5-base
- **UI:** Gradio (`app.py`)

Full write-up: [Project_Report.md](Project_Report.md)

## Download the trained models

The `.pt` weights are too large for git and are attached to the [latest release](https://github.com/SaTyAbHr2005/nlp-course-project/releases/latest).

| File | Size | Used by `app.py` |
|---|---|---|
| `electra_large_final_model.pt` | 1.34 GB | yes |
| `t5_simplifier_best.pt` | 892 MB | yes |
| `modernbert_final_model.pt` | 598 MB | no (alternative classifier) |
| `bart_simplifier_best.pt` | 558 MB | no (alternative simplifier) |
| `roberta_final_model.pt` | 501 MB | no (alternative classifier) |
| `distilbert_final_model.pt` | 267 MB | no (alternative classifier) |

Put the downloaded files in the same folder as `app.py`.

## Run

```bash
pip install -r requirements.txt
python app.py
```

Then open http://localhost:7860. The base models (`google/electra-large-discriminator`, `t5-base`) are downloaded from Hugging Face on first run. A GPU is used if available; CPU works but is slower.

## Repository layout

| Path | Purpose |
|---|---|
| `app.py` | Gradio app: classify, then simplify |
| `scaler.pkl` | `StandardScaler` fitted on the 10 numeric features (required by `app.py`) |
| `final_train/val/test.csv` | Classification dataset (text, label, 10 features) |
| `scraped_simplification_pairs.csv` | Hard → Easy parallel pairs for the simplifier |
| `scrape_news_levels.py` | Scraper that built the simplification pairs |
| `train_*_final_colab.ipynb` | Classifier training notebooks (DistilBERT, RoBERTa, ModernBERT, ELECTRA-large) |
| `step2_train_*_custom_colab.ipynb` | Simplifier training notebooks (T5, BART) |
| `test_medium.py`, `find_medium.py` | Quick inference checks |

All notebooks are written for Google Colab with a T4 GPU.

## Results

- Classifier test accuracy: 85-90% depending on the backbone
- T5 simplifier: BERTScore 0.899; Flesch-Kincaid grade 10.97 → 4.36 on average
