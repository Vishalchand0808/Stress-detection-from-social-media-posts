# stress-detection-text-representations
Binary stress-vs-non-stress classification on Reddit posts, comparing six text-representation pipelines in one notebook.

## Method overview
| Pipeline | Vectoriser | Classifier |
|----------|------------|------------|
| Bag-of-Words | CountVectorizer | Logistic Regression |
| TF-IDF | TfidfVectorizer | Logistic Regression |
| Word2Vec CBOW (self-train) | gensim | LR on mean-pooled vectors |
| Word2Vec Skip-Gram (self-train) | gensim | LR |
| Word2Vec CBOW (pre-trained) | Google-News 300 d | LR |
| Word2Vec (pre-trained → fine-tuned) | Google-News 300 d | LR |

## Quick start (Colab)
1. Open **Stress_Detection.ipynb** in Colab.  
2. Upload your `kaggle.json` when prompted – notebook auto-downloads Dreaddit train/test CSVs.  
3. `Runtime ▸ Run all` – plots and metrics appear at the end.

## Dataset
* **Dreaddit** – 3 555 Reddit posts (2 839 train / 716 test) with “stressed” vs “not stressed” labels.

## Main result
Fine-tuned Word2Vec + LR achieves **≈ 0.75 accuracy**; classical BoW/TF-IDF remain competitive (~0.72).

## Requirements (local run)
```
pandas
scikit-learn
gensim==4.3.3
nltk
seaborn
matplotlib
```
