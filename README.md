# Writer Identification from Handwriting (IAM Top 50)

A patch-based Convolutional Neural Network that identifies the writer of a handwritten sentence, trained on the top 50 most-represented writers from the IAM Handwriting Database.

## 📊 Dataset

[IAM Handwriting Top50](https://www.kaggle.com/datasets/tejasreddy/iam-handwriting-top50) — a curated subset of the offline IAM Handwriting Database containing scanned sentence images from the 50 writers with the most samples.

- ~4,900 sentence images
- 50 unique writers
- Labels derived from the accompanying `forms_for_parsing.txt` metadata file

## 🧠 Approach

Rather than feeding full, variable-width sentence images directly into a CNN, this project uses a **patch-based approach**:

1. **Preprocessing** — grayscale conversion, height-normalized resizing, and Otsu binarization to clean up scan noise.
2. **Patch extraction** — a sliding window (113×113, stride 56) breaks each sentence into multiple fixed-size patches, which:
   - Handles variable sentence lengths without distorting images
   - Multiplies the effective size of the training set
3. **CNN classification** — a compact convolutional network (4 conv/pool blocks) predicts the writer for each patch.
4. **Majority voting** — for a full sentence, predictions across all its patches are aggregated via majority vote to produce the final writer prediction.

### Handling class imbalance

One writer had significantly more samples than the rest (692 vs. a median of ~85). This was addressed using `class_weight="balanced"` during training rather than discarding data.

### Handling overfitting

An initial deeper model overfit quickly (train accuracy 87% vs. validation accuracy plateauing at ~69%). This was resolved by:

- Reducing model capacity (added a 4th conv/pool block to shrink the flattened feature map from 18,432 → 3,200 values, and reduced the dense layer from 256 → 128 units)
- Adding `EarlyStopping` (`patience=3`, `restore_best_weights=True`) to automatically halt training at the best-performing epoch

## 📈 Results

| Metric                                      | Score     |
| ------------------------------------------- | --------- |
| Patch-level test accuracy                   | 72.3%     |
| **Sentence-level accuracy (majority vote)** | **91.5%** |

For context, random chance across 50 classes would be ~2%, and published academic writer-identification methods on comparable setups typically report results in the 77–96% range — so this result is competitive.

## 🗂️ Repo structure

```
.
├── README.md
├── requirements.txt
├── writer_identification.ipynb
└── .gitignore
```

Note: the dataset images themselves are **not** included in this repo — download them directly from the [Kaggle dataset page](https://www.kaggle.com/datasets/tejasreddy/iam-handwriting-top50) linked above.

## ▶️ Running this project

This notebook was built and run on [Kaggle](https://www.kaggle.com/) with GPU acceleration enabled.

1. Fork/open the [dataset](https://www.kaggle.com/datasets/tejasreddy/iam-handwriting-top50) on Kaggle, or download it locally.
2. Open `writer_identification.ipynb` in Kaggle (or Jupyter, adjusting file paths as needed).
3. Enable a GPU accelerator (Settings → Accelerator → GPU) for reasonable training times.
4. Run all cells top to bottom.

## 🔗 Links

- 📓 [Notebook on Kaggle](#) <!-- replace with your Kaggle notebook URL -->
- 📊 [Dataset on Kaggle](https://www.kaggle.com/datasets/tejasreddy/iam-handwriting-top50)

## 🛠️ Tech stack

- Python
- TensorFlow / Keras
- OpenCV
- pandas, scikit-learn
- Kaggle Notebooks (GPU)
