# High-glucose cluster replication with PCA and stacking

Coursework by **Zainab Cheema and Jiabei Liu**. This repository is an educational experiment in preprocessing, K-means pseudo-labels, PCA, and stacked classifiers. It is **not** a diabetes diagnosis or clinical risk-prediction model.

## Question and method

Can supervised classifiers reproduce the partition made by a K-means model on glucose, BMI, and age? The input CSV has no observed diabetes outcome. The label `1` means “assigned to the cluster whose center has higher glucose,” **not** “has diabetes.” Accuracy therefore measures agreement with a generated cluster label, not medical accuracy.

```text
CSV → train/test split → train-fitted imputer and scaler
    → train-fitted K-means pseudo-labels → train-fitted PCA
    → Naive Bayes / KNN / MLP → decision-tree stacker → held-out agreement
```

The split now happens before fitting imputation, scaling, K-means, or PCA, so test rows do not influence those transformations. The test labels come from `KMeans.predict` using the training-fitted clusters. The earlier coursework version fit transformations before splitting; do not compare its numbers directly with the revised pipeline. IQR outlier removal was dropped from this evaluation path because it previously used full-dataset thresholds and changed the holdout population.

## Reproduce

Use Python 3.11+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python main.py
```

`main.py` reads the included `diabetes_project.csv`; the nested grid searches and MLP may take several minutes. The dataset's original source and licensing are not documented in this repository, so verify provenance before redistributing it or claiming broader generalization. The separate `stroke_dataset_adaptation.py` is a historical coursework extension, not a validated transfer-learning experiment.

In one local run of the revised code (Python 3.13, pandas 3.0.6, scikit-learn 1.9.1), held-out pseudo-label replication accuracy was **0.9184**. This is a reproducibility reference for that environment, **not** a clinical result or a guarantee across versions and datasets.

## Interpretation and next steps

- Report the holdout number as **pseudo-label replication accuracy** only. A classifier can score well here simply because it learns the same feature partition as K-means.
- Compare against a direct K-means assignment baseline and record class balance, confusion matrix, and seed variability before making stronger claims.
- A genuine health-outcome study would require verified outcome labels, documented dataset provenance, appropriate validation, subgroup evaluation, and clinical oversight. This repository does not provide those.
