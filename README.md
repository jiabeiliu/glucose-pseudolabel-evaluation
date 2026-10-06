# Glucose pseudo-label evaluation

## Reproducible demo

[View the read-only demo page](https://jiabeiliu.github.io/glucose-pseudolabel-evaluation/) for the experiment flow and a recorded local result. The page does not train a model in the browser or predict diabetes; run the commands below to execute the actual pipeline.

Run `python main.py` after installing `requirements.txt`. In the verified local run (Python 3.13, pandas 3.0.6, scikit-learn 1.9.1), the console ended with:

```text
Best KNN: {'n_neighbors': 7, 'weights': 'uniform'}
Best NN: {'alpha': 0.0001, 'hidden_layer_sizes': (10,)}
Best Meta Learner (DecisionTree): {'max_depth': 2, 'min_samples_split': 2}
Pseudo-label replication accuracy (not diagnostic accuracy): 0.9184
```

This is a demonstration of reproducing K-means-generated labels, **not** a diabetes prediction score. The [original DAMG6105 coursework repository](https://github.com/jiabeiliu/DAMG6105-final-project) is retained as a historical snapshot; use this repository for the corrected pipeline and tests. Both repositories contain the same underlying CSV data (with different filenames), so they are one project, not two independent studies.

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
