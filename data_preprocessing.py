"""Leakage-aware preprocessing for the coursework pseudo-label experiment."""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def preprocess_and_pca(filepath):
    """Fit transformations on training rows only, then make cluster pseudo-labels.

    These labels indicate membership in the higher-glucose cluster. They are
    not observed diabetes diagnoses and must not be described as such.
    """
    frame = pd.read_csv(filepath)
    features = frame.select_dtypes(include="number").drop(columns=["Outcome"], errors="ignore")
    required = {"Glucose", "BMI", "Age"}
    if not required.issubset(features.columns):
        raise ValueError(f"Input needs numeric columns: {', '.join(sorted(required))}")
    if len(features) < 10:
        raise ValueError("At least 10 rows are needed for this train/test experiment")

    train_raw, test_raw = train_test_split(features, test_size=0.2, random_state=42)
    imputer = SimpleImputer(strategy="median")
    train_imputed = imputer.fit_transform(train_raw)
    test_imputed = imputer.transform(test_raw)
    scaler = MinMaxScaler()
    train_scaled = pd.DataFrame(scaler.fit_transform(train_imputed), columns=features.columns)
    test_scaled = pd.DataFrame(scaler.transform(test_imputed), columns=features.columns)

    cluster_columns = ["Glucose", "BMI", "Age"]
    clusterer = KMeans(n_clusters=2, random_state=42, n_init=10)
    train_clusters = clusterer.fit_predict(train_scaled[cluster_columns])
    test_clusters = clusterer.predict(test_scaled[cluster_columns])
    higher_glucose_cluster = int(np.argmax(clusterer.cluster_centers_[:, 0]))
    y_train = (train_clusters == higher_glucose_cluster).astype(int)
    y_test = (test_clusters == higher_glucose_cluster).astype(int)

    pca = PCA(n_components=3)
    x_train = pd.DataFrame(pca.fit_transform(train_scaled), columns=["PC1", "PC2", "PC3"])
    x_test = pd.DataFrame(pca.transform(test_scaled), columns=["PC1", "PC2", "PC3"])
    return x_train, x_test, y_train, y_test
