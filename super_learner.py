import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score

def run_super_learner(X_train_pca, y_train, X_test_pca, y_test):
    print("🔍 Running GridSearchCV to tune hyperparameters...")

    # GaussianNB has no hyperparameters to tune
    nb = GaussianNB()

    # GridSearch for KNN
    knn_grid = GridSearchCV(KNeighborsClassifier(), {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }, cv=5)
    knn_grid.fit(X_train_pca, y_train)
    knn = knn_grid.best_estimator_
    print(f"✅ Best KNN: {knn_grid.best_params_}")

    # GridSearch for Neural Network
    nn_grid = GridSearchCV(MLPClassifier(max_iter=1000, random_state=42), {
        'hidden_layer_sizes': [(10,), (50,), (100,)],
        'alpha': [0.0001, 0.001, 0.01]
    }, cv=5)
    nn_grid.fit(X_train_pca, y_train)
    nn = nn_grid.best_estimator_
    print(f"✅ Best NN: {nn_grid.best_params_}")

    # Get cross-validated predictions for base models
    nb_preds = cross_val_predict(nb, X_train_pca, y_train, cv=5, method='predict')
    knn_preds = cross_val_predict(knn, X_train_pca, y_train, cv=5, method='predict')
    nn_preds = cross_val_predict(nn, X_train_pca, y_train, cv=5, method='predict')

    # Stack predictions for meta learner
    meta_features = pd.DataFrame({
        'NB': nb_preds,
        'KNN': knn_preds,
        'NN': nn_preds
    })

    # GridSearch for meta learner (Decision Tree)
    meta_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), {
        'max_depth': [2, 4, 6, 8],
        'min_samples_split': [2, 5, 10]
    }, cv=5)
    meta_grid.fit(meta_features, y_train)
    meta_learner = meta_grid.best_estimator_
    print(f"✅ Best Meta Learner (DecisionTree): {meta_grid.best_params_}")

    # Refit base models on full training set
    nb.fit(X_train_pca, y_train)
    knn.fit(X_train_pca, y_train)
    nn.fit(X_train_pca, y_train)

    # Predict on test set
    test_meta_features = pd.DataFrame({
        'NB': nb.predict(X_test_pca),
        'KNN': knn.predict(X_test_pca),
        'NN': nn.predict(X_test_pca)
    })
    final_predictions = meta_learner.predict(test_meta_features)

    accuracy = accuracy_score(y_test, final_predictions)
    print(f"\nPseudo-label replication accuracy (not diagnostic accuracy): {accuracy:.4f}")
    return accuracy
