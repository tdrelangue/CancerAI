from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
import numpy as np
import pandas as pd
import joblib

def train_decision_tree(dataset, save_path="random_forest_model.pkl"):
    # Extract features and labels
    X = dataset.trainingSet[0]  # Features
    y = dataset.trainingSet[1]  # Labels (Multi-label)

    # Convert to DataFrame for readability
    df_X = pd.DataFrame(X, columns=dataset.column_names[:-3])  # Features
    df_y = pd.DataFrame(y, columns=dataset.column_names[-3:])  # Labels

    # Initialize MultiOutputClassifier with Random Forest
    base_rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model = MultiOutputClassifier(base_rf)

    # Train the model
    rf_model.fit(df_X, df_y)

    # ✅ Save the trained model
    joblib.dump(rf_model, save_path)
    print(f"✅ Model saved at: {save_path}")


def load_model(path="random_forest_model.pkl"):
    """Load the trained model and retrieve feature importances with column names."""
    # Load the saved model
    rf_model = joblib.load(path)

    # ✅ Extract feature names from the first estimator (assuming they are the same across all)
    column_names = rf_model.estimators_[0].feature_names_in_

    # ✅ Extract feature importances from all estimators and average them
    feature_importance_list = np.mean([est.feature_importances_ for est in rf_model.estimators_], axis=0)

    # ✅ Create a DataFrame for feature importance ranking
    feature_ranking = pd.DataFrame({"Feature": column_names, "Importance": feature_importance_list})

    # Sort by importance score (descending)
    feature_ranking = feature_ranking.sort_values(by="Importance", ascending=False)

    # Print top 15 features
    print("\n🔹 Feature Importance Ranking:")
    print(feature_ranking.head(15))

    return rf_model, feature_ranking

# Run the function
rf_model, feature_ranking = load_model()