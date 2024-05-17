import pytest
import pandas as pd
from app import user_input_features
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

def test_user_input_features():
    # Simulate user inputs matching the default values in the Streamlit app
    inputs = {
        'age': 35,
        'job': 'admin.',
        'marital': 'divorced',
        'education': 'primary',
        'default': 'no',
        'balance': 0,
        'housing': 'no',
        'loan': 'no',
        'contact': 'cellular',
        'day': 15,
        'month': 'jan',
        'duration': 300,
        'campaign': 5,
        'pdays': -1,
        'previous': 0,
        'poutcome': 'failure'
    }

    # Create a sidebar input dataframe
    df = user_input_features()

    # Check that the dataframe has the correct structure
    for key, value in inputs.items():
        assert df[key][0] == value

def test_load_data():
    # Check that the data is loaded correctly
    data = pd.read_csv('data/banking.csv')
    assert not data.empty, "The data should not be empty"
    assert 'deposit' in data.columns, "The 'deposit' column should be present in the dataset"

def test_model_training():
    # Check if models can be trained without errors
    data = pd.read_csv('data/banking.csv')
    data.dropna(inplace=True)
    X = data.drop(columns=['deposit'])
    y = data['deposit']
    X = pd.get_dummies(X, drop_first=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Random Forest': RandomForestClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=200),  # Increase max_iter to avoid convergence issues
        'SVM': SVC(),
        'KNN': KNeighborsClassifier()
    }

    for model_name, model in models.items():
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        assert score > 0.5, f"The {model_name} model should have a score greater than 0.5"

if __name__ == "__main__":
    pytest.main()
