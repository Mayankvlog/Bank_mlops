import streamlit as st 
import pandas as pd
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import mlflow
import mlflow.sklearn
import pickle
import logging

# Set up logging
logging.basicConfig(filename='app.log', level=logging.INFO)

# Load banking dataset
data = pd.read_csv('data/banking.csv')

# Set page layout
st.set_page_config(
    page_title="Banking MLOps Project",
    page_icon="💰",
    layout="wide",
)

# Sidebar for user input parameters
st.sidebar.header('Input')

def user_input_features():
    age = st.sidebar.slider('Age', min_value=18, max_value=100, value=35)
    job = st.sidebar.selectbox('Job', ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'])
    marital = st.sidebar.selectbox('Marital Status', ['divorced', 'married', 'single'])
    education = st.sidebar.selectbox('Education', ['primary', 'secondary', 'tertiary', 'unknown'])
    default = st.sidebar.selectbox('Default', ['no', 'yes'])
    balance = st.sidebar.number_input('Balance', min_value=-10000, max_value=100000, value=0)
    housing = st.sidebar.selectbox('Housing Loan', ['no', 'yes'])
    loan = st.sidebar.selectbox('Personal Loan', ['no', 'yes'])
    contact = st.sidebar.selectbox('Contact Type', ['cellular', 'telephone', 'unknown'])
    day = st.sidebar.slider('Day', min_value=1, max_value=31, value=15)
    month = st.sidebar.selectbox('Month', ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'])
    duration = st.sidebar.slider('Duration (seconds)', min_value=0, max_value=5000, value=300)
    campaign = st.sidebar.slider('Campaign', min_value=1, max_value=50, value=5)
    pdays = st.sidebar.slider('Pdays', min_value=-1, max_value=500, value=-1)
    previous = st.sidebar.slider('Previous', min_value=0, max_value=50, value=0)
    poutcome = st.sidebar.selectbox('Poutcome', ['failure', 'other', 'success', 'unknown'])

    deposit = None  # We won't ask the user for deposit in the input form as it's the target variable

    data = {
        'age': age,
        'job': job,
        'marital': marital,
        'education': education,
        'default': default,
        'balance': balance,
        'housing': housing,
        'loan': loan,
        'contact': contact,
        'day': day,
        'month': month,
        'duration': duration,
        'campaign': campaign,
        'pdays': pdays,
        'previous': previous,
        'poutcome': poutcome,
        'deposit': deposit
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Main app content
st.write("""
# Banking MLOps Project
This app predicts whether a client will subscribe to a term deposit based on user input parameters!
""")

# Sidebar - User input features
df = user_input_features()

# Display user input parameters
st.subheader('User Input Parameters')
st.write(df)

# Drop rows with missing values
data.dropna(inplace=True)

# Split data into features and target
X = data.drop(columns=['deposit'])
y = data['deposit']

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Models
models = {
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'KNN': KNeighborsClassifier()
}

# User selects a model
selected_model = st.sidebar.selectbox('Select Model', list(models.keys()))

# Initialize the selected model
model = models[selected_model]

# Train the model
model.fit(X_train, y_train)

# Ensure input features match training features
df = pd.get_dummies(df, drop_first=True)
df = df.reindex(columns=X.columns, fill_value=0)

# Make predictions
prediction = model.predict(df)

# Display prediction
st.subheader('Predicted Deposit Subscription')
st.write(prediction[0])

# Save the trained model using pickle
model_filename = f'models/{selected_model.lower()}_model.pkl'
with open(model_filename, 'wb') as model_file:
    pickle.dump(model, model_file)


# Convert predicted value to numerical format
prediction_numeric = 1 if prediction[0] == 'yes' else 0

# Log parameters and metrics with MLflow
experiment_name = "banking_deposit_prediction"

# Set the experiment
mlflow.set_experiment(experiment_name)

# Log parameters and metrics with MLflow
if mlflow.active_run() is None:
   with mlflow.start_run():
      mlflow.log_param("Model", selected_model)
      mlflow.log_metric("PredictedDepositSubscription", prediction_numeric)  # Convert to int before logging

      # Log to file
      logging.info("Prediction: {prediction_numeric")

# Visualization
st.subheader("Visualization")

# Countplot for Deposit Subscription
st.subheader("Deposit Subscription Countplot:")
deposit_countplot = px.histogram(data, x='deposit', title='Deposit Subscription Countplot')
st.plotly_chart(deposit_countplot)

# Bar Plot for Job vs. Deposit Subscription
st.subheader("Bar Plot: Job vs. Deposit Subscription")
job_deposit_barplot = px.bar(data, x='job', y='deposit', title='Job vs. Deposit Subscription', color='deposit')
st.plotly_chart(job_deposit_barplot)

# Bar Plot for Marital Status vs. Deposit Subscription
st.subheader("Bar Plot: Marital Status vs. Deposit Subscription")
marital_deposit_barplot = px.bar(data, x='marital', y='deposit', title='Marital Status vs. Deposit Subscription', color='deposit')
st.plotly_chart(marital_deposit_barplot)

# Scatter Plot for Age vs. Balance
st.subheader("Scatter Plot: Age vs. Balance")
age_balance_scatterplot = px.scatter(data, x='age', y='balance', title='Age vs. Balance', color='deposit')
st.plotly_chart(age_balance_scatterplot)

# Save plots as artifacts
deposit_countplot.write_html("html/deposit_countplot.html")
mlflow.log_artifact("html/deposit_countplot.html")

job_deposit_barplot.write_html("html/job_deposit_barplot.html")
mlflow.log_artifact("html/job_deposit_barplot.html")

marital_deposit_barplot.write_html("html/marital_deposit_barplot.html")
mlflow.log_artifact("html/marital_deposit_barplot.html")

age_balance_scatterplot.write_html("html/age_balance_scatterplot.html")
mlflow.log_artifact("html/age_balance_scatterplot.html")

# Link to DAGsHub Repository
st.markdown("[DAGsHub Repository](https://dagshub.com/Mayankvlog/Bank_mlops.mlflow)")