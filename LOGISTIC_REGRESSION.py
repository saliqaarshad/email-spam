import pandas as pd
import numpy as np
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import streamlit as st

# Load and process the data
data = pd.read_csv('enron_spam_data.csv')
st.write('Data Loaded successfully')
st.write('Processing Data....')

data.dropna(inplace=True)

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s]', '', text)
    return text

data['Message'] = data['Message'].apply(preprocess_text)
data['Subject'] = data['Subject'].apply(preprocess_text)

# Initialize TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)

# Extract features
X_message = vectorizer.fit_transform(data['Message']).toarray()
X_subject = vectorizer.fit_transform(data['Subject']).toarray()

# Combine features for message and subject
X_combined = np.hstack((X_message, X_subject))
y = data['Spam/Ham'].apply(lambda x: 1 if x == 'spam' else 0).values

# Define feature sets
x_features = {
    'Message': X_message,
    'Subject': X_subject,
    'Combined': X_combined
}

accuracy = []
conf_matrix = []
class_reports = []
precision = []
recall = []
f1 = []

st.write('Processed data successfully')
st.write('Training Model....')

# Train and evaluate the model for each feature set
for feature_name, x in x_features.items():
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    dt = LogisticRegression(max_iter=1000)
    dt.fit(X_train, y_train)

    y_pred_dt = dt.predict(X_test)
    accuracy.append(accuracy_score(y_test, y_pred_dt))
    conf_matrix.append(confusion_matrix(y_test, y_pred_dt))
    
    # Get the classification report as a dictionary
    class_report = classification_report(y_test, y_pred_dt, output_dict=True)
    class_reports.append(class_report)
    
    # Extract precision, recall, and f1 scores for the '1' (spam) class
    precision.append(class_report['1']['precision'])
    recall.append(class_report['1']['recall'])
    f1.append(class_report['1']['f1-score'])

# Determine the best feature set
best_index = accuracy.index(max(accuracy))
best_feature = list(x_features.keys())[best_index]

# Display results using Streamlit
st.write(f'Best Feature Set: {best_feature}')
st.write(f'Accuracy: {accuracy[best_index]:.2f}')
st.write('Confusion Matrix:')
st.write(conf_matrix[best_index])

# Convert the best classification report to a DataFrame for clean display
report_df = pd.DataFrame(class_reports[best_index]).transpose()

# Display classification report as a DataFrame
st.write('Classification Report:')
st.dataframe(report_df)

# Display precision, recall, and f1 score
st.write(f'Precision: {precision[best_index]:.2f}')
st.write(f'Recall: {recall[best_index]:.2f}')
st.write(f'F1 Score: {f1[best_index]:.2f}')
