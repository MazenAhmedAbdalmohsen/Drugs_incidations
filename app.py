from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wit import Wit

app = Flask(__name__)
CORS(app)

# Load data from Dataset.json
with open('Dataset.json', encoding='utf-8') as file:
    data = json.load(file)

# Initialize the Wit.ai client
client = Wit('LMW6NAICDU3GK4IBPWQUHGZPVJAQ5CMU')

# Define a function to process the Wit.ai response and extract the intent
def process_response(response):
    if response.get('intents'):
        intent = response.get('intents')[0].get('name')
        return intent
    return None

# Define a function to search for medications based on user input
def search_medicine(user_input, search_type):
    if search_type == 'indications':
        symptoms = [medication['Indications'] for medication in data['prostate']]
        medication_names = [medication['DrugName'] for medication in data['prostate']]
    elif search_type == 'drug_names':
        symptoms = [medication['DrugName'] for medication in data['prostate']]
        medication_names = symptoms

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(symptoms)

    # Convert user input text to TF-IDF matrix
    user_tfidf = vectorizer.transform([user_input])

    # Calculate cosine similarity between user input and all indications/drug names
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix)

    # Get the indices of the top matching medications
    top_indices = np.argsort(similarity_scores, axis=1)[:, -1]

    # Get the names of the matching medications
    matching_medications = [medication_names[index] for index in top_indices]

    return matching_medications

# Define a function to get drug indications based on drug name
def get_drug_indications(drug_name):
    indications = [medication['Indications'] for medication in data['prostate']]
    matching_indications = [indication for indication, name in zip(indications, data['prostate']) if name['DrugName'] == drug_name]
    return matching_indications

@app.route('/', methods=['GET', 'POST'])
def search_medications():
    if request.method == 'POST':
        user_input = request.form['indications']
        search_type = 'indications'

        # Process the user input using Wit.ai
        response = client.message(user_input)
        intent = process_response(response)

        # Search for matching medications
        matching_medications = search_medicine(user_input, search_type)

        drug_name = matching_medications[0] if matching_medications else "Not found"

        return render_template('index.html', drug_name=drug_name)
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run()
