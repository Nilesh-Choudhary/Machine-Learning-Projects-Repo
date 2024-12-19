# app.py
from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Read the CSV file and shuffle the data
try:
    df = pd.read_csv("Phone_Details_Data.csv", index_col=0, dtype=str).sample(frac=1).reset_index(drop=True)
except FileNotFoundError:
    print("CSV file not found. Please make sure the file path is correct.")
    df = pd.DataFrame()  # Create an empty DataFrame if the file is not found

# Function to clean text data
def clean_text(text):
    text = re.sub("[^a-zA-Z0-9 ]", "", text)
    return text

# Clean text data in each column
cleaned_data = df.applymap(clean_text)

# Initialize TF-IDF vectorizers for each column
vectorizers = [TfidfVectorizer(ngram_range=(1,2)) for _ in range(len(df.columns))]

# Compute TF-IDF vectors for each column
tfidf_matrices = [vectorizer.fit_transform(cleaned_data.iloc[:, i]) for i, vectorizer in enumerate(vectorizers)]

# Function to perform individual column search
def individual_column_search(query):
    # Clean the query
    cleaned_query = clean_text(query)
    
    # Compute TF-IDF vector for the query
    query_vector = [vectorizer.transform([cleaned_query]) for vectorizer in vectorizers]
    
    # Calculate cosine similarity between query and each column
    similarities = [cosine_similarity(query_vec, tfidf_matrix).flatten() for query_vec, tfidf_matrix in zip(query_vector, tfidf_matrices)]
    
    # Find the index of the column with the highest similarity
    max_similarity_index = np.argmax([max(sim) for sim in similarities])
    
    # Get the top matching records from the most similar column
    max_sim_column = similarities[max_similarity_index]
    sorted_indices = np.argsort(-max_sim_column)
    top_records = df.iloc[sorted_indices[:10], :].copy()  # Copying the DataFrame to preserve original order
    
    # Reorder DataFrame columns to prioritize the matching column first
    columns_sorted = [df.columns[max_similarity_index]] + [col for col in df.columns if col != df.columns[max_similarity_index]]
    top_records = top_records[columns_sorted]
    
    # Reorder DataFrame so that matching records appear at the top
    top_records.insert(0, 'Similarity', max_sim_column[sorted_indices[:10]])
    top_records.sort_values(by='Similarity', ascending=False, inplace=True)
    top_records.drop(columns=['Similarity'], inplace=True)
    
    # Constructing the description sentence for top matching records
    def construct_description(row):
        description = ""
        for col in top_records.columns:
            if col == 'Description':
                continue
            if col in row.index:
                description += f"{get_sentence(col, row[col])} "
        if 'Description' in row.index:
            description += row['Description']
        return description
    
    top_records['Description'] = top_records.apply(construct_description, axis=1)
    
    return top_records

def get_sentence(column_name, value):
    if column_name == 'Battery':
        return f"Experience unmatched endurance with the {value} battery of the"
    elif column_name == 'Phone_Name':
        return f"Get seamless performance with {value}"
    elif column_name == 'RAM':
        return f"Enjoy swift multitasking with {value}"
    elif column_name == 'Display':
        return f"Immerse yourself in a captivating viewing experience on the expansive {value} display of the"
    elif column_name == 'Camera':
        return f"Capture every detail with stunning clarity using the {value} camera of the"
    elif column_name == 'Warranty':
        return f"Enjoy peace of mind with {value} warranty included with the"
    elif column_name == 'Price':
        return f"All these premium features come at an unbeatable value of {value} "
    else:
        return ""

@app.route('/', methods=['GET', 'POST'])
def index():
    search_result = pd.DataFrame()  # Initialize search_result as an empty DataFrame
    
    if request.method == 'POST':
        query = request.form['query']
        search_result = individual_column_search(query)
    
    return render_template('index.html', search_result=search_result)

if __name__ == '__main__':
    app.run(debug=True)
