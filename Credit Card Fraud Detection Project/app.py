from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib

# Initialize the Flask app
app = Flask(__name__)

# Load pre-trained model, scaler, and encoder
model = joblib.load('cc_rf_model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoder.pkl')

# Load data to extract unique values for dropdowns
data = pd.read_csv(r"C:\Users\ASUS\Documents\Prwatech Project work\Credit card fraud\cc_category_data.csv")

# Extract unique values for dropdowns
unique_values = {
    'merchant': sorted(data['merchant'].unique()),
    'category': sorted(data['category'].unique()),
    'gender': sorted(data['gender'].unique()),
    'city': sorted(data['city'].unique()),
    'state': sorted(data['state'].unique()),
    'job': sorted(data['job'].unique())
}

# Define the correct feature order used in model training
FEATURES = [
    'cc_num', 'merchant', 'category', 'amt', 'gender', 'city', 'state',
    'lat', 'long', 'city_pop', 'job', 'merch_lat', 'merch_long',
    'month', 'day', 'hour', 'birth_month', 'age', 'zip'
]

def preprocess_input(data):
    """Preprocess user input for model prediction."""
    try:
        # Convert transaction date to datetime and extract features
        date = pd.to_datetime(data.pop('date'))
        data['month'] = date.month
        data['day'] = date.day
        data['hour'] = date.hour

        # Encode categorical features with label encoder
        categorical_cols = ['merchant', 'category', 'gender', 'city', 'state', 'job']
        for col in categorical_cols:
            if data[col] not in encoder.classes_:
                data[col] = -1  # Assign -1 for unseen values
            else:
                data[col] = encoder.transform([data[col]])[0]

        # Create a DataFrame for prediction
        input_df = pd.DataFrame(data, index=[0])

        # Ensure all features are present with correct order
        input_df = input_df.reindex(columns=FEATURES, fill_value=0)

        # Scale numerical features
        scaled_data = scaler.transform(input_df)

        return scaled_data

    except Exception as e:
        # Log error without exposing to the user
        print(f"Error in preprocessing: {e}")
        return None  # Return None if there is an error

@app.route('/')
def home():
    return render_template('index.html', unique_values=unique_values)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect input from the form
        input_data = {
            'cc_num': float(request.form['cc_num']),
            'merchant': request.form['merchant'],
            'category': request.form['category'],
            'amt': float(request.form['amt']),
            'gender': request.form['gender'],
            'city': request.form['city'],
            'state': request.form['state'],
            'lat': float(request.form['lat']),
            'long': float(request.form['long']),
            'city_pop': int(request.form['city_pop']),
            'job': request.form['job'],
            'merch_lat': float(request.form['merch_lat']),
            'merch_long': float(request.form['merch_long']),
            'date': request.form['date'],
            'birth_month': int(request.form['birth_month']),
            'age': int(request.form['age']),
            'zip': int(request.form['zip'])
        }

        # Preprocess the input data
        input_features = preprocess_input(input_data)

        if input_features is not None:
            # Make prediction
            prediction = model.predict(input_features)[0]
            result = "Transaction is fraudulent" if prediction == 1 else "Transaction is not fraudulent"
        else:
            result = "Transaction is not fraudulent"  # Default response for any processing issues

        # Redirect to the result page with prediction result
        return redirect(url_for('result', prediction=result))

    except Exception:
        # Redirect to the result page with default response if an error occurs
        return redirect(url_for('result', prediction="Transaction is not fraudulent"))

@app.route('/result')
def result():
    # Get prediction result from query parameters
    prediction = request.args.get('prediction', "Transaction is not fraudulent")
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
