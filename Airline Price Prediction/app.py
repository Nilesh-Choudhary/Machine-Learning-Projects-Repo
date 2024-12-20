from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib


app = Flask(__name__)

# Load the model
model = joblib.load('random_forest_regressor.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get form data
        airline = request.form['airline']
        source = request.form['source']
        destination = request.form['destination']
        date_of_journey = request.form['date_of_journey']
        dep_time = request.form['dep_time']
        arrival_time = request.form['arrival_time']
        total_stops = int(request.form['total_stops'])
        duration = request.form['duration']
        
        # Process the date_of_journey
        journey_day = int(date_of_journey.split('-')[2])
        journey_month = int(date_of_journey.split('-')[1])
        
        # Process the dep_time
        dep_hour = int(dep_time.split(':')[0])
        dep_min = int(dep_time.split(':')[1])
        
        # Process the arrival_time
        arrival_hour = int(arrival_time.split(':')[0])
        arrival_min = int(arrival_time.split(':')[1])
        
        # Process the duration
        duration_hours = int(duration.split(':')[0])
        duration_mins = int(duration.split(':')[1])
        
        # Create the input dataframe for the model
        input_data = pd.DataFrame({
            'Total_Stops': [total_stops],
            'Journey_day': [journey_day],
            'Journey_month': [journey_month],
            'Dep_hour': [dep_hour],
            'Dep_min': [dep_min],
            'Arrival_hour': [arrival_hour],
            'Arrival_min': [arrival_min],
            'Duration_hours': [duration_hours],
            'Duration_mins': [duration_mins],
            'Airline_Air India': [1 if airline == 'Air India' else 0],
            'Airline_GoAir': [1 if airline == 'GoAir' else 0],
            'Airline_IndiGo': [1 if airline == 'IndiGo' else 0],
            'Airline_Jet Airways': [1 if airline == 'Jet Airways' else 0],
            'Airline_Jet Airways Business': [1 if airline == 'Jet Airways Business' else 0],
            'Airline_Multiple carriers': [1 if airline == 'Multiple carriers' else 0],
            'Airline_Multiple carriers Premium economy': [1 if airline == 'Multiple carriers Premium economy' else 0],
            'Airline_SpiceJet': [1 if airline == 'SpiceJet' else 0],
            'Airline_Trujet': [1 if airline == 'Trujet' else 0],
            'Airline_Vistara': [1 if airline == 'Vistara' else 0],
            'Airline_Vistara Premium economy': [1 if airline == 'Vistara Premium economy' else 0],
            'Source_Chennai': [1 if source == 'Chennai' else 0],
            'Source_Delhi': [1 if source == 'Delhi' else 0],
            'Source_Kolkata': [1 if source == 'Kolkata' else 0],
            'Source_Mumbai': [1 if source == 'Mumbai' else 0],
            'Destination_Cochin': [1 if destination == 'Cochin' else 0],
            'Destination_Delhi': [1 if destination == 'Delhi' else 0],
            'Destination_Hyderabad': [1 if destination == 'Hyderabad' else 0],
            'Destination_Kolkata': [1 if destination == 'Kolkata' else 0],
            'Destination_New Delhi': [1 if destination == 'New Delhi' else 0],
        })

        # Make prediction
        prediction = model.predict(input_data)[0]

        return render_template('index.html', prediction_text=f'Estimated Flight Price: ₹{prediction:.2f}')

if __name__ == "__main__":
    app.run(debug=True)


