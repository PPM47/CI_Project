from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)
model = joblib.load('model.pkl')  # Load your trained model

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        # Read the CSV file into a DataFrame
        data = pd.read_csv(file)
        
        # Make predictions (assuming the model expects the same features as the CSV columns)
        predictions = model.predict(data)
        
        # Convert predictions to a list for easy JSON serialization
        return jsonify({'prediction': predictions.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
