from flask import Flask, render_template, request, jsonify
import subprocess  # To run the ML model script

app = Flask(__name__)

# Route to render the main HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle ML predictions via AJAX
@app.route('/predict', methods=['POST'])
def predict():
    # Example of running an ML script
    result = subprocess.run(['python', 'ml_model.py'], capture_output=True, text=True)
    output = result.stdout.strip()  # Strip any unnecessary whitespace
    
    # Return the output as JSON to the frontend
    return jsonify({'prediction': output})

if __name__ == '__main__':
    app.run(debug=True)
