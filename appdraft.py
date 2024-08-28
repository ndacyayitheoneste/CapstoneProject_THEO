import pandas as pd
from flask import Flask, request, render_template_string, send_file
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import os

# Load the dataset
data = pd.read_csv('Crop_recommendation.csv')

# Preprocess the data
X = data.drop('label', axis=1)
y = data['label']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")

# Save the model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Create a Flask app
app = Flask(__name__, static_folder='static')

# Load the trained model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define HTML template
HTML_TEMPLATE = '''
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Crop Alternation</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script>
      document.addEventListener('DOMContentLoaded', function() {
        const districts = {
          'Northern Province': ['Burera', 'Gakenke', 'Gicumbi', 'Musanze', 'Rulindo'],
          'Southern Province': ['Gisagara', 'Huye', 'Kamonyi', 'Muhanga', 'Nyamagabe', 'Nyanza', 'Nyaruguru', 'Ruhango'],
          'Eastern Province': ['Bugesera', 'Gatsibo', 'Kayonza', 'Kirehe', 'Ngoma', 'Nyagatare', 'Rwamagana'],
          'Western Province': ['Karongi', 'Ngororero', 'Nyabihu', 'Nyamasheke', 'Rubavu', 'Rusizi', 'Rutsiro'],
          'City of Kigali': ['Gasabo', 'Kicukiro', 'Nyarugenge']
        };
        const provinceSelect = document.getElementById('Province');
        const districtSelect = document.getElementById('District');
        provinceSelect.addEventListener('change', function() {
          const selectedProvince = provinceSelect.value;
          const options = districts[selectedProvince] || [];
          districtSelect.innerHTML = '<option value="" disabled selected>Select District</option>';
          options.forEach(function(district) {
            const option = document.createElement('option');
            option.value = district;
            option.textContent = district;
            districtSelect.appendChild(option);
          });
        });
        provinceSelect.dispatchEvent(new Event('change'));
      });
    </script>
  </head>
  <style>
    h1 {
        color: mediumseagreen;
        text-align: center;
    }

    .warning {
        color: red;
        font-weight: bold;
        text-align: center;
    }
    .card{
        margin-left:410px;
        margin-top: 20px;
        color: white;
    }
    .container{
        background:#7d8e7f;
        font-weight: bold;
        padding-bottom:10px;
        border-radius: 15px;
    }
  </style>
  <body style="background:#BCBBB8">
    <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
      <img height="50" width="50" src="{{ url_for('static', filename='ALL_Users/images/logo2.png') }}" alt="Logo">
      <div class="container-fluid">
        <a class="navbar-brand" href="{{ url_for('home') }}">Crop Alternation</a>
        <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
          <span class="navbar-toggler-icon"></span>
        </button>
        <div class="collapse navbar-collapse" id="navbarSupportedContent">
          <ul class="navbar-nav me-auto mb-2 mb-lg-0">
            <li class="nav-item">
              <a class="nav-link active" aria-current="page" href="{{ url_for('static', filename='ALL_Users/myindex.html') }}">Home</a>
            </li>
          </ul>
          <ul class="navbar-nav me-auto mb-2 mb-lg-0">
            <li class="nav-item">
              <a class="nav-link active" aria-current="page" href="{{ url_for('view_performance') }}">Forecasted Crop Aggregates</a>
            </li>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;

            <li class="nav-item">
              <a class="nav-link active" aria-current="page" href="{{ url_for('view_district_performance') }}">District Crop Insights</a>
            </li>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            <li class="nav-item">
              <a class="nav-link active" aria-current="page" href="{{ url_for('view_features') }}">view_features</a>
            </li>
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
            <li class="nav-item">
              <a class="nav-link active" aria-current="page" href="{{ url_for('visualize') }}">Next_Time_Forecast</a>
            </li>

          </ul>
        </div>
      </div>
    </nav>

    <div class="container my-3 mt-3">
      <h1 class="text-success">Crop Alternation System<span class="text-success"></span></h1>
      <form action="/predict" method="POST">
        <div class="row">
          <div class="col-md-4">
            <label for="Province">Province</label>
            <select id="Province" name="Province" class="form-control">
              <option value="" disabled selected>Select Province</option>
              <option value="Northern Province">Northern Province</option>
              <option value="Southern Province">Southern Province</option>
              <option value="Eastern Province">Eastern Province</option>
              <option value="Western Province">Western Province</option>
              <option value="City of Kigali">City of Kigali</option>
            </select>
          </div>
          <div class="col-md-4">
            <label for="District">District</label>
            <select id="District" name="District" class="form-control">
              <option value="" disabled selected>Select District</option>
            </select>
          </div>
          <div class="col-md-4">
            <label for="Nitrogen">Nitrogen</label>
            <input type="number" id="Nitrogen" name="Nitrogen" placeholder="Enter Nitrogen" class="form-control" required step="0">
          </div>
        </div>
        <div class="row mt-4">
          <div class="col-md-4">
            <label for="Phosporus">Phosphorus</label>
            <input type="number" id="Phosporus" name="Phosporus" placeholder="Enter Phosphorus" class="form-control" required step="0.01">
          </div>
          <div class="col-md-4">
            <label for="Potassium">Potassium</label>
            <input type="number" id="Potassium" name="Potassium" placeholder="Enter Potassium" class="form-control" required step="0.01">
          </div>
           <div class="col-md-4">
            <label for="pH">pH</label>
            <input type="number" step="0.01" id="Ph" name="Ph" placeholder="Enter pH value" class="form-control" required>
          </div>
        </div>
        <div class="row mt-4">
          <div class="col-md-4">
            <label for="Temperature">Temperature</label>
            <input type="number" step="0.01" id="Temperature" name="Temperature" placeholder="Enter Temperature in °C" class="form-control" required>
          </div>
          <div class="col-md-4">
            <label for="Humidity">Humidity</label>
            <input type="number" step="0.01" id="Humidity" name="Humidity" placeholder="Enter Humidity in %" class="form-control" required>
          </div>
          <div class="col-md-4">
            <label for="Rainfall">Rainfall</label>
            <input type="number" step="0.01" id="Rainfall" name="Rainfall" placeholder="Enter Rainfall in mm" class="form-control" required>
          </div>
        </div>
        <button type="submit" class="btn btn-primary mt-4">Predict</button>
      </form>

      {% if result %}
        <div class="card bg-dark" style="width: 18rem;">
          <img src="{{url_for('static', filename='ALL_Users/images/img.jpg')}}" class="card-img-top" alt="Prediction">
          <div class="card-body">
            <h5 class="card-title">Based on the input values,</h5>
            <p class="card-text"> The Alternated crop is: <strong>{{ result }}</strong></p>
            <a href="{{ url_for('generate_pdf', result=result) }}" class="btn btn-secondary">Download PDF Report</a>
          </div>
        </div>
      {% endif %}
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
  </body>
</html>
'''


@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    province = request.form.get('Province')
    district = request.form.get('District')
    nitrogen = float(request.form.get('Nitrogen'))
    phosphorus = float(request.form.get('Phosporus'))
    potassium = float(request.form.get('Potassium'))
    temperature = float(request.form.get('Temperature'))
    humidity = float(request.form.get('Humidity'))
    rainfall = float(request.form.get('Rainfall'))
    pH = float(request.form.get('Ph'))

    # Prepare input for prediction
    input_data = np.array([[nitrogen, phosphorus, potassium, temperature, humidity, rainfall, pH]])
    input_data = scaler.transform(input_data)

    # Make prediction
    prediction = model.predict(input_data)
    result = prediction[0]

    return render_template_string(HTML_TEMPLATE, result=result)


@app.route('/generate_pdf')
def generate_pdf():
    result = request.args.get('result', 'No result')
    file_path = 'static/report.pdf'

    c = canvas.Canvas(file_path, pagesize=letter)
    width, height = letter

    c.drawString(100, height - 100, 'Crop Alternation Report')
    c.drawString(100, height - 140, f'The alternated crop is: {result}')

    c.save()

    return send_file(file_path, as_attachment=True, attachment_filename='crop_report.pdf')


if __name__ == '__main__':
    app.run(debug=True)
