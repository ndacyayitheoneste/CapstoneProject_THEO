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
import os
from flask import Flask, request, send_file
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter





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
app = Flask(__name__)

# Load the trained model and scaler
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define path for static files
app.config['STATIC_FOLDER'] = 'static'

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

        <div class="row mt-4">
          <div class="col-md-12 text-center">
            <button type="submit" class="btn btn-primary btn-lg">Get Alternation</button>
          </div>
        </div>
      </form>

      {% if result %}
        <div class="card bg-dark" style="width: 18rem;">
          <img src="{{url_for('static', filename='ALL_Users/images/img.jpg')}}" class="card-img-top" alt="Prediction">
          <div class="card-body">
            <h5 class="card-title">Based on the input values,</h5>
            <p class="card-text"> The Alternated crop is: <strong>{{ result }}</strong></p>
          </div>
        </div>
      {% endif %}
    </div>
      
  <img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo"><img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo"><img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo"><img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo"><img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo"><img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo"><img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo"><img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo"><img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo"><img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo"><img height="94" width="215" src="{{ url_for('static', filename='ALL_Users/images/img4.jpg') }}" alt="Logo"><img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo"><img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo"><img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo"><img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo"><img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo"><img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo"><img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo"><img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo"><img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo"><img height="94" width="50" src="{{ url_for('static', filename='ALL_Users/images/crop02.png') }}" alt="Logo">
   
  </body>
</html>
'''


@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Extract form data
        try:
            features = [
                float(request.form['Nitrogen']),
                float(request.form['Phosporus']),
                float(request.form['Potassium']),
                float(request.form['Temperature']),
                float(request.form['Humidity']),
                float(request.form['Rainfall']),
                float(request.form['Ph']),
            ]
            # Scale the features and predict
            scaled_features = scaler.transform([features])
            prediction = model.predict(scaled_features)[0]

            return render_template_string(HTML_TEMPLATE, result=prediction)

        except Exception as e:
            return str(e)


@app.route('/view_performance')
def view_performance():
    # Generate the bar plot
    fig, ax = plt.subplots(figsize=(24, 10))  # Adjust the size as needed (width, height)
    ax.bar(['rice', 'maize', 'chickpea', 'kidneybean', 'pigeonpeas', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
            'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton',
            'jute', 'coffee'],
           [75, 96, 85, 64, 78, 11, 65, 91, 74, 92, 13, 53, 71, 79, 15, 20, 54, 48, 85, 32, 82])
    ax.set_xlabel('Crops')
    ax.set_ylabel('Performance')

    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    # Encode the image data in base64 to pass it to the template
    import base64
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return render_template_string('''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Features for Crop Prediction</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
              body { background: #BCBBB8; }
              .container { background: #7d8e7f; font-weight: bold; padding: 20px; border-radius: 15px; }
              h1 { color: mediumseagreen; text-align: center; }
              img { display: block; margin-left: auto; margin-right: auto; }
            </style>
        </head>
        <body>
            <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <img height="50" width="50" src="{{ url_for('static', filename='ALL_Users/images/logo2.png') }}" alt="Logo">
                <div class="container-fluid">
                    <a class="navbar-brand" href="{{ url_for('home') }}">Crop Alternation</a>
                    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                        <span class="navbar-toggler-icon"></span>
                    </button>
                    <div class="collapse navbar-collapse" id="navbarSupportedContent">
                        <ul class="navbar-nav me-auto mb-2 mb-lg-0">
                            <li class="nav-item"><a class="nav-link active" href="{{ url_for('home') }}">Go_Back</a></li>
                            <li class="nav-item"><a class="nav-link active" href="{{ url_for('view_performance') }}">Performances</a></li>
                        </ul>
                    </div>
                </div>
            </nav>
            <div class="container my-3">
                <h2><b>The Key Elements considered to view the Aggregate of Alternated Crops</b></h2>

                <ul class="list-group">
                    {% for feature in features %}
                        <li class="list-group-item">{{ feature }}</li>
                    {% endfor %}
                </ul>

                <img src="data:image/png;base64,{{ image_base64 }}" alt="Performance Plot">

            </div>
        </body>
        </html>
    ''', image_base64=image_base64, features=['rice', 'maize', 'chickpea', 'kidneybean', 'pigeonpeas', 'mungbean', 'blackgram', 'lentil', 'pomegranate',
            'banana', 'mango', 'grapes', 'watermelon', 'muskmelon', 'apple', 'orange', 'papaya', 'coconut', 'cotton',
            'jute', 'coffee'])


@app.route('/view_district_performance')
def view_district_performance():
    districts = ['Ngoma', 'Nyamasheke', 'Musanze', 'Kamonyi', 'Nyanza', 'Gasabo']
    crops = ['rice', 'maize', 'chickpea', 'kidneybean', 'pigeonpeas', 'mungbean', 'blackgram']

    performance_data = {
        'rice': [70, 12, 75, 99, 71, 10],
        'maize': [98, 34, 85, 74, 45, 76],
        'chickpea': [34, 96, 36, 45, 67, 73],
        'kidneybean': [25, 17, 85, 76, 99, 51],
        'pigeonpeas': [60, 56, 97, 82, 20, 77],
        'mungbean': [75, 11, 85, 49, 35, 39],
        'blackgram': [68, 78, 13, 62, 47, 99],
    }

    styles = {
        'rice': {'color': 'blue', 'linestyle': 'solid', 'marker': 'o'},
        'maize': {'color': 'green', 'linestyle': 'dashed', 'marker': 's'},
        'chickpea': {'color': 'orange', 'linestyle': 'dotted', 'marker': '^'},
        'kidneybean': {'color': 'red', 'linestyle': 'dashdot', 'marker': 'D'},
        'pigeonpeas': {'color': 'purple', 'linestyle': 'solid', 'marker': 'p'},
        'mungbean': {'color': 'brown', 'linestyle': 'dashed', 'marker': '*'},
        'blackgram': {'color': 'cyan', 'linestyle': 'dotted', 'marker': 'x'},
    }

    # Create the radar plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    # Store the most predicted crop for each district
    most_predicted_crops = {}

    for i, district in enumerate(districts):
        max_performance = 0
        best_crop = None

        # Find the best crop for the current district
        for crop, values in performance_data.items():
            if values[i] > max_performance:
                max_performance = values[i]
                best_crop = crop

        most_predicted_crops[district] = best_crop

        # Plot each crop's performance
        for crop, values in performance_data.items():
            style = styles[crop]
            district_values = values + [values[0]]
            ax.plot(districts + [districts[0]], district_values, label=crop,
                    color=style['color'], linestyle=style['linestyle'], marker=style['marker'])
            ax.fill(districts + [districts[0]], district_values, color=style['color'], alpha=0.1)

    # Add labels and title
    ax.set_title('District Performance by Crop', size=20, color='mediumseagreen', y=1.1)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Save the radar plot to a BytesIO object
    buf_radar = io.BytesIO()
    plt.savefig(buf_radar, format='png')
    buf_radar.seek(0)
    plt.close(fig)

    # Encode the image in base64
    image_base64 = base64.b64encode(buf_radar.getvalue()).decode('utf-8')

    # Log the most predicted crops for each district
    print("Most predicted crops by district:")
    for district, crop in most_predicted_crops.items():
        print(f"{district}: {crop}")

    # Render the template with the image and feature list
    return render_template_string('''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Features for Crop Prediction</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
              body { background: #BCBBB8; }
              .container { background: #7d8e7f; font-weight: bold; padding: 20px; border-radius: 15px; }
              h1 { color: mediumseagreen; text-align: center; }
              img { display: block; margin-left: auto; margin-right: auto; }
            </style>
        </head>
        <body>
            <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <img height="50" width="50" src="{{ url_for('static', filename='ALL_Users/images/logo2.png') }}" alt="Logo">
                <div class="container-fluid">
                    <a class="navbar-brand" href="{{ url_for('home') }}">Crop Alternation</a>
                    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                        <span class="navbar-toggler-icon"></span>
                    </button>
                    <div class="collapse navbar-collapse" id="navbarSupportedContent">
                    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                        <ul class="navbar-nav me-auto mb-2 mb-lg-0">
                            <li class="nav-item"><a class="nav-link active" href="{{ url_for('home') }}">Go_Back</a></li>
                            <li class="nav-item"><a class="nav-link active" href="{{ url_for('view_district_performance') }}">District Crop Insights</a></li>
                            <li class="nav-item"><a class="nav-link active" href="{{ url_for('download_system_report') }}">System_Report</a></li>
                            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
                            <li>
                        <a href="{{ url_for('download_image') }}" class="btn btn-primary my-3">Save Image</a>
                        </li>
                        </ul>
                        
                    </div>
                </div>
            </nav>
            <div class="container my-3">
                <h2><b>District Performances for Crop Alternation</b></h2>

                <ul class="list-group">
                    {% for feature in features %}
                        <li class="list-group-item">{{ feature }}</li>
                    {% endfor %}
                </ul>

                <img src="data:image/png;base64,{{ image_base64 }}" alt="District_performance Plot">

                

            </div>
        </body>
        </html>
    ''', image_base64=image_base64, features=['rice', 'maize', 'chickpea', 'kidneybean', 'pigeonpeas', 'mungbean', 'blackgram'])

@app.route('/download_image')
def download_image():
    districts = ['Ngoma', 'Nyamasheke', 'Musanze', 'Kamonyi', 'Nyanza', 'Gasabo']
    crops = ['rice', 'maize', 'chickpea', 'kidneybean', 'pigeonpeas', 'mungbean', 'blackgram']

    performance_data = {
        'rice': [70, 12, 75, 99, 71, 10],
        'maize': [98, 34, 85, 74, 45, 76],
        'chickpea': [34, 96, 36, 45, 67, 73],
        'kidneybean': [25, 17, 85, 76, 99, 51],
        'pigeonpeas': [60, 56, 97, 82, 20, 77],
        'mungbean': [75, 11, 85, 49, 35, 39],
        'blackgram': [68, 78, 13, 62, 47, 99],
    }

    styles = {
        'rice': {'color': 'blue', 'linestyle': 'solid', 'marker': 'o'},
        'maize': {'color': 'green', 'linestyle': 'dashed', 'marker': 's'},
        'chickpea': {'color': 'orange', 'linestyle': 'dotted', 'marker': '^'},
        'kidneybean': {'color': 'red', 'linestyle': 'dashdot', 'marker': 'D'},
        'pigeonpeas': {'color': 'purple', 'linestyle': 'solid', 'marker': 'p'},
        'mungbean': {'color': 'brown', 'linestyle': 'dashed', 'marker': '*'},
        'blackgram': {'color': 'cyan', 'linestyle': 'dotted', 'marker': 'x'},
    }

    # Create the radar plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    for i, district in enumerate(districts):
        # Plot each crop's performance
        for crop, values in performance_data.items():
            style = styles[crop]
            district_values = values + [values[0]]
            ax.plot(districts + [districts[0]], district_values, label=crop,
                    color=style['color'], linestyle=style['linestyle'], marker=style['marker'])
            ax.fill(districts + [districts[0]], district_values, color=style['color'], alpha=0.1)

    # Add labels and title
    ax.set_title('District Performance by Crop', size=20, color='mediumseagreen', y=1.1)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Save the radar plot to a BytesIO object
    buf_radar = io.BytesIO()
    plt.savefig(buf_radar, format='png')
    buf_radar.seek(0)
    plt.close(fig)

    return send_file(buf_radar, as_attachment=True, download_name='district_performance_plot.png', mimetype='image/png')

@app.route('/download_pdf')
def download_pdf():
    buf_pdf = io.BytesIO()
    c = canvas.Canvas(buf_pdf, pagesize=letter)
    width, height = letter

    c.drawString(72, height - 72, "District Performance Report")
    c.drawString(72, height - 100, "Performance data for different crops across districts.")

    c.drawString(72, height - 140, "Districts:")
    for i, district in enumerate(['Ngoma', 'Nyamasheke', 'Musanze', 'Kamonyi', 'Nyanza', 'Gasabo']):
        c.drawString(72, height - 160 - (i * 20), f"{district}")

    c.drawString(72, height - 220, "Crops:")
    for i, crop in enumerate(['rice', 'maize', 'chickpea', 'kidneybean', 'pigeonpeas', 'mungbean', 'blackgram']):
        c.drawString(72, height - 240 - (i * 20), f"{crop}")

    c.drawString(72, height - 300, "Performance Data:")

    c.showPage()
    c.save()

    buf_pdf.seek(0)
    return send_file(buf_pdf, as_attachment=True, download_name='district_performance_report.pdf', mimetype='application/pdf')

@app.route('/download_system_report')
def download_system_report():
    buf_pdf = io.BytesIO()
    c = canvas.Canvas(buf_pdf, pagesize=letter)
    width, height = letter

    c.drawString(72, height - 72, "System Report")
    c.drawString(72, height - 100, "Detailed Crop Performance by District")

    c.drawString(72, height - 140, "District")
    c.drawString(150, height - 140, "Rice")
    c.drawString(220, height - 140, "Maize")
    c.drawString(290, height - 140, "Chickpea")
    c.drawString(360, height - 140, "Kidneybean")
    c.drawString(430, height - 140, "Pigeonpeas")
    c.drawString(500, height - 140, "Mungbean")
    c.drawString(570, height - 140, "Blackgram")

    districts = ['Ngoma', 'Nyamasheke', 'Musanze', 'Kamonyi', 'Nyanza', 'Gasabo']
    crops = ['rice', 'maize', 'chickpea', 'kidneybean', 'pigeonpeas', 'mungbean', 'blackgram']
    performance_data = {
        'rice': [70, 12, 75, 99, 71, 10],
        'maize': [98, 34, 85, 74, 45, 76],
        'chickpea': [34, 96, 36, 45, 67, 73],
        'kidneybean': [25, 17, 85, 76, 99, 51],
        'pigeonpeas': [60, 56, 97, 82, 20, 77],
        'mungbean': [75, 11, 85, 49, 35, 39],
        'blackgram': [68, 78, 13, 62, 47, 99],
    }

    y_position = height - 160
    for i, district in enumerate(districts):
        c.drawString(72, y_position, district)
        for j, crop in enumerate(crops):
            c.drawString(150 + j * 70, y_position, str(performance_data[crop][i]))
        y_position -= 20

    c.showPage()
    c.save()

    buf_pdf.seek(0)
    return send_file(buf_pdf, as_attachment=True, download_name='system_report.pdf', mimetype='application/pdf')


    # Return the image as a response
    #return Response(buf_radar, mimetype='image/png')



@app.route('/view_features')
def view_features():
    features = [
        'Nitrogen',
        'Phosphorus',
        'Potassium',
        'Temperature',
        'Humidity',
        'Rainfall',
        'pH'
    ]
    return render_template_string('''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Features for Crop Prediction</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
              body { background: #BCBBB8; }
              .container { background: #7d8e7f; font-weight: bold; padding: 20px; border-radius: 15px; }
              h1 { color: mediumseagreen; text-align: center; }
            </style>
        </head>
        <body>
            <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <img height="50" width="50" src="{{ url_for('static', filename='ALL_Users/images/logo2.png') }}" alt="Logo">
                <div class="container-fluid">
                    <a class="navbar-brand" href="{{ url_for('home') }}">Crop Alternation</a>
                    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                        <span class="navbar-toggler-icon"></span>
                    </button>
                    <div class="collapse navbar-collapse" id="navbarSupportedContent">
                        <ul class="navbar-nav me-auto mb-2 mb-lg-0">
                            <li class="nav-item"><a class="nav-link active" href="{{ url_for('home') }}">Go_Back</a></li>
                            <li class="nav-item"><a class="nav-link active" href="{{ url_for('view_features') }}">Features</a></li>
                        </ul>
                    </div>
                </div>
            </nav>
            <div class="container my-3">
                <h2><b>Key Features Considered for Crop Alternation</b></h2>
                
                <ul class="list-group">
                                
                    {% for feature in features %}
                        <li class="list-group-item">{{ feature }}</li>
                    {% endfor %}
                    
                </ul>
                
            </div>
        </body>
        </html>
    ''', features=features)




@app.route('/visualization', methods=['GET', 'POST'])
def visualize():
    districts = ['Ngoma', 'Nyamasheke', 'Musanze', 'Kamonyi', 'Nyanza', 'Gasabo']
    crops = ['rice', 'maize', 'chickpea', 'kidneybean', 'pigeonpeas', 'mungbean', 'blackgram']

    # Performance data for each year
    performance_data_2024 = {
        'rice': [70, 12, 75, 99, 71, 10],
        'maize': [98, 34, 85, 74, 45, 76],
        'chickpea': [34, 96, 36, 45, 67, 73],
        'kidneybean': [25, 17, 85, 76, 99, 51],
        'pigeonpeas': [60, 56, 97, 82, 20, 77],
        'mungbean': [75, 11, 85, 49, 35, 39],
        'blackgram': [68, 78, 13, 62, 47, 99],
    }

    performance_data_2025 = {
        'rice': [65, 15, 70, 95, 68, 15],
        'maize': [90, 40, 80, 70, 50, 70],
        'chickpea': [30, 90, 35, 50, 65, 70],
        'kidneybean': [30, 20, 80, 72, 95, 55],
        'pigeonpeas': [55, 50, 90, 80, 25, 70],
        'mungbean': [70, 15, 80, 45, 40, 45],
        'blackgram': [65, 80, 10, 60, 50, 95],
    }

    performance_data_2026 = {
        'rice': [60, 20, 65, 90, 65, 20],
        'maize': [85, 45, 75, 65, 55, 65],
        'chickpea': [25, 85, 30, 55, 60, 65],
        'kidneybean': [35, 25, 75, 70, 90, 60],
        'pigeonpeas': [50, 45, 85, 75, 30, 65],
        'mungbean': [65, 20, 75, 40, 45, 50],
        'blackgram': [60, 85, 5, 55, 55, 90],
    }

    performance_data_2027 = {
        'rice': [58, 25, 60, 88, 62, 25],
        'maize': [80, 50, 70, 60, 60, 60],
        'chickpea': [20, 80, 25, 60, 55, 60],
        'kidneybean': [40, 30, 70, 65, 85, 65],
        'pigeonpeas': [45, 40, 80, 70, 35, 60],
        'mungbean': [60, 25, 70, 35, 50, 55],
        'blackgram': [55, 90, 10, 50, 60, 85],
    }

    performance_data_2028 = {
        'rice': [55, 30, 55, 85, 60, 30],
        'maize': [75, 55, 65, 55, 65, 55],
        'chickpea': [18, 75, 20, 65, 50, 55],
        'kidneybean': [45, 35, 65, 60, 80, 70],
        'pigeonpeas': [40, 35, 75, 65, 40, 55],
        'mungbean': [55, 30, 65, 30, 55, 50],
        'blackgram': [50, 95, 15, 45, 65, 80],
    }

    performance_data_2029 = {
        'rice': [52, 35, 50, 82, 58, 35],
        'maize': [70, 60, 60, 50, 70, 50],
        'chickpea': [16, 70, 15, 70, 45, 50],
        'kidneybean': [50, 40, 60, 55, 75, 75],
        'pigeonpeas': [35, 30, 70, 60, 45, 50],
        'mungbean': [50, 35, 60, 25, 60, 45],
        'blackgram': [45, 100, 20, 40, 70, 75],
    }

    performance_data_2030 = {
        'rice': [50, 40, 45, 80, 55, 40],
        'maize': [65, 65, 55, 45, 75, 45],
        'chickpea': [14, 65, 10, 75, 40, 45],
        'kidneybean': [55, 45, 55, 50, 70, 80],
        'pigeonpeas': [30, 25, 65, 55, 50, 45],
        'mungbean': [45, 40, 55, 20, 65, 40],
        'blackgram': [40, 105, 25, 35, 75, 70],
    }

    performance_data_2031 = {
        'rice': [48, 45, 40, 78, 52, 45],
        'maize': [60, 70, 50, 40, 80, 40],
        'chickpea': [12, 60, 5, 80, 35, 40],
        'kidneybean': [60, 50, 50, 45, 65, 85],
        'pigeonpeas': [25, 20, 60, 50, 55, 40],
        'mungbean': [40, 45, 50, 15, 70, 35],
        'blackgram': [35, 110, 30, 30, 80, 65],
    }

    performance_data_2032 = {
        'rice': [46, 50, 35, 75, 50, 50],
        'maize': [55, 75, 45, 35, 85, 35],
        'chickpea': [10, 55, 0, 85, 30, 35],
        'kidneybean': [65, 55, 45, 40, 60, 90],
        'pigeonpeas': [20, 15, 55, 45, 60, 35],
        'mungbean': [35, 50, 45, 10, 75, 30],
        'blackgram': [30, 115, 35, 25, 85, 60],
    }

    performance_data_2033 = {
        'rice': [44, 55, 30, 72, 48, 55],
        'maize': [50, 80, 40, 30, 90, 30],
        'chickpea': [8, 50, 0, 90, 25, 30],
        'kidneybean': [70, 60, 40, 35, 55, 95],
        'pigeonpeas': [15, 10, 50, 40, 65, 30],
        'mungbean': [30, 55, 40, 5, 80, 25],
        'blackgram': [25, 120, 40, 20, 90, 55],
    }

    performance_data_2034 = {
        'rice': [42, 60, 25, 70, 45, 60],
        'maize': [45, 85, 35, 25, 95, 25],
        'chickpea': [6, 45, 0, 95, 20, 25],
        'kidneybean': [75, 65, 35, 30, 50, 100],
        'pigeonpeas': [10, 5, 45, 35, 70, 25],
        'mungbean': [25, 60, 35, 0, 85, 20],
        'blackgram': [20, 125, 45, 15, 95, 50],
    }

    performance_data_2035 = {
        'rice': [40, 65, 20, 68, 42, 65],
        'maize': [40, 90, 30, 20, 100, 20],
        'chickpea': [5, 40, 0, 100, 15, 20],
        'kidneybean': [80, 70, 30, 25, 45, 105],
        'pigeonpeas': [5, 0, 40, 30, 75, 20],
        'mungbean': [20, 65, 30, 0, 90, 15],
        'blackgram': [15, 130, 50, 10, 100, 45],
    }

    # Determine which year's data to use
    year = request.form.get('year', '2024')

    if year == '2024':
        performance_data = performance_data_2024
    elif year == '2025':
        performance_data = performance_data_2025
    elif year == '2026':
        performance_data = performance_data_2026
    elif year == '2027':
        performance_data = performance_data_2027
    elif year == '2028':
        performance_data = performance_data_2028
    elif year == '2029':
        performance_data = performance_data_2029
    elif year == '2030':
        performance_data = performance_data_2030
    elif year == '2031':
        performance_data = performance_data_2031
    elif year == '2032':
        performance_data = performance_data_2032
    elif year == '2033':
        performance_data = performance_data_2033
    elif year == '2034':
        performance_data = performance_data_2034
    elif year == '2035':
        performance_data = performance_data_2035
    else:
        performance_data = performance_data_2024  # Default to 2024 if year is not recognized

    styles = {
        'rice': {'color': 'blue', 'linestyle': 'solid', 'marker': 'o'},
        'maize': {'color': 'green', 'linestyle': 'dashed', 'marker': 's'},
        'chickpea': {'color': 'orange', 'linestyle': 'dotted', 'marker': '^'},
        'kidneybean': {'color': 'red', 'linestyle': 'dashdot', 'marker': 'D'},
        'pigeonpeas': {'color': 'purple', 'linestyle': 'solid', 'marker': 'p'},
        'mungbean': {'color': 'brown', 'linestyle': 'dashed', 'marker': '*'},
        'blackgram': {'color': 'cyan', 'linestyle': 'dotted', 'marker': 'x'},
    }

    # Create the radar plot
    fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))

    for crop, values in performance_data.items():
        style = styles[crop]
        district_values = values + [values[0]]
        ax.plot(districts + [districts[0]], district_values, label=crop,
                color=style['color'], linestyle=style['linestyle'], marker=style['marker'])
        ax.fill(districts + [districts[0]], district_values, color=style['color'], alpha=0.1)

    # Add labels and title
    ax.set_title(f'District Performance by Crop for {year}', size=20, color='mediumseagreen', y=1.1)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))

    # Save the radar plot to a BytesIO object
    buf_radar = io.BytesIO()
    plt.savefig(buf_radar, format='png')
    buf_radar.seek(0)
    plt.close(fig)

    # Encode the image in base64
    image_base64 = base64.b64encode(buf_radar.getvalue()).decode('utf-8')

    # Render the template with the image and year selection
    return render_template_string('''
        <!doctype html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Features for Crop Prediction</title>
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
              body { background: #BCBBB8; }
              .container { background: #7d8e7f; font-weight: bold; padding: 20px; border-radius: 15px; }
              h1 { color: mediumseagreen; text-align: center; }
              img { display: block; margin-left: auto; margin-right: auto; }
            </style>
        </head>
        <body>
            <nav class="navbar navbar-expand-lg navbar-dark bg-dark">
            <img height="50" width="50" src="{{ url_for('static', filename='ALL_Users/images/logo2.png') }}" alt="Logo">
                <div class="container-fluid">
                    <a class="navbar-brand" href="{{ url_for('home') }}">Crop Alternation</a>
                    <button class="navbar-toggler" type="button" data-bs-toggle="collapse" data-bs-target="#navbarSupportedContent" aria-controls="navbarSupportedContent" aria-expanded="false" aria-label="Toggle navigation">
                        <span class="navbar-toggler-icon"></span>
                    </button>
                    <div class="collapse navbar-collapse" id="navbarSupportedContent">
                        <ul class="navbar-nav me-auto mb-2 mb-lg-0">
                            <li class="nav-item"><a class="nav-link active" href="{{ url_for('home') }}">Go_Back</a></li>
                            <li class="nav-item"><a class="nav-link active" href="{{ url_for('visualize') }}">Visualizations</a></li>
                        </ul>
                    </div>
                </div>
            </nav>
            <div class="container my-3">
                <h2><b>District Performances Visualization Over Next 11 Years</b></h2>

                <form method="post">
                    <div class="form-group">
                        <label for="yearSelect">Select Year:</label>
                        <select class="form-control" id="yearSelect" name="year">
                            <option value="2024" {% if year == '2024' %}selected{% endif %}>2024</option>
                            <option value="2025" {% if year == '2025' %}selected{% endif %}>2025</option>
                            <option value="2026" {% if year == '2026' %}selected{% endif %}>2026</option>
                            <option value="2027" {% if year == '2027' %}selected{% endif %}>2027</option>
                            <option value="2028" {% if year == '2028' %}selected{% endif %}>2028</option>
                            <option value="2029" {% if year == '2029' %}selected{% endif %}>2029</option>
                            <option value="2030" {% if year == '2030' %}selected{% endif %}>2030</option>
                            <option value="2031" {% if year == '2031' %}selected{% endif %}>2031</option>
                            <option value="2032" {% if year == '2032' %}selected{% endif %}>2032</option>
                            <option value="2033" {% if year == '2033' %}selected{% endif %}>2033</option>
                            <option value="2034" {% if year == '2034' %}selected{% endif %}>2034</option>
                            <option value="2035" {% if year == '2035' %}selected{% endif %}>2035</option>
                        </select>
                    </div>
                    <button type="submit" class="btn btn-primary my-3">View Performance</button>
                </form>

                <img src="data:image/png;base64,{{ image_base64 }}" alt="Radar Plot">
            </div>
            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
        </body>
        </html>
    ''', image_base64=image_base64, year=year)




if __name__ == '__main__':
    app.run(debug=True)

