import matplotlib.pyplot as plt
import io
import base64
from flask import Response
import pandas as pd


# Your RandomForestClassifier model and scaler would be loaded here
# from your_model import model, scaler

def generate_performance_plot(district, nitrogen, phosphorus, potassium, temperature, humidity, rainfall, ph):
    # Simulate some performance data (replace this with actual model prediction logic)
    crops = ['Maize', 'Rice', 'Wheat', 'Beans', 'Cassava']
    performance = [round(nitrogen + phosphorus + potassium + temperature + humidity + rainfall + ph, 2) for crop in
                   crops]

    # Create a DataFrame for easier plotting
    df = pd.DataFrame({'Crop': crops, 'Performance': performance})

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(df['Crop'], df['Performance'], color='skyblue')
    plt.xlabel('Performance')
    plt.ylabel('Crops')
    plt.title(f'Performance of Crops in {district}')

    # Save plot to a bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    # Encode to base64 string
    image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()

    return image_base64
