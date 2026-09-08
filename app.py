
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
MODEL_PATH = "solar_power_prediction_model.pkl"
model = joblib.load(MODEL_PATH)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the form
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        solar_irradiance = float(request.form["solar_irradiance"])
        wind_speed = float(request.form["wind_speed"])

        # Prepare input data
        input_data = pd.DataFrame(
            [[
                temperature,
                humidity,
                solar_irradiance,
                wind_speed
            ]],
            columns=[
                "temperature",
                "humidity",
                "solar_irradiance",
                "wind_speed"
            ]
        )

        # Predict solar power output
        prediction = model.predict(input_data)[0]

        # Avoid negative prediction
        prediction = max(0, prediction)

        return render_template(
            "index.html",
            prediction=round(prediction, 2)
        )

    except Exception as e:
        return render_template(
            "index.html",
            error=str(e)
        )


if __name__ == "__main__":
    app.run(debug=True)
