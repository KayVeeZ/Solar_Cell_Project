from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained machine learning model
model = joblib.load("perovskites.joblib")

# Read the CSV file and create a dictionary mapping IDs to rows
df = pd.read_csv("perovs_feat.csv")

name_to_row = {row["formula"]: row for index, row in df.iterrows()}

# Define route for the home page
@app.route("/")
def home():
    return render_template("index.html", names=name_to_row.keys())

# Define route for handling form submission
@app.route("/predict", methods=["POST"])
def predict():
    # Get the selected ID from the form
    selected_name = request.form["text"]
  
    # Fetch the corresponding row from the dictionary
    selected_row = name_to_row[selected_name]

    # Extract the required data from the row
    feature1 = selected_row["density"]
    feature2 = selected_row["vpa"]
    feature3 = selected_row["packing fraction"]
    feature4 = selected_row["structural complexity per atom"]
    feature5 = selected_row["structural complexity per cell"]
    feature6 = selected_row["max packing efficiency"]

    # Use the extracted data as input features for your model
    # Assuming your model expects a list of features
    features = [feature1, feature2, feature3, feature4, feature5, feature6]

    # Generate prediction using the machine learning model
    prediction = model.predict([features])[0]

    # Determine the stability based on the prediction
    stability = "Stable" if prediction == True else "Not Viable"

    # Render the result template with the prediction
    return render_template("result.html", stability=stability, selected_name=selected_name)

if __name__ == "__main__":
    app.run(debug=True)
