from flask import Flask, render_template, request,jsonify
import pandas as pd
import joblib
import re

app = Flask(__name__)

# Read the CSV file and create a dictionary mapping IDs to rows
df = pd.read_csv("perovs_feat.csv")

def remove_numbers(input_string):
  return re.sub(r'\d+', '', input_string)

def check_element_exist(input_combination):
  for formula in df['formula']:
      if remove_numbers(formula) == input_combination:
          return formula
  return False

# Load the trained machine learning model
model = joblib.load("perovskites.joblib")

name_to_row = {row["formula"]: row for index, row in df.iterrows()}

# Function to find elements in A, B, and C and print their intersections
def find_elements(formulas):
    elements_A = set()
    elements_B = set()
    elements_C = set()

    # Iterate over each formula
    for formula in formulas:
        # Extract elements from the formula using regular expression
        elements = re.findall('[A-Z][a-z]*', formula)
        if len(elements) == 3:
            elements_A.add(elements[0])
            elements_B.add(elements[1])
            elements_C.add(elements[2])

    return sorted(elements_A), sorted(elements_B), sorted(elements_C)
chemical_formulas= df['formula'].unique()
elements_A, elements_B, elements_C=find_elements(chemical_formulas)

@app.route('/')
def home_page():
  return render_template('index.html')
  
# Define route for the predictor
@app.route("/predictor")
def predictor():
    return render_template("predictor.html", names=name_to_row.keys(), elements_A=elements_A, elements_B=elements_B, elements_C=elements_C)

# Define route for handling form submission
@app.route("/predict", methods=["GET","POST"])
def predict():
  if request.method == 'POST':
    element_a = request.form["element_a"].capitalize()
    element_b = request.form["element_b"].capitalize()
    element_c = request.form["element_c"].capitalize()
    
    input_combination = element_a + element_b + element_c
    status = check_element_exist(input_combination)
    if status == False:
      return render_template("not_found.html")
    elif status != False:
      selected_name = status
    
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
      stability = "Can be Used in creating Solar Cells" if prediction == True else "Not Viable"
    
      # Render the result template with the prediction
      return render_template("result.html", stability=stability, selected_name=selected_name)
  else:
    return render_template('predict.html')

@app.route('/predict2', methods=['GET','POST'])
def predict2():
  if request.method == 'POST':
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
    stability = "Can be Used in creating Solar Cells" if prediction == True else "Not Viable"
  
    # Render the result template with the prediction
    return render_template("result.html", stability=stability, selected_name=selected_name)
  else:
    return render_template('predict2.html',names=name_to_row.keys())



@app.route('/about')
def about():
  return render_template('about.html')

@app.route('/solar')
def solar():
  return render_template('solar.html')

@app.route('/ml')
def ml():
  return render_template('ml.html')

if __name__ == "__main__":
    app.run(debug=True)
