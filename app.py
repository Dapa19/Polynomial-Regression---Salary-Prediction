from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the model and PolynomialFeatures
model = joblib.load('polynomial_regression.pkl')
poly_reg = joblib.load('polynomial_features.pkl')

@app.template_filter('format_currency')
def format_currency(value):
    """Filter to format a number as currency."""
    if value is None:
        return ""
    return f"${value:,.2f}"

@app.route('/')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get the input data
        position_level = float(request.form['position_level'])

        # Generate data for the chart (levels 1 to 10)
        position_levels = np.arange(1, 11, 1).reshape(-1, 1)
        position_levels_transformed = poly_reg.transform(position_levels)
        predicted_salaries = model.predict(position_levels_transformed)

        # Predict the salary for the input level
        position_level_transformed = poly_reg.transform([[position_level]])
        prediction = model.predict(position_level_transformed)[0]

        # Render the result template with data
        return render_template('result.html',
                               prediction=prediction,
                               position_levels=position_levels.flatten().tolist(),
                               predicted_salaries=predicted_salaries.tolist())

if __name__ == '__main__':
    app.run(debug=True)
