import os
import pandas as pd
from flask import Flask, request, render_template, redirect, url_for, flash

from welding_quality import predict_defects, load_data, train_and_evaluate

app = Flask(__name__)
app.secret_key = 'supersecretkey'

MODEL_FILE = 'welding_defect_model.joblib'
PARAMS_FILE = 'preprocessing_parameters.json'
TRAINING_DATA_FILE = 'welding_data.csv'

def check_model_files_exist():
    return os.path.exists(MODEL_FILE) and os.path.exists(PARAMS_FILE)

@app.route('/', methods=['GET', 'POST'])
def index():
    if not check_model_files_exist():
        flash(f"Model files ({MODEL_FILE}, {PARAMS_FILE}) not found. Please train the model first.", 'warning')

    if request.method == 'POST':
        try:
            welding_speed = request.form.get('welding_speed', type=float)
            material_property = request.form.get('material_property', type=str)
            environment_factor = request.form.get('environment_factor', type=float)

            if welding_speed is None or material_property is None or environment_factor is None:
                flash('Missing one or more input values. Please fill all fields.', 'error')
                return render_template('index.html', prediction_result=None)

            input_df = pd.DataFrame({
                'welding_speed': [welding_speed],
                'material_property': [material_property],
                'environment_factor': [environment_factor]
            })

            if not check_model_files_exist():
                flash('Model not available for prediction. Please train first.', 'error')
                prediction_text = 'Error: Model not available'
            else:
                predictions = predict_defects(input_df)
                prediction_text = predictions[0] if predictions else 'Error in prediction'

            return render_template('index.html', prediction_result=prediction_text,
                                   prev_welding_speed=welding_speed,
                                   prev_material_property=material_property,
                                   prev_environment_factor=environment_factor)
        except Exception as e:
            flash(f'Error processing prediction: {e}', 'error')
            return render_template('index.html', prediction_result=None)

    return render_template('index.html', prediction_result=None)

@app.route('/retrain', methods=['GET'])
def retrain_model():
    flash('Attempting to retrain the model...', 'info')
    raw_data = load_data(TRAINING_DATA_FILE)
    if raw_data is not None:
        try:
            best_model, metrics = train_and_evaluate(raw_data)
            if best_model:
                flash('Model retraining completed successfully!', 'success')
            else:
                flash('Model retraining attempted, but no best model was selected.', 'warning')
        except Exception as e:
            flash(f'Error during retraining: {e}', 'error')
    else:
        flash(f'Could not load training data {TRAINING_DATA_FILE} for retraining.', 'error')
    return redirect(url_for('index'))

if __name__ == '__main__':
    if not os.path.exists('templates'):
        os.makedirs('templates')
    app.run(debug=True, host='0.0.0.0', port=8080)
