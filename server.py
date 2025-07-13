from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)
CORS(app)

model = joblib.load('model.pkl')

@app.route('/', methods=['POST'])
def predict():
    data = request.get_json()
    try:
        input_data = pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],
                                  data=np.array([
                                      data['name'],
                                    data['company'],
                                    data['year'],
                                    data['kms_driven'],
                                    data['fuel_type']
                                  ]).reshape(1,5))
        prediction = model.predict(input_data)[0]
        return jsonify({'Price': round(prediction, 2)})
    
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == "__main__":
    app.run(debug=True, port=5000)