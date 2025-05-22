from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from src.model_predict import carregar_modelo, prever_nota

app = Flask(__name__)
CORS(app)
modelo, preprocessor = carregar_modelo()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    dados = request.get_json()
    nota = prever_nota(dados, modelo, preprocessor)
    return jsonify({'nota_prevista': nota})

if __name__ == '__main__':
    app.run(debug=True, port=5000)