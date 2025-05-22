# acessi-experience-predictor
Acessi Experience Predictor — API Flask e pipeline de Machine Learning para prever a nota de experiência dos passageiros no transporte público. Projeto acadêmico desenvolvido


# Projeto Acessi - Previsão de Nota de Experiência

Este arquivo contém a implementação da **Sprint 4** do Projeto Acessi: um pipeline de machine learning para prever a nota de experiência de passageiros de metrô, deploy como API Flask e uma interface web simples para consulta.

---

## 📁 Estrutura de Pastas

```
acessi-ml-api/
├── app.py                  # API Flask
├── readme.md               # Este documento
├── requirements.txt        # Dependências Python
│
├── data/                   # Dados de entrada
│   └── dados.csv
│
├── models/                 # Modelos serializados
│   └── melhor_modelo.pkl
│
├── src/                    # Código-fonte
│   ├── data_prep.py
│   ├── model_trainer.py
│   └── model_predict.py
│
└── web/                    # Interface web estática
    ├── index.html
    └── styles.css
```

---

## 🚀 Setup Inicial

1. **Obtenha o projeto**  
   Faça download ou copie este diretório para o seu computador.

2. **Crie e ative um ambiente virtual**  
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # Linux/macOS
   venv\Scripts\activate     # Windows
   ```

3. **Instale as dependências**  
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🧹 Treinamento e Serialização do Modelo

1. **Prepare os dados** em `data/dados.csv`.  
2. **Execute o trainer** para treinar, comparar e salvar o melhor modelo:  
   ```bash
   python src/model_trainer.py
   ```
   - O script faz:
     - Leitura e pré-processamento (imputação + OneHotEncoder);
     - Divisão treino/validação/teste (60/20/20);
     - Treino de Random Forest, XGBoost e MLP;
     - Seleção e re-treino do melhor modelo (XGBoost);
     - Salvamento em `models/melhor_modelo.pkl`.

3. **Confira as métricas** exibidas no terminal (RMSE, MAE, R²).

---

## 🌐 Execução da API Flask

1. **Inicie o servidor**:  
   ```bash
   python app.py
   ```
   - A aplicação rodará em `http://127.0.0.1:5000/`.

2. **Endpoint de predição**  
   - **POST** `/predict`  
   - **Headers**:
     ```
     Content-Type: application/json
     ```
   - **Body** (JSON):
     ```json
     {
       "Faixa etária": "26-35",
       "Frequência de uso": "Diariamente",
       "Necessidade especial": "Não",
       "Horários de uso": "Manhã",
       "Problemas enfrentados":"Atraso, Lotação",
       "Reação a problemas": "Procuro alternativa",
       "Planejamento de viagem": "Planejo com apps",
       "Tempo de viagem": "20-40 min",
       "Perda de compromissos": "Não",
       "Uso de apps": "Sim",
       "Funcionalidades preferidas": "Rotas",
       "Como gostaria de receber alertas": "Notificação",
       "Uso de assistente virtual": "Nunca usei"
     }
     ```
   - **Resposta**:
     ```json
     { "nota_prevista": 3.0 }
     ```

---

## 🖥️ Interface Web

- Abra `web/index.html` no navegador.  
- Preencha o formulário e clique em **Prever Nota de Experiência**.  
- O resultado aparecerá diretamente na página.

> **Obs:** o front faz uma requisição `fetch()` para o endpoint `/predict`.

---

## ⚙️ Scripts e Utilitários

- **`src/data_preprocessing.py`**: função `carregar_e_preprocessar_dados()`.  
- **`src/model_trainer.py`**: script de treinamento e serialização.  
- **`src/model_predictor.py`**: funções `carregar_modelo()` e `prever_nota()`.  
- **`app.py`**: definindo rotas Flask.

---

## 
