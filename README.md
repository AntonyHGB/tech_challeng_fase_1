# ML Tech Challenge — Fase 1 (Churn)

---

## 1) Estrutura do projeto

```text
.
├── data/
│  ├── raw/
│  └── processed/
├── docs/
│  └── ml_canvas.md
├── models/
│  ├── best_model.joblib     ← melhor modelo serializado (gerado pelo pipeline)
│  ├── baseline_metrics.csv
│  └── reports/
├── notebooks/
│  ├── 01_eda_baselines.ipynb
│  └── 02_churn_insights_clear.ipynb
├── src/
│  └── churn_predictor/
│     ├── __init__.py
│     ├── data.py
│     ├── logging_utils.py
│     ├── api/
│     │  ├── __init__.py
│     │  ├── app.py
│     │  ├── model_loader.py
│     │  ├── schemas.py
│     │  └── routers/
│     │     ├── __init__.py
│     │     └── predict.py
│     ├── models/
│     │  ├── __init__.py
│     │  ├── dataset.py
│     │  └── mlp.py
│     └── pipelines/
│        ├── __init__.py
│        ├── baselines.py
│        └── neural_network.py
├── tests/
│  ├── conftest.py
│  ├── test_api.py
│  ├── test_schema.py
│  └── test_smoke.py
├── .gitignore
├── Makefile
├── pyproject.toml
└── README.md
```

---

## 2) O que cada arquivo principal faz

- `docs/ml_canvas.md`  
  Define problema de negócio, stakeholders, métricas técnicas e de negócio, riscos e SLOs.

- `notebooks/01_eda_baselines.ipynb`  
  EDA completa + execução e comparação dos baselines.

- `src/churn_predictor/data.py`  
  Download/carregamento do dataset, limpeza básica, split X/y e hash de versão.

- `src/churn_predictor/logging_utils.py`  
  Configuração de logging estruturado para pipelines e API.

- `src/churn_predictor/pipelines/baselines.py`  
  Pré-processamento, treino dos baselines (Dummy + Logistic Regression), tracking MLflow e serialização do melhor modelo em `models/best_model.joblib`.

- `src/churn_predictor/models/mlp.py` & `dataset.py`  
  Arquitetura MLP em PyTorch com `PyTorchMLPWrapper` compatível com scikit-learn e `TabularDataset` para alimentar o `DataLoader`.

- `src/churn_predictor/pipelines/neural_network.py`  
  Pipeline de treinamento da MLP com Early Stopping e tracking MLflow.

- `src/churn_predictor/api/app.py`  
  Aplicação FastAPI com middleware de latência e endpoints `/health` e `/predict`.

- `src/churn_predictor/api/schemas.py`  
  Modelos Pydantic para validação de entrada (`PredictionRequest`) e saída (`PredictionResponse`).

- `src/churn_predictor/api/routers/predict.py`  
  Endpoint `POST /predict` — carrega o modelo, pré-processa e retorna probabilidade de churn.

- `tests/`  
  17 testes automatizados: smoke (importações), schema (validação Pydantic) e API (endpoints).

- `Makefile`  
  Comandos prontos para lint, testes, treinamento e execução da API.

---

## 3) Requisitos

- Python 3.10+
- Git
- pip atualizado

---

## 4) Passo a passo para rodar

### 4.1 Clonar repositório
```bash
git clone https://github.com/AntonyHGB/tech_challeng_fase_1.git
cd tech_challeng_fase_1
```

### 4.2 Criar ambiente virtual

**Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (cmd):**
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

**Linux / macOS:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 4.3 Instalar dependências
```bash
pip install --upgrade pip
pip install -e .[dev]
```

Ou usando o Makefile:
```bash
make install
```

### 4.4 Treinar os modelos

> **Importante:** este passo é obrigatório antes de subir a API. Ele gera o arquivo `models/best_model.joblib` que a API usa para fazer inferência.

```bash
# Treina baselines (Logistic Regression, Dummy) e serializa o melhor modelo
python -m churn_predictor.pipelines.baselines

# Treina a Rede Neural (MLP PyTorch) e adiciona ao comparativo
python -m churn_predictor.pipelines.neural_network
```

Ou em um único comando:
```bash
make train
```

**Saídas esperadas:**
- `models/baseline_metrics.csv` — comparativo de todos os modelos
- `models/best_model.joblib` — pipeline treinado pronto para a API
- `models/reports/*.json` — classification reports detalhados
- `mlruns/` — experimentos rastreados pelo MLflow

### 4.5 Subir a API de inferência

```bash
uvicorn churn_predictor.api.app:app --reload --host 0.0.0.0 --port 8000
```

Ou usando o Makefile:
```bash
make run
```

A API estará disponível em:
- **Documentação interativa:** `http://localhost:8000/docs`
- **Health check:** `GET http://localhost:8000/health`
- **Predição:** `POST http://localhost:8000/predict`

**Exemplo de payload para `/predict`:**
```json
{
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 1,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 29.85
}
```

**Resposta esperada:**
```json
{
  "churn_probability": 0.6821,
  "churn_prediction": 1,
  "model_used": "logistic_regression"
}
```

### 4.6 Visualizar experimentos no MLflow

```bash
mlflow ui
```

Abrir no navegador: `http://127.0.0.1:5000`

### 4.7 Abrir notebook de EDA

```bash
jupyter notebook notebooks/01_eda_baselines.ipynb
```

### 4.8 Rodar testes
```bash
pytest
```

Ou:
```bash
make test
```

### 4.9 Rodar linter
```bash
ruff check src/ tests/
```

Ou:
```bash
make lint
```

---

## 5) Comandos rápidos (Makefile)

| Comando | Descrição |
|---|---|
| `make install` | Instala todas as dependências |
| `make train` | Treina baselines + MLP e serializa o modelo |
| `make run` | Sobe a API FastAPI em `localhost:8000` |
| `make test` | Executa os 17 testes automatizados |
| `make lint` | Verifica o código com ruff |
| `make format` | Formata e corrige o código automaticamente |
| `make clean` | Remove caches e artefatos de build |

---

## 6) Checklist antes de abrir uma nova PR

1. Garantir branch atualizada com `master` e sem conflitos.
2. Rodar lint:
```bash
make lint
```
3. Rodar testes:
```bash
make test
```
4. Rodar pipelines para validar execução ponta a ponta:
```bash
make train
```
5. Revisar arquivos alterados:
```bash
git status
git diff --staged
```
6. Confirmar mensagem de commit no padrão Conventional Commits (ex.: `feat: ...`, `fix: ...`, `docs: ...`).

---

## 7) Dependências

**Runtime:**
- numpy, pandas, scikit-learn
- mlflow
- matplotlib, seaborn
- torch
- fastapi, uvicorn[standard], pydantic, joblib

**Desenvolvimento:**
- pytest, pytest-cov
- ruff
- jupyter
- httpx, mypy
