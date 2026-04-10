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
│  ├── baseline_metrics.csv
│  └── reports/
├── notebooks/
│  └── 01_eda_baselines.ipynb
├── src/
│  └── churn_predictor/
│     ├── __init__.py
│     ├── data.py
│     ├── logging_utils.py
│     └── pipelines/
│        ├── __init__.py
│        └── baselines.py
├── tests/
├── .gitignore
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## 2) O que cada arquivo principal faz

- `docs/ml_canvas.md`  
  Define problema de negócio, stakeholders, métricas técnicas e de negócio, riscos e SLOs.

- `notebooks/01_eda_baselines.ipynb`  
  Notebook da etapa 1 com EDA em tabelas e gráficos + execução dos baselines.

- `src/churn_predictor/data.py`  
  Download/carregamento do dataset, limpeza básica, split X/y e hash de versão do dataset.

- `src/churn_predictor/logging_utils.py`  
  Configuração de logging estruturado para pipeline e notebook.

- `src/churn_predictor/pipelines/baselines.py`  
  Pipeline ponta a ponta: preprocessamento, treino, avaliação técnica/negócio, tracking MLflow e export de métricas.

- `models/baseline_metrics.csv`  
  Tabela final com métricas dos baselines.

- `models/reports/*.json`  
  Classification reports por modelo.

- `pyproject.toml`  
  Configuração central do projeto (dependências, pytest, ruff e empacotamento).

---

## 3) Requisitos

- Python 3.10+
- Git
- pip atualizado

---

## 4) Passo a passo para rodar

## 4.1 Clonar repositório
```bash
git clone https://github.com/AntonyHGB/tech_challeng_fase_1.git
cd tech_challeng_fase_1
```

## 4.2 Criar ambiente virtual

### Windows (PowerShell)
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows (cmd)
```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

## 4.3 Instalar dependências

### Opção recomendada (pyproject)
```bash
pip install --upgrade pip
pip install -e .[dev]
```

## 4.4 Rodar pipeline baseline
```bash
python -m churn_predictor.pipelines.baselines
```

Saídas esperadas:
- `models/baseline_metrics.csv`
- `models/reports/*.json`
- `mlruns/` (ou `notebooks/mlruns/`, conforme diretório de execução)

## 4.5 Abrir notebook de EDA
```bash
jupyter notebook notebooks/01_eda_baselines.ipynb
```

## 4.6 Abrir interface do MLflow
```bash
mlflow ui
```
Depois abrir no navegador: `http://127.0.0.1:5000`.

## 4.7 Rodar testes
```bash
pytest
```

---

## 5) Dependências utilizadas na etapa atual

Dependências de runtime:
- numpy
- pandas
- scikit-learn
- mlflow
- matplotlib
- seaborn

Dependências de desenvolvimento:
- pytest
- pytest-cov
- ruff
- jupyter