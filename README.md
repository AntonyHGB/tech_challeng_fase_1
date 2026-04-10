# ML Tech Challenge — Fase 1

Estrutura inicial de projeto de Machine Learning preparada para:

- **PyTorch** (rede neural MLP)
- **Scikit-Learn** (pipelines e modelos baseline)
- **MLflow** (tracking de experimentos)

> Status atual: **Etapa 1 concluída**

---

## Estrutura do projeto

```text
.
├── data/
├── docs/
├── models/
├── notebooks/
├── src/
├── tests/
├── .gitignore
├── pyproject.toml
└── README.md
```

### Diretórios

- `src/`: código-fonte do projeto (treinamento, inferência, utilitários).
- `data/`: dados locais de trabalho (não versionados, exceto `.gitkeep`).
- `models/`: artefatos de modelos gerados localmente.
- `tests/`: testes automatizados com `pytest`.
- `notebooks/`: notebooks de exploração e prototipagem.
- `docs/`: documentação técnica e funcional.

---

## Requisitos

- Python **3.10+**
- `pip` atualizado
- Git

---

## Setup do ambiente

### 1) Clonar o repositório

```bash
git clone 
cd tech_challeng_fase_1
```

### 2) Criar e ativar ambiente virtual

**Windows (PowerShell):**

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**Windows (cmd):**

```bash
python -m venv .venv
.venv\Scripts\activate.bat
```

### 3) Instalar dependências

```bash
pip install --upgrade pip
pip install -e .[dev]
```

---

## Execução da Etapa 1

### Rodar testes

```bash
pytest
```

### Rodar pipeline baseline (Dummy + Regressão Logística)

```bash
python -m churn_predictor.pipelines.baselines
```

### Abrir tracking com MLflow

```bash
mlflow ui
```

### Notebook de EDA

- Arquivo: `notebooks/01_eda_baselines.ipynb`
- Cobre: carga/limpeza, qualidade, distribuições, correlações, métrica de negócio e execução de baselines.

### Artefatos esperados

- `notebooks/models/baseline_metrics.csv`
- `notebooks/models/reports/*.json`
- `notebooks/mlruns/` com parâmetros, métricas e metadados de versão do dataset
- `docs/ml_canvas.md` com definição de stakeholders, SLOs e métricas

## Dependências principais

- `torch`
- `scikit-learn`
- `mlflow`
- `fastapi`
- `uvicorn`

Configuração centralizada no `pyproject.toml` (dependências, linting e pytest).