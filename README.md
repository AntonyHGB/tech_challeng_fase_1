# ML Tech Challenge — Fase 1

Estrutura inicial de projeto de Machine Learning preparada para:

- **PyTorch** (rede neural MLP)
- **Scikit-Learn** (pipelines e modelos baseline)
- **MLflow** (tracking de experimentos)
- **FastAPI** (API de inferência)

> Este repositório está na fase de **estruturação**. A implementação dos códigos será adicionada posteriormente.

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
git clone <URL_DO_REPOSITORIO>
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

## Execução (quando os módulos forem implementados)

### Rodar testes

```bash
pytest
```

### Rodar lint

```bash
ruff check .
```

### Rodar formatação (opcional)

```bash
ruff format .
```

### Subir API FastAPI (planejado)

```bash
uvicorn src.api.main:app --reload
```

### Executar tracking com MLflow (planejado)

```bash
mlflow ui
```

---

## Dependências principais

- `torch`
- `scikit-learn`
- `mlflow`
- `fastapi`
- `uvicorn`

Configuração centralizada no `pyproject.toml` (dependências, linting e pytest).

---

## Convenções de versionamento e commits

- Commits pequenos e semânticos.
- Histórico limpo, focado por etapa:
  - estrutura base
  - configuração do projeto
  - documentação
  - evolução de código

---

## Próximos passos (implementação)

1. Definir schema de dados e pipeline de features.
2. Criar baseline com Scikit-Learn.
3. Implementar MLP com PyTorch.
4. Integrar tracking com MLflow.
5. Expor inferência via FastAPI.
6. Cobrir fluxo com testes automatizados.

---

## Licença

Definir licença do projeto na próxima etapa (`LICENSE`).