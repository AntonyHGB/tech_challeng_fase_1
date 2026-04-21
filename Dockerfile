FROM python:3.10-slim

WORKDIR /app

# Cria usuário não-root por segurança
RUN useradd -m -u 1000 appuser

# Copia arquivos base para instalação das dependências
COPY pyproject.toml README.md /app/

# Cria uma estrutura dummy de src para o pip conseguir instalar as dependências
# Isso garante que o Docker faça cache das bibliotecas pesadas sem quebrar no pip install
RUN mkdir -p /app/src/churn_predictor && touch /app/src/churn_predictor/__init__.py

# Instala todas as dependências do projeto
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir .

# Agora copia o código-fonte real e os artefatos
COPY src /app/src
COPY models /app/models

# Instala o projeto novamente (apenas o código do app, sem reinstalar libs) para linkar o source correto
RUN pip install --no-cache-dir --no-deps .

# Ajusta as permissões
RUN chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "churn_predictor.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
