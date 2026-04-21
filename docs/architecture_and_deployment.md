# Arquitetura e Deploy

## 1. Abordagem Escolhida: Real-time (API Inference) vs. Batch

Para a resolução do problema de churn desta empresa de telecomunicações, a arquitetura escolhida foi o **Deploy Real-Time (Síncrono via API)**, construído com **FastAPI**. 

### Justificativa da Escolha

O caso de uso de prevenção de churn muitas vezes exige ações imediatas ou integrações diretas com os canais de atendimento:
- **Atendimento Reativo/Contato Simultâneo:** Quando um cliente liga para a central de atendimento (Contact Center) com uma reclamação, o sistema do atendente pode consultar a API em tempo real para obter o *score* atualizado de churn daquele cliente. Se for alto, o atendente recebe a instrução de oferecer retenção imediata.
- **Navegação no App:** Se o cliente acessar a aba de "Cancelamento de Plano" no aplicativo ou portal web, a aplicação cliente pode bater na API para decidir, em milissegundos, se exibe uma tela de desconto automático ou se prossegue com o fluxo de cancelamento normal.

Um processamento **Batch** (inferência de toda a base durante a madrugada) não permitiria respostas instantâneas a comportamentos novos ou a chamadas simultâneas de clientes críticos, o que é vital para ações de retenção no momento exato de frustração do usuário.

## 2. Desenho da Solução

O fluxo de dados desde o treino até a inferência está consolidado nos seguintes módulos:

```text
[ MLflow Tracking ] <---- (Treinamento) --- [ Scikit-Learn / PyTorch ]
                                                 |
                                         (Serialização .joblib)
                                                 |
                                                 v
[ Cliente HTTP / CRM ] ----> (Request JSON) ----> [ FastAPI Endpoint (/predict) ]
                                                         |
                                                 (Validação Pydantic)
                                                         |
                                                  [ Pré-processamento ]
                                                         |
                                                    [ Inferência ]
                                                         |
[ Cliente HTTP / CRM ] <--- (Response JSON) <-------------
```

### Componentes Principais:
1. **Model Registry/Storage:** Modelos treinados localmente são gerados em `models/best_model.joblib`. Em um ambiente Cloud, isso residiria no MLflow Model Registry ou em um bucket (ex: S3).
2. **FastAPI (Inference Server):** Responsável por prover as rotas `/health` (para load balancers validarem a disponibilidade) e `/predict`. Incorpora um middleware para monitoramento de latência e logs estruturados.
3. **Pydantic (Validação de Schema):** Garante que payloads com features faltando ou com tipos incorretos (ex: texto no lugar de int) retornem erro *422 Unprocessable Entity*, protegendo o modelo de inputs inválidos que levariam a crashes de inferência.

## 3. Arquitetura de Nuvem Proposta (Deploy Opcional / Visão de Futuro)

Para colocar a solução em produção na nuvem (AWS/GCP/Azure):
- **Containerização:** O projeto deve ser empacotado em um contêiner Docker.
- **Orquestração:**
  - *Opção Simples/Serverless:* AWS App Runner, Google Cloud Run ou Azure Container Apps. Estes serviços escalam automaticamente a API para zero quando ociosa e para milhares de instâncias sob tráfego, garantindo baixo custo.
  - *Opção Enterprise:* Kubernetes (EKS, GKE, AKS) gerenciando Pods da API, acoplado a um Load Balancer.
- **CI/CD:** O GitHub Actions executará o comando `make test` e `make lint` a cada commit, realizando o build da imagem Docker e o deploy na nuvem caso a branch seja `main`.
