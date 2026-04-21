# Plano de Monitoramento

O deploy de modelos de ML exige monitoramento constante, já que modelos se degradam ao longo do tempo de forma silenciosa. Este plano define as métricas, os alertas e as estratégias de resposta a incidentes.

## 1. Métricas de Monitoramento

As métricas estão divididas em três pilares principais:

### 1.1 Métricas de Sistema (Infraestrutura)
Mapeiam a saúde da API (SLIs - Service Level Indicators):
- **Latência de Inferência (p90, p95 e p99):** Tempo de processamento do endpoint `/predict`. O SLO estabelecido é de p95 $\leq 200\,ms$.
- **Taxa de Erro (HTTP 4xx e 5xx):** Monitora principalmente erros 422 (problemas no schema submetido) e 500 (crash interno na inferência).
- **Throughput (Requests Per Minute - RPM):** Volume de chamadas recebidas para garantir que o autoscaling está funcionando e não há DDoS.

### 1.2 Métricas de Dados (Data Drift e Quality)
Avaliam a estabilidade dos dados que chegam para o modelo:
- **Taxa de Campos Nulos:** Frequência de features cruciais (como `TotalCharges` ou `Tenure`) chegando vazias.
- **Data Drift (Desvio de Distribuição):** Mudanças estatísticas significativas na distribuição das features de entrada (ex: média de `MonthlyCharges` repentinamente $50\%$ maior do que no período de treinamento). Monitorado pelo `Population Stability Index (PSI)` ou teste KS.

### 1.3 Métricas de Modelo (Performance e Concept Drift)
Acompanham a degradação da predição:
- **Concept Drift:** A relação entre features e variável alvo mudou?
- **Distribuição de Scores:** Alterações na média e mediana da probabilidade de churn predita. Se o modelo passar a prever 90% de churn para toda a base subitamente, é um alerta vermelho.
- **Performance de Negócio Aproximada:** Caso demore semanas para sabermos se o cliente realmente deu churn (ground truth), usamos a taxa de conversão das ações de retenção como uma "proxy" da performance do modelo.

## 2. Alertas

Os alertas serão enviados para o canal do Slack da equipe de MLOps:

| Severidade | Condição (Trigger) | Ação Esperada |
| :--- | :--- | :--- |
| **CRITICAL (P1)** | Taxa de Erro HTTP 5xx $> 5\%$ em 5 min | Intervenção imediata. Reversão (Rollback) para a versão anterior da API/Modelo. |
| **CRITICAL (P1)** | Latência p95 $> 1000\,ms$ em 15 min | Verificar carga do servidor / memory leak. Acionar engenharia de plataforma. |
| **WARNING (P2)** | Taxa de Erro HTTP 422 $> 10\%$ em 1 h | Investigar sistemas *upstream* que chamam a API, provavelmente estão enviando payloads quebrados. |
| **WARNING (P3)** | Data Drift detectado (PSI $> 0.2$ em 3 features) | Iniciar pipeline de retreino offline para atualizar modelo na próxima janela. |

## 3. Playbook de Resposta (Retreinamento)

Se um alerta de Degradação de Performance ou Concept Drift for acionado:
1. **Identificação e Isolamento:**
   - Verificar se o drift é por uma falha técnica no banco de dados da empresa (ex: pipeline de dados quebrou e todas as idades estão zeradas) ou se é uma mudança real do mercado (novo plano da concorrência).
2. **Retreinamento (Shadow Mode):**
   - Treinar uma nova versão do modelo localmente ou na nuvem com dados recentes (janela deslizante dos últimos 3 meses).
   - Validar métricas offline (AUC-ROC e F1) no MLflow.
   - Fazer o deploy em modo Shadow (o modelo novo gera predições em background sem afetar a decisão final).
3. **Rollout:**
   - Efetuar o Canary Deployment, direcionando $10\%$ do tráfego para o novo modelo. Se as métricas sistêmicas continuarem saudáveis e a distribuição de saídas for normal, aumentar para $100\%$.
