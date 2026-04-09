# ML Canvas — Churn Predictor (Telecom)

## 1) Problema de negócio

Reduzir churn de clientes de telecom por meio de identificação proativa de clientes com alta propensão de cancelamento, suportando ações de retenção com melhor retorno financeiro.

## 2) Stakeholders

- diretoria executiva (sponsor de resultado)
- equipe de retenção / crm
- equipe de marketing
- equipe de atendimento (contact center)
- time de dados / ml engineering
- time de produto digital
- time de compliance e segurança

## 3) Decisão suportada pelo modelo

Priorizar clientes para campanhas de retenção (contato ativo, oferta, benefício) com base na probabilidade de churn prevista.

## 4) Entrada e saída

### Entradas esperadas
- perfil do cliente (gênero, senioridade, dependentes, parceiro)
- características contratuais (tipo de contrato, método de pagamento, tenure)
- serviços contratados (internet, telefonia, add-ons)
- variáveis financeiras (monthlycharges, totalcharges)

### Saída
- probabilidade de churn em $[0,1]$
- classe binária (churn / não churn) a partir de threshold operacional

## 5) Métricas de sucesso

### Técnicas
- auc-roc
- pr-auc
- f1-score

### Negócio (estimada na etapa 1)
$$
\text{valor\_líquido\_estimado} = (TP \cdot p_{retenção} \cdot V_{churn}) - ((TP + FP) \cdot C_{contato})
$$

onde:
- $TP$: churners corretamente identificados
- $FP$: não churners acionados
- $p_{retenção}$: taxa de sucesso da ação de retenção
- $V_{churn}$: valor recuperado por churn evitado
- $C_{contato}$: custo da ação de contato

## 6) Restrições e riscos

- desbalanceamento de classes
- drift de comportamento de clientes ao longo do tempo
- custo operacional de falso positivo elevado
- risco de viés por segmentos de clientes
- qualidade de dados (campos faltantes e inconsistências)

## 7) SLOs propostos (fase inicial)

- disponibilidade do serviço de inferência: $\geq 99.5\%$ mensal
- latência p95 da inferência online: $\leq 200\,ms$
- atualização de baseline offline: 1 execução diária
- completude do tracking no mlflow: 100% dos runs com parâmetros, métricas e versão do dataset
- degradacão máxima de auc-roc em produção antes de re-treino: $\leq 5\%$ vs referência validada

## 8) Plano de entrega da etapa 1

- eda completa no notebook `notebooks/01_eda_baselines.ipynb`
- baselines com dummyclassifier e regressão logística (scikit-learn)
- métricas técnicas e de negócio registradas em mlflow
- versionamento do dataset por hash
- documentação do canvas em `docs/ml_canvas.md`