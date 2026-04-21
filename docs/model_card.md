# Model Card: Predição de Churn em Telecom

## 1. Detalhes do Modelo
- **Desenvolvedor:** Antonio Henrique Gimenes Bertagnolli
- **Data da versão:** Abril de 2026
- **Versão:** 1.0.0
- **Tipo do Modelo:** Classificador Binário (Multi-Layer Perceptron - PyTorch & Logistic Regression - Scikit-Learn)
- **Licença:** MIT

## 2. Intenção de Uso
- **Casos de Uso Primários:** O modelo tem como objetivo prever a probabilidade de um cliente de telecomunicações cancelar o serviço (churn) no curto prazo. Ele será usado primariamente pela equipe de retenção/CRM para priorizar contatos proativos e ofertas personalizadas, maximizando o ROI de ações de retenção.
- **Casos de Uso Fora de Escopo:** O modelo não deve ser usado para recusa de serviços essenciais, aprovação ou reprovação de crédito, ou qualquer decisão que afete negativamente de forma automática o cliente sem supervisão humana.

## 3. Dados
- **Dataset de Treinamento:** Dataset público de telecomunicações focado em Churn (ex. Telco Customer Churn - IBM).
- **Características (Features):** Incluem dados demográficos (gênero, idade, dependentes), perfil de uso dos serviços (linhas, internet, streaming, tech support), tipo de contrato e histórico financeiro (MonthlyCharges, TotalCharges, Tenure).
- **Processamento:**
  - Variáveis categóricas tratadas via One-Hot Encoding ou Ordinal Encoding.
  - Variáveis numéricas escalonadas via `StandardScaler`.
  - Tratamento de nulos/vazios e inconsistências nos tipos (ex.: TotalCharges vazio mapeado para 0 em novos clientes).

## 4. Performance e Métricas
- **Métricas de Avaliação:** Foram otimizadas a AUC-ROC (capacidade geral de discriminação) e F1-Score da classe 1 (Churn), dada a natureza desbalanceada dos dados (menos casos de churn do que retenção). A métrica de negócio foca no valor recuperado vs. custo de ação (FP vs TP).
- **Baselines vs. Neural Network:**
  - O baseline (Logistic Regression) apresentou excelente estabilidade e facilidade de interpretação.
  - A Rede Neural (MLP) obteve flexibilidade extra na captura de relações não-lineares, com a ressalva de custo computacional levemente maior.
  - (Os resultados alcançados na última execução foram: AUC-ROC ≈ 0.84, F1 ≈ 0.60 no baseline otimizado).

## 5. Limitações e Vieses
- **Vieses Demográficos:** Como features de gênero e senioridade ("SeniorCitizen") são utilizadas, é recomendável avaliar periodicamente o `Disparate Impact` (impacto desproporcional) entre grupos demográficos, garantindo que o modelo não foque ofertas de retenção excessivamente em perfis específicos de modo discriminatório.
- **Limitações de Generalização:** O modelo foi treinado em dados históricos que refletem as políticas de preços atuais da companhia. Mudanças bruscas de mercado (como a entrada de um concorrente agressivo) não serão captadas instantaneamente e podem degradar a performance.

## 6. Cenários de Falha e Riscos
- **Custo de Falsos Positivos (FP):** Clientes sem intenção real de churn receberem benefícios caros (descontos). Isso gera desperdício orçamentário.
- **Custo de Falsos Negativos (FN):** Clientes com alto risco ignorados pelo modelo, gerando perda direta de receita recorrente.
- **Data Drift:** Se a distribuição dos dados de entrada via API (ex: planos novos lançados) for diferente da distribuição de treino, o modelo retornará predições não confiáveis.

## 7. Recomendações
- Implementar um limiar (threshold) dinâmico, guiado pelo custo marginal da campanha de marketing no mês (foco em FP vs TP).
- Retreinar o modelo (ou refinar seus pesos) a cada trimestre ou a cada grande lançamento de portfólio de produtos.
