# 🏦 CrediTech - Desafio de Ciência de Dados
> **Diagnóstico do Sistema de Crédito Financeiro com o German Credit Data**

Este projeto foi desenvolvido com o objetivo de avaliar, tunar e selecionar o melhor modelo de Machine Learning para prever o risco de inadimplência de clientes na CrediTech. O foco central da análise é otimizar o **Recall** (Detecção de Maus Pagadores) para proteger o caixa da instituição, sem destruir a viabilidade comercial (Precisão).

---

## 📁 Estrutura do Projeto

```text
creditech_modelos_ml/
├── data/
│   ├── german_credit_gaussian_nb.csv  # Dataset com codificação ordinal (Naive Bayes)
│   └── german_credit_knn_dt.csv       # Dataset com One-Hot Encoding e Normalização
├── reports/
│   └── curvas_roc_creditech.png       # Gráfico comparativo das curvas ROC
├── src/
│   ├── models/
│   │   ├── naive_bayes_gaussian.py    # Modelo probabilístico base
│   │   ├── decision_tree.py           # Árvore de decisão original (Sem Poda)
│   │   ├── decision_tree_selec.py     # Árvore com espaço vetorial reduzido (18 colunas)
│   │   ├── knn_selec.py               # KNN otimizado com distância de Manhattan e Gini
│   │   └── gradient_boosting_test.py  # Gradient Boosting com Threshold Otimizado (0.375)
│   ├── evaluate_all.py                # Script orquestrador, K-Fold e Teste t
│   └── preprocessing.py               # Limpeza e separação dos datasets
├── .gitignore                         # Proteção de arquivos locais e caches do Python
├── requirements.txt                   # Dependências do projeto
└── README.md                          # Documentação e relatório gerencial do projeto
```

## 📊 Resultados e Métricas dos Modelos

| Modelo | Estratégia de Dados / Ajustes | AUC-ROC | Falsos Negativos (Risco) | Falsos Positivos (Comercial) | Recall (Alvo) | Precisão | F1-Score |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Naive Bayes | Ordinal Encoding | 0.7521 | 33 | 63 | 63,33% | 47,50% | 0.5429 |
| Árvore Simples | One-Hot (Sem Poda) | 0.5881 | 51 | 54 | 43,33% | 41,94% | 0.4262 |
| 👑 Árvore Ajustada | One-Hot + Poda + Pesos Balanceados | 0.7368 | 20 | 84 | 77,78% | 45,45% | 0.5738 |
| KNN Otimizado | Manhattan + Feature Selection (K=3) | 0.6860 | 47 | 27 | 47,78% | 61,43% | 0.5375 |
| 🎯 Gradient Boosting | Threshold Customizado (0.375) | 0.7883 | 30 | 42 | 66,67% | 58,82% | 0.6250 |

## ✍️ Principais Conclusões do Diagnóstico

* **O Campeão em Proteção de Caixa (Abordagem Conservadora):** A Árvore de Decisão Otimizada consolidou-se como a escolha mais segura para estancar prejuízos. Ao atingir 77,78% de Recall, ela permitiu a passagem de apenas 20 inadimplentes (Falsos Negativos). É o modelo ideal para cenários de forte aversão ao risco financeiro.
* **O Campeão em Eficiência de Negócio (Abordagem de Crescimento):** O Gradient Boosting com Threshold Customizado em 0.375 apresentou a maior robustez estatística do projeto, com AUC-ROC de 0.7883 e o maior F1-Score (0.6250). Ao mover o limiar de decisão, o modelo cortou os alarmes falsos pela metade (42 contra 84 da árvore) mantendo um Recall sólido de 66,67%, tornando-se a melhor opção para o equilíbrio comercial da CrediTech.
* **Inadequação do KNN:** Os experimentos geométricos comprovaram que o KNN sofre severamente com o desbalanceamento de classes em dados esparsos de crédito. Mesmo após a redução para as 18 colunas mais importantes e a alteração para a distância de Manhattan, o modelo foi incapaz de superar a barreira de 47,78% de Recall.

## 🚀 Como Executar os Testes

Certifique-se de ter as bibliotecas instaladas executando `pip install -r requirements.txt`. Para rodar a avaliação final completa, execute na raiz do projeto:

```bash
python src/evaluate_all.py
```
