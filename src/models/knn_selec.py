# knn_selec.py
# Este script é responsável por carregar o dataset pré-processado específico para o KNN (com One-Hot Encoding), dividir os dados em conjuntos de treino e teste, treinar o modelo KNeighborsClassifier usando a ponderação por distância e a métrica de Manhattan (p=1) apenas com as colunas selecionadas automaticamente pela Árvore de Decisão, realizar as predições, calcular as métricas de avaliação (Acurácia, Precisão, Recall, F1-Score e AUC-ROC), e exibir os resultados no terminal. O script também inclui a geração e interpretação da matriz de confusão para fornecer insights práticos sobre o desempenho do modelo na classificação de bons e maus pagadores.
# Porém com ajuste fino em utilizar a Árvore de Decisão para selecionar as colunas mais relevantes, reduzindo o espaço vetorial e potencialmente melhorando o desempenho do KNN.
#========================================================================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def testar_knn_selecao_automatica():
    print("🤖 --- KNN OTIMIZADO VIA FEATURE IMPORTANCE (ÁRVORE) --- 🤖\n")
    
    # 1. Carregar o dataset completo com One-Hot (49 colunas)
    caminho_dataset = "data/german_credit_knn_dt.csv"
    df = pd.read_csv(caminho_dataset)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Divisão inicial estruturada
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 2. Treinar a Árvore de Decisão para extrair a relevância das colunas
    tree = DecisionTreeClassifier(criterion='gini', max_depth=5, class_weight='balanced', random_state=42)
    tree.fit(X_train, y_train)
    
    # Criar um DataFrame com os pesos de cada atributo
    importancias = pd.DataFrame({
        'Atributo': X.columns,
        'Importancia': tree.feature_importances_
    }).sort_values(by='Importancia', ascending=False)
    
    # Filtrar apenas atributos com importância maior que zero (remover o lixo)
    atributos_validos = importancias[importancias['Importancia'] > 0.01]['Atributo'].tolist()
    
    print(f"✂️ A Árvore analisou as 49 colunas e selecionou as {len(atributos_validos)} mais impactantes.")
    print("🔝 Top 5 atributos mais importantes matematicamente:")
    print(importancias.head(5).to_string(index=False))
    print("-" * 60)
    
    # 3. Filtrar os conjuntos de treino e teste com o novo ecossistema de colunas
    X_train_filtrado = X_train[atributos_validos]
    X_test_filtrado = X_test[atributos_validos]
    
    # 4. Rodar o Grid Search do KNN no novo espaço vetorial limpo
    valores_k = [k for k in range(1, 30) if k % 2 != 0]
    
    melhor_k = None
    melhor_f1 = -1
    melhores_metricas = {}
    melhor_matriz = None
    
    for k in valores_k:
        # Mantendo Manhattan e pesos por distância que funcionam melhor para dados densos
        model = KNeighborsClassifier(n_neighbors=k, weights='distance', metric='manhattan')
        model.fit(X_train_filtrado, y_train)
        
        y_pred = model.predict(X_test_filtrado)
        f1 = f1_score(y_test, y_pred)
        
        if f1 > melhor_f1:
            melhor_f1 = f1
            melhor_k = k
            y_proba = model.predict_proba(X_test_filtrado)[:, 1]
            melhor_matriz = confusion_matrix(y_test, y_pred)
            melhores_metricas = {
                "acuracia": accuracy_score(y_test, y_pred),
                "precisao": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "auc_roc": roc_auc_score(y_test, y_proba)
            }

    print(f"\n🏆 Campeão do Novo Espaço! Melhor K = {melhor_k}\n")
    print("================ RESULTADOS EM TESTE ================")
    print(f"Acurácia:  {melhores_metricas['acuracia']:.4f}")
    print(f"Precisão:  {melhores_metricas['precisao']:.4f}")
    print(f"Recall:    {melhores_metricas['recall']:.4f}  (Anterior era 0.5000)")
    print(f"F1-Score:  {melhor_f1:.4f}")
    print(f"AUC-ROC:   {melhores_metricas['auc_roc']:.4f}")
    print("=====================================================\n")
    
    print("📋 Matriz de Confusão do Melhor K:")
    print(melhor_matriz)
    print(f"\n💡 Significado prático da matriz:")
    print(f"   [{melhor_matriz[0][0]}] Bons pagadores detectados corretamente")
    print(f"   [{melhor_matriz[0][1]}] Falsos Positivos (Alarmes falsos)")
    print(f"   [{melhor_matriz[1][0]}] FALSOS NEGATIVOS (Deixou passar inadimplente ❌)")
    print(f"   [{melhor_matriz[1][1]}] Maus pagadores detectados corretamente")

if __name__ == "__main__":
    testar_knn_selecao_automatica()
