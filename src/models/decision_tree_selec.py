# decision_tree_selec.py
# Este script é responsável por carregar o dataset pré-processado específico para a Árvore de Decisão (com One-Hot Encoding), dividir os dados em conjuntos de treino e teste, treinar o modelo DecisionTreeClassifier usando o critério Gini, realizar as predições, calcular as métricas de avaliação (Acurácia, Precisão, Recall, F1-Score e AUC-ROC), e exibir os resultados no terminal. O script também inclui a geração e interpretação da matriz de confusão para fornecer insights práticos sobre o desempenho do modelo na classificação de bons e maus pagadores.
# Porém com ajuste fino em utilizar a Árvore de Decisão para selecionar as colunas mais relevantes, reduzindo o espaço vetorial e potencialmente (em tese) melhorando o desempenho do modelo.
#========================================================================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def testar_arvore_seletiva():
    print("🌳 --- ÁRVORE DE DECISÃO COM SELEÇÃO DE ATRIBUTOS CRÍTICOS --- 🌳\n")
    
    # 1. Carregar o dataset variante ordinal para filtrar as colunas antes do One-Hot
    caminho_dataset = "data/german_credit_gaussian_nb.csv" 
    df = pd.read_csv(caminho_dataset)
    
    # O mesmo filtro de 8 variáveis originais (que geram as 18 colunas pós-One-Hot)
    colunas_relevantes = [
        "status_conta", "duracao_meses", "historico_credito", 
        "valor_credito", "poupanca", "tempo_emprego", "idade", "target"
    ]
    df_filtrado = df[colunas_relevantes]
    
    # Aplicamos o One-Hot apenas nesse grupo essencial
    df_dummies = pd.get_dummies(df_filtrado, columns=["status_conta", "historico_credito", "poupanca", "tempo_emprego"], drop_first=True)
    
    X = df_dummies.drop('target', axis=1)
    y = df_dummies['target']
    
    # 2. Divisão com estratificação (70% treino, 30% teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"✂️ Dataset reduzido de 49 para {X_train.shape[1]} colunas antes do treino.\n")
    
    # 3. Inicializar a Árvore Otimizada (Com os parâmetros que salvaram o modelo antes)
    model = DecisionTreeClassifier(
        criterion='gini', 
        max_depth=5, 
        class_weight='balanced', 
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 4. Predições
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 5. Avaliação de Métricas
    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_proba)
    
    # 6. Exibir Resultados no Terminal
    print("================ RESULTADOS EM TESTE ================")
    print(f"Acurácia:  {acuracia:.4f}")
    print(f"Precisão:  {precisao:.4f}")
    print(f"Recall:    {recall:.4f}  (Modelo com 49 colunas tinha 0.7778)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print("=====================================================\n")
    
    print("📋 Matriz de Confusão:")
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)
    print(f"\n💡 Significado prático da matriz:")
    print(f"   [{matrix[0][0]}] Bons pagadores detectados corretamente")
    print(f"   [{matrix[0][1]}] Falsos Positivos (Alarmes falsos)")
    print(f"   [{matrix[1][0]}] FALSOS NEGATIVOS (Deixou passar inadimplente ❌)")
    print(f"   [{matrix[1][1]}] Maus pagadores detectados corretamente")

if __name__ == "__main__":
    testar_arvore_seletiva()
