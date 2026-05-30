# naive_bayes_gaussian.py
# Este script é responsável por carregar o dataset pré-processado específico para o Naive Bayes (com Ordinal Encoding), dividir os dados em conjuntos de treino e teste, treinar o modelo GaussianNB, realizar as predições, calcular as métricas de avaliação (Acurácia, Precisão, Recall, F1-Score e AUC-ROC), e exibir os resultados no terminal. O script também inclui a geração e interpretação da matriz de confusão para fornecer insights práticos sobre o desempenho do modelo na classificação de bons e maus pagadores.
#========================================================================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

def testar_gaussian_nb():
    print("🦊 --- INICIANDO TESTE DO GAUSSIAN NAIVE BAYES --- 🦊\n")
    
    # 1. Carregar o dataset específico para o Naive Bayes (variante ordinal)
    caminho_dataset = "data/german_credit_gaussian_nb.csv"
    df = pd.read_csv(caminho_dataset)
    
    # Separar X e y
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 2. Divisão em Treino (70%) e Teste (30%) 
    # Usamos o stratify=y para garantir a mesma proporção de inadimplentes nos dois lados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"📊 Tamanho do Treino: {X_train.shape[0]} amostras")
    print(f"📊 Tamanho do Teste: {X_test.shape[0]} amostras")
    print(f"📉 Proporção de Inadimplentes (Classe 1) no teste: {y_test.sum() / len(y_test):.2%}\n")
    
    # 3. Inicializar e Treinar o Modelo
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    # 4. Predições
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] # Probabilidades para a curva ROC
    
    # 5. Avaliação de Métricas
    acuracia = accuracy_score(y_test, y_pred)
    precisao = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_proba)
    
    # 6. Exibir Resultados no Terminal
    print("================ RESULTADOS EM TESTE ================")
    print(f"Acurácia:  {acuracia:.4f}")
    print(f"Precisão:  {precisao:.4f}  (Dos classificados como maus, quantos eram mesmo?)")
    print(f"Recall:    {recall:.4f}  (De todos os maus reais, quantos o modelo pegou?)")
    print(f"F1-Score:  {f1:.4f}  (Média harmônica importante para o desbalanceamento)")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print("=====================================================\n")
    
    print("📋 Matriz de Confusão:")
    matrix = confusion_matrix(y_test, y_pred)
    print(matrix)
    print(f"\n💡 Significado prática da matriz:")
    print(f"   [{matrix[0][0]}] Bons pagadores detectados corretamente (Verdadeiros Negativos)")
    print(f"   [{matrix[0][1]}] Bons pagadores que o modelo achou que eram ruins (Falsos Positivos)")
    print(f"   [{matrix[1][0]}] MAUS PAGADORES QUE O MODELO NÃO VIU! (Falsos Negativos - Custo Alto ❌)")
    print(f"   [{matrix[1][1]}] Maus pagadores detectados corretamente (Verdadeiros Positivos)")

if __name__ == "__main__":
    testar_gaussian_nb()
