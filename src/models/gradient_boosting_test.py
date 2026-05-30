# gradient_boosting_test.py
# Este script é responsável por carregar o dataset pré-processado específico para o Gradient Boosting (com One-Hot Encoding), dividir os dados em conjuntos de treino e teste, treinar o modelo GradientBoostingClassifier usando uma configuração personalizada para tentar superar o desempenho do AdaBoost, realizar as predições, calcular as métricas de avaliação (Acurácia, Precisão, Recall, F1-Score e AUC-ROC), e exibir os resultados no terminal. O script também inclui a geração e interpretação da matriz de confusão para fornecer insights práticos sobre o desempenho do modelo na classificação de bons e maus pagadores.
#========================================================================================================================

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def testar_gradient_boosting():
    print("📈 --- INICIANDO TESTE COM GRADIENT BOOSTING --- 📈\n")
    
    # 1. Carregar o dataset completo com One-Hot (49 colunas)
    caminho_dataset = "data/german_credit_knn_dt.csv"
    df = pd.read_csv(caminho_dataset)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 2. Divisão estruturada (70% treino, 30% teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # 3. Configurar o Gradient Boosting
    model = GradientBoostingClassifier(
        n_estimators=100,      # Número de árvores sequenciais
        learning_rate=0.05,    # Passos mais suaves para não desequilibrar a Precisão
        max_depth=3,           # Árvores levemente mais profundas para capturar padrões complexos
        subsample=0.8,         # Usa 80% dos dados por árvore para dar robustez contra ruído
        random_state=42
    )
    model.fit(X_train, y_train)
    
    # 4. Predições com Threshold Customizado (Injeção do seu insight)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    novo_threshold = 0.375
    y_pred_ajustado = (y_proba >= novo_threshold).astype(int)
    
    # 5. Métricas utilizando o vetor AJUSTADO (y_pred_ajustado)
    acuracia = accuracy_score(y_test, y_pred_ajustado)
    precisao = precision_score(y_test, y_pred_ajustado)
    recall = recall_score(y_test, y_pred_ajustado)
    f1 = f1_score(y_test, y_pred_ajustado)
    auc_roc = roc_auc_score(y_test, y_proba) # AUC-ROC não muda pois usa as probabilidades brutas
    
    print("================ RESULTADOS EM TESTE ==================")
    print(f"Acurácia:  {acuracia:.4f}")
    print(f"Precisão:  {precisao:.4f}")
    print(f"Recall:    {recall:.4f}  (Com corte de 0.375)")
    print(f"F1-Score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc_roc:.4f}")
    print("=====================================================\n")
    
    print("📋 Matriz de Confusão Ajustada (Threshold 0.375):")
    matrix = confusion_matrix(y_test, y_pred_ajustado)
    print(matrix)
    print(f"\n💡 Significado prático da matriz:")
    print(f"   [{matrix[0][0]}] Bons pagadores detectados corretamente")
    print(f"   [{matrix[0][1]}] Falsos Positivos (Alarmes falsos)")
    print(f"   [{matrix[1][0]}] FALSOS NEGATIVOS (Deixou passar inadimplente ❌)")
    print(f"   [{matrix[1][1]}] Maus pagadores detectados corretamente")

if __name__ == "__main__":
    testar_gradient_boosting()
