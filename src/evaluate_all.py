# evaluate_all.py
# Este script atua como o orquestrador final do projeto CrediTech. Ele é responsável por carregar as variantes de dataset apropriadas (Ordinal e One-Hot) e instanciar os quatro modelos finais otimizados: Naive Bayes, Árvore de Decisão Ajustada, KNN com Seleção de Atributos e Gradient Boosting. Seu objetivo é realizar uma avaliação estatística robusta utilizando K-Fold Cross Validation (K=10) para calcular a AUC-ROC média, aplicar o Teste t de Student para comprovar a significância estatística do modelo vencedor e gerar uma visualização comparativa unificada das Curvas ROC, salvando o gráfico resultante no diretório de relatórios.
#========================================================================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, auc
from scipy.stats import ttest_rel

# Importação dos modelos
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier

def run_final_evaluation():
    print("🚀 --- INICIANDO VALIDAÇÃO ESTATÍSTICA E GERAÇÃO DE CURVAS ROC --- 🚀\n")

    # ==========================================
    # 1. PREPARAÇÃO DOS DATASETS ESPECÍFICOS
    # ==========================================
    y = None # Alvo global
    datasets = {} # Dicionário para guardar o X de cada modelo

    # A. Dataset Ordinal (Para Naive Bayes)
    df_ordinal = pd.read_csv("data/german_credit_gaussian_nb.csv")
    y = df_ordinal['target']
    datasets['Naive Bayes'] = df_ordinal.drop('target', axis=1)

    # B. Dataset One-Hot Completo (Para Gradient Boosting)
    df_onehot = pd.read_csv("data/german_credit_knn_dt.csv")
    datasets['Gradient Boosting'] = df_onehot.drop('target', axis=1)

    # C. Dataset One-Hot Seletivo (Para Árvore Ajustada)
    colunas_relevantes = [
        "status_conta", "duracao_meses", "historico_credito", 
        "valor_credito", "poupanca", "tempo_emprego", "idade", "target"
    ]
    df_filtrado = df_ordinal[colunas_relevantes]
    df_dummies = pd.get_dummies(df_filtrado, columns=["status_conta", "historico_credito", "poupanca", "tempo_emprego"], drop_first=True)
    datasets['Árvore Ajustada'] = df_dummies.drop('target', axis=1)

    # D. Dataset One-Hot Otimizado por Feature Importance (Para KNN)
    # Recriando a lógica do seu knn_selec.py
    tree_temp = DecisionTreeClassifier(criterion='gini', max_depth=5, class_weight='balanced', random_state=42)
    tree_temp.fit(datasets['Gradient Boosting'], y)
    importancias = pd.Series(tree_temp.feature_importances_, index=datasets['Gradient Boosting'].columns)
    atributos_validos_knn = importancias[importancias > 0.01].index.tolist()
    datasets['KNN Otimizado'] = datasets['Gradient Boosting'][atributos_validos_knn]


    # ==========================================
    # 2. INSTANCIAÇÃO DOS MODELOS CAMPEÕES
    # ==========================================
    modelos = {
        'Naive Bayes': GaussianNB(),
        'Árvore Ajustada': DecisionTreeClassifier(criterion='gini', max_depth=5, class_weight='balanced', random_state=42),
        'KNN Otimizado': KNeighborsClassifier(n_neighbors=5, metric='manhattan', weights='distance'),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, subsample=0.8, random_state=42)
    }

    # ==========================================
    # 3. VALIDAÇÃO ESTATÍSTICA (K-FOLD K=10)
    # ==========================================
    print("🔄 Rodando K-Fold Cross Validation (K=10)...")
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    resultados_auc = {nome: [] for nome in modelos.keys()}

    for train_idx, test_idx in cv.split(datasets['Gradient Boosting'], y): # Usando o df base para gerar os mesmos índices
        y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]

        for nome, modelo in modelos.items():
            X_atual = datasets[nome]
            X_train_fold = X_atual.iloc[train_idx]
            X_test_fold = X_atual.iloc[test_idx]

            modelo.fit(X_train_fold, y_train_fold)
            y_proba = modelo.predict_proba(X_test_fold)[:, 1]
            auc_fold = roc_auc_score(y_test_fold, y_proba)
            resultados_auc[nome].append(auc_fold)

    print("\n📊 Resultados da Validação Cruzada (AUC-ROC Média):")
    for nome in modelos.keys():
        media = np.mean(resultados_auc[nome])
        desvio = np.std(resultados_auc[nome])
        print(f" - {nome}: {media:.4f} (+/- {desvio:.4f})")

    # ==========================================
    # 4. TESTE T DE STUDENT (Gradient Boosting vs Árvore Ajustada)
    # ==========================================
    print("\n⚖️ Teste t de Student (p < 0.05)")
    print("Comparando os dois melhores modelos: Gradient Boosting vs Árvore Ajustada...")
    
    stat, p_value = ttest_rel(resultados_auc['Gradient Boosting'], resultados_auc['Árvore Ajustada'])
    
    print(f"Valor-p encontrado: {p_value:.5f}")
    if p_value < 0.05:
        print("✅ Conclusão: A diferença de performance É estatisticamente significativa.")
    else:
        print("⚠️ Conclusão: A diferença NÃO É estatisticamente significativa (pode ser obra do acaso).")

    # ==========================================
    # 5. PLOTAGEM DA CURVA ROC CONJUNTA
    # ==========================================
    print("\n📈 Gerando gráfico das Curvas ROC...")
    plt.figure(figsize=(10, 8))

    for nome, modelo in modelos.items():
        X_atual = datasets[nome]
        # Treinando com todo o dataset para gerar a curva representativa
        modelo.fit(X_atual, y)
        y_proba = modelo.predict_proba(X_atual)[:, 1]
        
        fpr, tpr, _ = roc_curve(y, y_proba)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, lw=2, label=f'{nome} (AUC = {roc_auc:.4f})')

    # Linha de base (Modelo Aleatório)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatório (AUC = 0.5000)')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)', fontsize=12)
    plt.title('Comparação das Curvas ROC - CrediTech', fontsize=14, fontweight='bold')
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)

    nome_arquivo = 'reports/curvas_roc_creditech.png'
    plt.savefig(nome_arquivo, dpi=300, bbox_inches='tight')
    print(f"✅ Gráfico salvo com sucesso em: {nome_arquivo}")
    plt.show()

if __name__ == "__main__":
    run_final_evaluation()