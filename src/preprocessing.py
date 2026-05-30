# preprocessing.py
# Este script é responsável por ler o dataset German Credit Data que foi baixado e salvo localmente, ajustar o target para a classificação binária, normalizar os atributos contínuos usando Min-Max Scaling, e salvar o resultado em dois arquivos CSV distintos para uso posterior. O dataset original possui 1000 amostras e 20 atributos, onde o target é originalmente representado por 1 (bom pagador) e 2 (mau pagador). Após o ajuste, teremos 0 para bom pagador e 1 para mau pagador, facilitando a aplicação de algoritmos de classificação binária. O script também gera duas variantes do dataset: uma com One-Hot Encoding para KNN e Árvore de Decisão, e outra com Ordinal Encoding para GaussianNB.
#========================================================================================================================

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def processar_e_salvar_datasets():
    caminho_original = "data/german.data"
    
    print(f"📂 Lendo o German Credit Data de: {caminho_original}...")
    
    # Nomes das colunas conforme o mapeamento oficial
    colunas = [
        "status_conta", "duracao_meses", "historico_credito", "proposito", 
        "valor_credito", "poupanca", "tempo_emprego", "taxa_parcela", 
        "status_sexo", "outros_devedores", "residencia_desde", "propriedades", 
        "idade", "outros_planos_parcelas", "moradia", "creditos_existentes", 
        "emprego", "dependentes", "telefone", "trabalhador_estrangeiro", "target"
    ]
    
    if not os.path.exists(caminho_original):
        raise FileNotFoundError(f"❌ Erro: O arquivo '{caminho_original}' não foi encontrado na pasta 'data/'.")
    
    # 1. Leitura do arquivo original
    df = pd.read_csv(caminho_original, sep=' ', header=None, names=colunas)
    
    # 2. Ajuste do Target (0 = Bom, 1 = Mau Pagador)
    df['target'] = df['target'].map({1: 0, 2: 1})
    
    # 3. Normalização Min-Max das variáveis contínuas
    print("⚖️ Aplicando normalização Min-Max nas variáveis contínuas...")
    atributos_continuos = ["duracao_meses", "valor_credito", "idade"]
    scaler = MinMaxScaler()
    df[atributos_continuos] = scaler.fit_transform(df[atributos_continuos])
    
    # Separando as variáveis preditoras (X) e o alvo (y)
    X = df.drop('target', axis=1)
    y = df['target']
    
    print("\n🔄 Gerando as duas variantes do dataset...")

    # -------------------------------------------------------------------------
    # VARIANTE 1: ONE-HOT ENCODING (Para KNN e Árvore de Decisão)
    # -------------------------------------------------------------------------
    # pd.get_dummies converte textos em colunas binárias de 0 e 1
    X_onehot = pd.get_dummies(X, drop_first=True)
    # Juntamos de volta com o target
    df_knn_dt = pd.concat([X_onehot, y], axis=1)
    
    caminho_knn_dt = "data/german_credit_knn_dt.csv"
    df_knn_dt.to_csv(caminho_knn_dt, index=False)
    print(f"💾 Dataset para KNN e DT salvo em: {caminho_knn_dt} (Colunas: {df_knn_dt.shape[1]})")

    # -------------------------------------------------------------------------
    # VARIANTE 2: ORDINAL ENCODING (Para GaussianNB)
    # -------------------------------------------------------------------------
    X_ordinal = X.copy()
    # Identifica as colunas que contêm texto e converte em números sequenciais (0, 1, 2...)
    colunas_categoricas = X_ordinal.select_dtypes(include=['object', 'string']).columns
    for col in colunas_categoricas:
        X_ordinal[col] = X_ordinal[col].astype('category').cat.codes
        
    # Juntamos de volta com o target
    df_gaussian_nb = pd.concat([X_ordinal, y], axis=1)
    
    caminho_nb = "data/german_credit_gaussian_nb.csv"
    df_gaussian_nb.to_csv(caminho_nb, index=False)
    print(f"💾 Dataset para GaussianNB salvo em: {caminho_nb} (Colunas: {df_gaussian_nb.shape[1]})")
    
    print("\n✨ Tudo pronto! Os dois datasets foram normalizados e salvos com sucesso.")

if __name__ == "__main__":
    processar_e_salvar_datasets()
