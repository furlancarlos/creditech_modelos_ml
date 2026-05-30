# preprocessing1.py
# Este script é responsável por baixar o dataset German Credit Data da UCI,
# ajustar o target para a classificação binária, normalizar os atributos contínuos
# usando Min-Max Scaling, e salvar o resultado em um arquivo CSV para uso posterior.
# O dataset original possui 1000 amostras e 20 atributos, onde o target é originalmente
# representado por 1 (bom pagador) e 2 (mau pagador). Após o ajuste, teremos 0 para bom pagador
# e 1 para mau pagador, facilitando a aplicação de algoritmos de classificação binária.
#=====================================================================================================

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def baixar_e_normalizar_dados():
    print("📥 Baixando o German Credit Data da UCI...")
    
    # URL oficial do dataset original (formato de texto com espaços)
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
    
    # O dataset não possui cabeçalho, então definimos os nomes das 20 colunas + target
    colunas = [
        "status_conta", "duracao_meses", "historico_credito", "proposito", 
        "valor_credito", "poupanca", "tempo_emprego", "taxa_parcela", 
        "status_sexo", "outros_devedores", "residencia_desde", "propriedades", 
        "idade", "outros_planos_parcelas", "moradia", "creditos_existentes", 
        "emprego", "dependentes", "telefone", "trabalhador_estrangeiro", "target"
    ]
    
    # Lendo o arquivo diretamente da internet
    df = pd.read_csv(url, sep=' ', header=None, names=colunas)
    
    print(f"✅ Dataset carregado com sucesso! Formato: {df.shape}")
    
    # --- 1. AJUSTE DO TARGET ---
    # No original: 1 = Bom, 2 = Mau. 
    # Vamos transformar em: 0 = Bom, 1 = Mau pagador (nossa classe de interesse/custosa)
    df['target'] = df['target'].map({1: 0, 2: 1})
    
    # --- 2. IDENTIFICAÇÃO DOS ATRIBUTOS CONTÍNUOS ---
    # Conforme o enunciado, o KNN precisa de normalização nos atributos contínuos
    # Neste dataset, os principais atributos numéricos contínuos/escala são:
    atributos_continuos = ["duracao_meses", "valor_credito", "idade"]
    
    print("⚖️ Aplicando normalização Min-Max nos atributos contínuos...")
    scaler = MinMaxScaler()
    df[atributos_continuos] = scaler.fit_transform(df[atributos_continuos])
    
    # --- 3. EXPORTAÇÃO ---
    # Criando a pasta 'data' se ela não existir para salvar o resultado
    os.makedirs("data", exist_ok=True)
    caminho_salvamento = "data/german_credit_normalized.csv"
    
    df.to_csv(caminho_salvamento, index=False)
    print(f"💾 Arquivo salvo com sucesso em: {caminho_salvamento}")
    
    return df

if __name__ == "__main__":
    df_tratado = baixar_e_normalizar_dados()
    # Mostra as primeiras linhas para conferir o resultado
    print("\n👀 Prévia dos dados tratados e normalizados:")
    print(df_tratado[["duracao_meses", "valor_credito", "idade", "target"]].head())
