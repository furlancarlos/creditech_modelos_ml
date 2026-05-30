# preprocessing2.py
# Arquivo para ler diretamente na pasta local o dataset German Credit Data, ajustar o target para a classificação binária,
# normalizar os atributos contínuos usando Min-Max Scaling, e salvar o resultado em um arquivo CSV para uso posterior.
# O dataset original possui 1000 amostras e 20 atributos, onde o target é originalmente representado por 1 (bom pagador) e 2 (mau pagador). Após o ajuste, teremos 0 para bom pagador e 1 para mau pagador, facilitando a aplicação de algoritmos de classificação binária. Esse ajuste pode ser visualizado na coluna 'target' do dataset tratado.
#========================================================================================================================

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def baixar_e_normalizar_dados():
    # Caminho do arquivo original baixado diretamente do site da UCI e salvo na pasta 'data' com o nome 'german.data'
    caminho_original = "data/german.data"
    
    print(f"📂 Lendo o German Credit Data localmente de: {caminho_original}...")
    
    # O dataset não possui cabeçalho, então definimos os nomes das 20 colunas + target conforme a descrição oficial do dataset
    colunas = [
        "status_conta", "duracao_meses", "historico_credito", "proposito", 
        "valor_credito", "poupanca", "tempo_emprego", "taxa_parcela", 
        "status_sexo", "outros_devedores", "residencia_desde", "propriedades", 
        "idade", "outros_planos_parcelas", "moradia", "creditos_existentes", 
        "emprego", "dependentes", "telefone", "trabalhador_estrangeiro", "target"
    ]
    
    # Verifica se o arquivo realmente existe na pasta antes de tentar ler
    if not os.path.exists(caminho_original):
        raise FileNotFoundError(
            f"❌ Erro: O arquivo '{caminho_original}' não foi encontrado! "
            f"Certifique-se de que ele está renomeado exatamente como 'german.data' dentro da pasta 'data/'."
        )
    
    # Lendo o arquivo local (usando o separador de espaço simples)
    df = pd.read_csv(caminho_original, sep=' ', header=None, names=colunas)
    
    print(f"✅ Dataset carregado com sucesso! Formato: {df.shape}")
    
    # --- 1. AJUSTE DO TARGET ---
    df['target'] = df['target'].map({1: 0, 2: 1})
    
    # --- 2. IDENTIFICAÇÃO DOS ATRIBUTOS CONTÍNUOS ---
    atributos_continuos = ["duracao_meses", "valor_credito", "idade"]
    
    print("⚖️ Aplicando normalização Min-Max nos atributos contínuos...")
    scaler = MinMaxScaler()
    df[atributos_continuos] = scaler.fit_transform(df[atributos_continuos])
    
    # --- 3. EXPORTAÇÃO ---
    # Salva o arquivo final tratado com outro nome para não sobrescrever o original
    caminho_salvamento = "data/german_credit_normalizado.csv"
    df.to_csv(caminho_salvamento, index=False)
    print(f"💾 Arquivo normalizado salvo com sucesso em: {caminho_salvamento}")
    
    return df

if __name__ == "__main__":
    df_tratado = baixar_e_normalizar_dados()
    print("\n👀 Prévia dos dados tratados e normalizados:")
    print(df_tratado[["duracao_meses", "valor_credito", "idade", "target"]].head())
