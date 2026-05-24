# ============================================================
# ETAPA 3 — SLIDING WINDOWS PARA LSTM
# Disciplina: Aprendizado de Máquina
# Objetivo:
# Transformar os dados tabulares em sequências temporais
# adequadas para entrada em redes LSTM
# ============================================================

# ============================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

# ============================================================
# 2. CARREGAMENTO DOS DADOS TRATADOS
# ============================================================

df = pd.read_csv("nba_dados_tratados.csv")

# ============================================================
# 3. FEATURES SELECIONADAS
# ============================================================

# Features ampliadas para melhorar aprendizado temporal

selected_features = [
    'PTS_MA5',
    'PTS_MA10',
    'FG_PCT',
    'FG3_PCT',
    'AST',
    'REB',
    'TOV',
    'HOME',
    'REST_DAYS'
]

# ============================================================
# 4. TARGET
# ============================================================

target = 'PTS'

# ============================================================
# 5. DEFINIÇÃO DAS FEATURES
# ============================================================

X_data = df[selected_features].values

# ============================================================
# 6. ESCALONAMENTO DO TARGET
# ============================================================

# MUITO IMPORTANTE PARA LSTM
# Agora o modelo aprenderá melhor

target_scaler = StandardScaler()

# reshape para coluna

y_data = df[target].values.reshape(-1, 1)

# escalonamento

y_data = target_scaler.fit_transform(y_data)

# volta para vetor 1D

y_data = y_data.flatten()

# ============================================================
# 7. SALVAMENTO DO SCALER
# ============================================================

# Salva média e desvio padrão
# para usar na desnormalização futura

target_mean = target_scaler.mean_[0]

target_std = target_scaler.scale_[0]

np.save("target_mean.npy", target_mean)

np.save("target_std.npy", target_std)

print("\nScaler do target salvo com sucesso!")


# ============================================================
# 8. FUNÇÃO PARA CRIAR SEQUÊNCIAS
# ============================================================

def create_sequences(X, y, window_size):
    """
    Cria sequências temporais para entrada na LSTM

    Entrada:
    X -> features
    y -> target
    window_size -> tamanho da janela

    Saída:
    X_seq -> tensor 3D
    y_seq -> targets
    """

    X_seq = []
    y_seq = []

    # percorre criando janelas

    for i in range(window_size, len(X)):
        # sequência temporal

        X_seq.append(
            X[i - window_size:i]
        )

        # target correspondente

        y_seq.append(
            y[i]
        )

    return np.array(X_seq), np.array(y_seq)


# ============================================================
# 9. TAMANHOS DAS JANELAS
# ============================================================

window_sizes = [5, 10, 15, 20]

# ============================================================
# 10. DICIONÁRIO DOS DATASETS
# ============================================================

datasets = {}

# ============================================================
# 11. CRIAÇÃO DAS SEQUÊNCIAS
# ============================================================

for window in window_sizes:
    print("\n================================================")
    print(f"CRIANDO WINDOW SIZE = {window}")
    print("================================================")

    X_seq, y_seq = create_sequences(
        X_data,
        y_data,
        window
    )

    datasets[window] = {
        'X': X_seq,
        'y': y_seq
    }

    # ========================================================
    # EXIBE FORMATOS
    # ========================================================

    print("\nFormato de X:")
    print(X_seq.shape)

    print("\nFormato de y:")
    print(y_seq.shape)

# ============================================================
# 12. EXEMPLO DE SEQUÊNCIA
# ============================================================

X_example = datasets[5]['X']

y_example = datasets[5]['y']

print("\n================================================")
print("EXEMPLO DE SEQUÊNCIA")
print("================================================")

print("\nPrimeira sequência:")

print(X_example[0])

print("\nTarget correspondente:")

print(y_example[0])

# ============================================================
# 13. DIVISÃO TREINO/TESTE
# ============================================================

# IMPORTANTE:
# Não usar shuffle em séries temporais

train_test_data = {}

for window in window_sizes:
    X_seq = datasets[window]['X']

    y_seq = datasets[window]['y']

    # índice de separação

    split_index = int(len(X_seq) * 0.8)

    # ========================================================
    # TREINO
    # ========================================================

    X_train = X_seq[:split_index]

    y_train = y_seq[:split_index]

    # ========================================================
    # TESTE
    # ========================================================

    X_test = X_seq[split_index:]

    y_test = y_seq[split_index:]

    # ========================================================
    # ARMAZENA
    # ========================================================

    train_test_data[window] = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }

    # ========================================================
    # EXIBE RESULTADOS
    # ========================================================

    print("\n================================================")
    print(f"WINDOW SIZE = {window}")
    print("================================================")

    print("\nX_train:")
    print(X_train.shape)

    print("\ny_train:")
    print(y_train.shape)

    print("\nX_test:")
    print(X_test.shape)

    print("\ny_test:")
    print(y_test.shape)

# ============================================================
# 14. SALVAMENTO DOS ARRAYS
# ============================================================

for window in window_sizes:
    data = train_test_data[window]

    # ========================================================
    # TREINO
    # ========================================================

    np.save(
        f'X_train_window_{window}.npy',
        data['X_train']
    )

    np.save(
        f'y_train_window_{window}.npy',
        data['y_train']
    )

    # ========================================================
    # TESTE
    # ========================================================

    np.save(
        f'X_test_window_{window}.npy',
        data['X_test']
    )

    np.save(
        f'y_test_window_{window}.npy',
        data['y_test']
    )

# ============================================================
# 15. FINALIZAÇÃO
# ============================================================

print("\n================================================")
print("PROCESSO FINALIZADO COM SUCESSO!")
print("================================================")

print("\nArquivos salvos:")

for window in window_sizes:
    print(f"\nWINDOW {window}")

    print(f"- X_train_window_{window}.npy")
    print(f"- y_train_window_{window}.npy")
    print(f"- X_test_window_{window}.npy")
    print(f"- y_test_window_{window}.npy")

print("\nScaler salvo:")
print("- target_mean.npy")
print("- target_std.npy")