# ============================================================
# ETAPA 4 — MODELO LSTM PARA REGRESSÃO
# Disciplina: Aprendizado de Máquina
# Objetivo:
# Construir, treinar e avaliar o modelo LSTM
# para previsão de pontos (PTS)
# ============================================================

# ============================================================
# IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import (
    EarlyStopping
)

from tensorflow.keras.optimizers import Adam

# ============================================================
# JANELAS
# ============================================================

window_sizes = [5, 10, 15, 20]

# ============================================================
# RESULTADOS
# ============================================================

results = []

# ============================================================
# LOOP PRINCIPAL
# ============================================================

for window in window_sizes:
    print("\n================================================")
    print(f"WINDOW {window}")
    print("================================================")

    # ========================================================
    # CARREGAMENTO DOS DADOS
    # ========================================================

    X_train = np.load(f'X_train_window_{window}.npy')
    y_train = np.load(f'y_train_window_{window}.npy')

    X_test = np.load(f'X_test_window_{window}.npy')
    y_test = np.load(f'y_test_window_{window}.npy')

    # ========================================================
    # MODELO
    # ========================================================

    model = Sequential()

    # --------------------------------------------------------
    # LSTM SIMPLES
    # --------------------------------------------------------

    model.add(
        LSTM(
            units=16,
            input_shape=(
                X_train.shape[1],
                X_train.shape[2]
            )
        )
    )

    # --------------------------------------------------------
    # DROPOUT
    # --------------------------------------------------------

    model.add(
        Dropout(0.2)
    )

    # --------------------------------------------------------
    # SAÍDA
    # --------------------------------------------------------

    model.add(
        Dense(1)
    )

    # ========================================================
    # OTIMIZADOR
    # ========================================================

    optimizer = Adam(
        learning_rate=0.001
    )

    # ========================================================
    # COMPILAÇÃO
    # ========================================================

    model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )

    # ========================================================
    # EARLY STOPPING
    # ========================================================

    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    # ========================================================
    # TREINAMENTO
    # ========================================================

    history = model.fit(
        X_train,
        y_train,

        validation_split=0.2,

        epochs=100,

        batch_size=4,

        callbacks=[early_stop],

        verbose=1
    )

    # ========================================================
    # PREVISÕES
    # ========================================================

    y_pred = model.predict(X_test)

    y_pred = y_pred.flatten()

    # ========================================================
    # DESNORMALIZAÇÃO
    # ========================================================

    from sklearn.preprocessing import StandardScaler

    # Recria scaler do target

    target_scaler = StandardScaler()

    target_values = pd.read_csv(
        "nba_dados_tratados.csv"
    )['PTS'].values.reshape(-1, 1)

    target_scaler.fit(target_values)

    # Desnormaliza previsões

    y_pred = target_scaler.inverse_transform(
        y_pred.reshape(-1, 1)
    ).flatten()

    # Desnormaliza y_test

    y_test = target_scaler.inverse_transform(
        y_test.reshape(-1, 1)
    ).flatten()
    
    # ========================================================
    # MÉTRICAS
    # ========================================================

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        y_pred
    )

    # ========================================================
    # RESULTADOS
    # ========================================================

    print("\n================================================")
    print("MÉTRICAS")
    print("================================================")

    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

    # ========================================================
    # SALVA RESULTADOS
    # ========================================================

    results.append({
        'Window': window,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    })

    # ========================================================
    # LOSS
    # ========================================================

    plt.figure(figsize=(10, 5))

    plt.plot(
        history.history['loss'],
        label='Treino'
    )

    plt.plot(
        history.history['val_loss'],
        label='Validação'
    )

    plt.title(f'Loss - Window {window}')

    plt.xlabel('Épocas')

    plt.ylabel('Erro')

    plt.legend()

    plt.grid(True)

    plt.show()

    # ========================================================
    # REAL VS PREVISTO
    # ========================================================

    plt.figure(figsize=(12, 6))

    plt.plot(
        y_test,
        label='Real'
    )

    plt.plot(
        y_pred,
        label='Previsto'
    )

    plt.title(f'Real vs Previsto - Window {window}')

    plt.xlabel('Jogos')

    plt.ylabel('Pontos')

    plt.legend()

    plt.grid(True)

    plt.show()

# ============================================================
# RESULTADOS FINAIS
# ============================================================

results_df = pd.DataFrame(results)

print("\n================================================")
print("RESULTADOS FINAIS")
print("================================================")

print(results_df)

# ============================================================
# SALVAMENTO
# ============================================================

results_df.to_csv(
    "resultados_lstm_melhorado.csv",
    index=False
)

print("\nResultados salvos com sucesso!")