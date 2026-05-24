# ============================================================
# ETAPA 5 — LSTM PARA CLASSIFICAÇÃO
# RF2:
# Prever se a equipe fará:
# - acima de 100 pontos
# - acima de 30 rebotes
# - acima de 20 assistências
# ============================================================

# ============================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================================

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
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
# 2. CARREGAMENTO DOS DADOS
# ============================================================

df = pd.read_csv("nba_dados_tratados.csv")

# ============================================================
# 3. FEATURES
# ============================================================

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
# 4. TARGETS DE CLASSIFICAÇÃO
# ============================================================

classification_targets = [
    'PTS_100',
    'REB_30',
    'AST_20'
]

# ============================================================
# 5. TAMANHO DA JANELA
# ============================================================

# Utilizaremos a melhor janela encontrada

WINDOW_SIZE = 20

# ============================================================
# 6. FUNÇÃO PARA CRIAR SEQUÊNCIAS
# ============================================================

def create_sequences(X, y, window_size):
    
    X_seq = []
    y_seq = []
    
    for i in range(window_size, len(X)):
        
        X_seq.append(
            X[i-window_size:i]
        )
        
        y_seq.append(
            y[i]
        )
    
    return np.array(X_seq), np.array(y_seq)

# ============================================================
# 7. LOOP DOS TARGETS
# ============================================================

results = []

for target in classification_targets:
    
    print("\n================================================")
    print(f"TARGET = {target}")
    print("================================================")
    
    # ========================================================
    # DEFINIÇÃO DOS DADOS
    # ========================================================
    
    X_data = df[selected_features].values
    
    y_data = df[target].values
    
    # ========================================================
    # CRIAÇÃO DAS SEQUÊNCIAS
    # ========================================================
    
    X_seq, y_seq = create_sequences(
        X_data,
        y_data,
        WINDOW_SIZE
    )
    
    # ========================================================
    # DIVISÃO TREINO/TESTE
    # ========================================================
    
    split_index = int(len(X_seq) * 0.8)
    
    X_train = X_seq[:split_index]
    y_train = y_seq[:split_index]
    
    X_test = X_seq[split_index:]
    y_test = y_seq[split_index:]
    
    # ========================================================
    # EXIBE FORMATOS
    # ========================================================
    
    print("\nX_train:")
    print(X_train.shape)
    
    print("\ny_train:")
    print(y_train.shape)
    
    # ========================================================
    # MODELO
    # ========================================================
    
    model = Sequential()
    
    # --------------------------------------------------------
    # LSTM
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
    # CAMADA DE SAÍDA
    # --------------------------------------------------------
    
    model.add(
        Dense(
            1,
            activation='sigmoid'
        )
    )
    
    # ========================================================
    # COMPILAÇÃO
    # ========================================================
    
    optimizer = Adam(
        learning_rate=0.001
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
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
    
    y_pred_prob = model.predict(X_test)
    
    # Converte probabilidades para classes
    
    y_pred = (
        y_pred_prob > 0.5
    ).astype(int)
    
    # Flatten
    
    y_pred = y_pred.flatten()
    
    # ========================================================
    # MÉTRICAS
    # ========================================================
    
    accuracy = accuracy_score(
        y_test,
        y_pred
    )
    
    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )
    
    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )
    
    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )
    
    # ========================================================
    # RESULTADOS
    # ========================================================
    
    print("\n================================================")
    print("MÉTRICAS")
    print("================================================")
    
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    
    # ========================================================
    # MATRIZ DE CONFUSÃO
    # ========================================================
    
    cm = confusion_matrix(
        y_test,
        y_pred
    )
    
    print("\nMatriz de confusão:")
    print(cm)
    
    # ========================================================
    # RELATÓRIO
    # ========================================================
    
    print("\nClassification Report:")
    
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )
    
    # ========================================================
    # SALVA RESULTADOS
    # ========================================================
    
    results.append({
        'Target': target,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })
    
    # ========================================================
    # GRÁFICO DA LOSS
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
    
    plt.title(f'Loss - {target}')
    
    plt.xlabel('Épocas')
    
    plt.ylabel('Binary Crossentropy')
    
    plt.legend()
    
    plt.grid(True)
    
    plt.show()
    
    # ========================================================
    # GRÁFICO DE ACCURACY
    # ========================================================
    
    plt.figure(figsize=(10, 5))
    
    plt.plot(
        history.history['accuracy'],
        label='Treino'
    )
    
    plt.plot(
        history.history['val_accuracy'],
        label='Validação'
    )
    
    plt.title(f'Accuracy - {target}')
    
    plt.xlabel('Épocas')
    
    plt.ylabel('Accuracy')
    
    plt.legend()
    
    plt.grid(True)
    
    plt.show()

# ============================================================
# 8. RESULTADOS FINAIS
# ============================================================

results_df = pd.DataFrame(results)

print("\n================================================")
print("RESULTADOS FINAIS")
print("================================================")

print(results_df)

print(df['PTS_100'].value_counts())

print(df['REB_30'].value_counts())

print(df['AST_20'].value_counts())

# ============================================================
# 9. SALVAMENTO
# ============================================================

results_df.to_csv(
    "resultados_classificacao.csv",
    index=False
)

print("\nResultados salvos com sucesso!")

# ============================================================
# 10. MELHOR MODELO
# ============================================================

best_model = results_df.sort_values(
    by='Accuracy',
    ascending=False
).iloc[0]

print("\n================================================")
print("MELHOR TARGET")
print("================================================")

print(best_model)

# ============================================================
# 11. FINALIZAÇÃO
# ============================================================

print("\n================================================")
print("PROCESSO FINALIZADO COM SUCESSO!")
print("================================================")