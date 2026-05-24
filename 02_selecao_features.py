# ============================================================
# ETAPA 2 — SELEÇÃO AUTOMÁTICA DE FEATURES
# Disciplina: Aprendizado de Máquina
# Objetivo:
# Selecionar as melhores variáveis utilizando:
# - P-Value
# - Random Forest Importance
# ============================================================

# ============================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================================

import pandas as pd
import numpy as np

import statsmodels.api as sm

from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt

# ============================================================
# 2. CARREGAMENTO DOS DADOS TRATADOS
# ============================================================

df = pd.read_csv("nba_dados_tratados.csv")

# ============================================================
# 3. DEFINIÇÃO DAS FEATURES
# ============================================================

features = [
    'FG_PCT',
    'FG3_PCT',
    'FTA',
    'AST',
    'REB',
    'TOV',
    'HOME',
    'REST_DAYS',
    'WIN_STREAK',
    'PTS_MA5',
    'PTS_MA10',
    'REB_MA5',
    'AST_MA5',
    'FG_PCT_MA5',
    'FG3_PCT_MA5',
    'TOV_MA5',
    'PTS_TREND'
]

# ============================================================
# 4. DEFINIÇÃO DO TARGET
# ============================================================

# Inicialmente faremos seleção de features
# para previsão de pontos (PTS)

target = 'PTS'

# ============================================================
# 5. SEPARAÇÃO DOS DADOS
# ============================================================

X = df[features]

y = df[target]

# ============================================================
# 6. ADIÇÃO DA CONSTANTE
# ============================================================

# Necessário para regressão linear do statsmodels

X_const = sm.add_constant(X)

# ============================================================
# 7. REGRESSÃO LINEAR
# ============================================================

model = sm.OLS(y, X_const).fit()

# ============================================================
# 8. RESUMO ESTATÍSTICO
# ============================================================

print("\n================ RESUMO OLS =================")

print(model.summary())

# ============================================================
# 9. EXTRAÇÃO DOS P-VALUES
# ============================================================

p_values = model.pvalues

# ============================================================
# 10. ORGANIZAÇÃO DOS P-VALUES
# ============================================================

pvalue_df = pd.DataFrame({
    'Feature': p_values.index,
    'P_Value': p_values.values
})

# Remove a constante

pvalue_df = pvalue_df[
    pvalue_df['Feature'] != 'const'
]

# Ordena pelos menores p-values

pvalue_df = pvalue_df.sort_values(
    by='P_Value'
)

# ============================================================
# 11. EXIBIÇÃO DOS P-VALUES
# ============================================================

print("\n================ P-VALUES =================")

print(pvalue_df)

# ============================================================
# 12. FEATURES SIGNIFICATIVAS
# ============================================================

# Critério:
# p-value < 0.05

significant_features = pvalue_df[
    pvalue_df['P_Value'] < 0.05
]

print("\n================ FEATURES SIGNIFICATIVAS =================")

print(significant_features)

# ============================================================
# 13. RANDOM FOREST REGRESSOR
# ============================================================

rf = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

# ============================================================
# 14. TREINAMENTO DO RANDOM FOREST
# ============================================================

rf.fit(X, y)

# ============================================================
# 15. IMPORTÂNCIA DAS FEATURES
# ============================================================

importance = rf.feature_importances_

# ============================================================
# 16. DATAFRAME DE IMPORTÂNCIA
# ============================================================

importance_df = pd.DataFrame({
    'Feature': features,
    'Importance': importance
})

# Ordena pela importância

importance_df = importance_df.sort_values(
    by='Importance',
    ascending=False
)

# ============================================================
# 17. EXIBIÇÃO DAS IMPORTÂNCIAS
# ============================================================

print("\n================ RANDOM FOREST IMPORTANCE =================")

print(importance_df)

# ============================================================
# 18. GRÁFICO DE IMPORTÂNCIA
# ============================================================

plt.figure(figsize=(12, 8))

plt.barh(
    importance_df['Feature'],
    importance_df['Importance']
)

plt.xlabel("Importância")

plt.ylabel("Features")

plt.title("Random Forest Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.show()

# ============================================================
# 19. COMBINAÇÃO DOS MÉTODOS
# ============================================================

# Features:
# - estatisticamente significativas
# - importantes no Random Forest

significant_list = significant_features[
    'Feature'
].tolist()

importance_threshold = 0.03

important_rf = importance_df[
    importance_df['Importance'] > importance_threshold
]

important_rf_list = important_rf[
    'Feature'
].tolist()

# Interseção entre os métodos

final_features = list(
    set(significant_list)
    &
    set(important_rf_list)
)

# ============================================================
# 20. FEATURES FINAIS
# ============================================================

print("\n================ FEATURES FINAIS =================")

print(final_features)

# ============================================================
# 21. SALVAMENTO DAS FEATURES
# ============================================================

final_features_df = pd.DataFrame({
    'Selected_Features': final_features
})

final_features_df.to_csv(
    "features_selecionadas.csv",
    index=False
)

print("\nFeatures selecionadas salvas com sucesso!")

# ============================================================
# 22. SALVAMENTO DOS RANKINGS
# ============================================================

pvalue_df.to_csv(
    "ranking_pvalues.csv",
    index=False
)

importance_df.to_csv(
    "ranking_random_forest.csv",
    index=False
)

print("\nRankings salvos com sucesso!")

# ============================================================
# 23. CONCLUSÃO
# ============================================================

print("\n================ PROCESSO FINALIZADO =================")

print(f"Quantidade inicial de features: {len(features)}")

print(f"Quantidade final de features: {len(final_features)}")