# ============================================================
# ETAPA 1 — ENGENHARIA DE DADOS PARA LSTM NBA
# Disciplina: Aprendizado de Máquina
# Objetivo:
# Preparar os dados da NBA para modelos LSTM
# ============================================================

# ============================================================
# 1. IMPORTAÇÃO DAS BIBLIOTECAS
# ============================================================

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

# ============================================================
# 2. CARREGAMENTO DOS DADOS
# ============================================================

# Substitua pelo nome correto do arquivo CSV
# Recomenda-se converter o PDF para CSV antes

df = pd.read_csv("BaseA.csv")

# ============================================================
# 3. INSPEÇÃO INICIAL DOS DADOS
# ============================================================

print("\n================ HEAD =================")
print(df.head())

print("\n================ COLUNAS =================")
print(df.columns)

print("\n================ INFO =================")
print(df.info())

print("\n================ SHAPE =================")
print(df.shape)

# ============================================================
# 4. REMOÇÃO DE COLUNAS DESNECESSÁRIAS
# ============================================================

# Algumas colunas não ajudam na previsão

cols_to_drop = [
    'VIDEO_AVAILABLE'
]

df = df.drop(columns=cols_to_drop, errors='ignore')

# ============================================================
# 5. CONVERSÃO DE DATA
# ============================================================

# Converte para formato datetime

df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

# ============================================================
# 6. ORDENAÇÃO TEMPORAL
# ============================================================

# MUITO IMPORTANTE PARA LSTM

df = df.sort_values(
    by=['TEAM_ABBREVIATION', 'GAME_DATE']
)

# ============================================================
# 7. ESCOLHA DAS EQUIPES
# ============================================================

# Escolha das equipes dos playoffs

teams = ['BOS', 'OKC']

df = df[df['TEAM_ABBREVIATION'].isin(teams)]

# ============================================================
# 8. TRANSFORMAÇÃO DE VITÓRIA/DERROTA
# ============================================================

# W -> 1
# L -> 0

df['WL'] = df['WL'].map({
    'W': 1,
    'L': 0
})

# ============================================================
# 9. VARIÁVEL CASA/FORA
# ============================================================

# "vs." = jogo em casa
# "@" = jogo fora

df['HOME'] = df['MATCHUP'].apply(
    lambda x: 1 if 'vs.' in x else 0
)

# ============================================================
# 10. DIAS DE DESCANSO
# ============================================================

# Diferença de dias entre partidas

df['REST_DAYS'] = (
    df.groupby('TEAM_ABBREVIATION')['GAME_DATE']
    .diff()
    .dt.days
)

# Preenche o primeiro jogo da equipe

df['REST_DAYS'] = df['REST_DAYS'].fillna(2)

# ============================================================
# 11. MÉDIAS MÓVEIS
# ============================================================

# ------------------------------------------------------------
# Pontos
# ------------------------------------------------------------

df['PTS_MA5'] = (
    df.groupby('TEAM_ABBREVIATION')['PTS']
    .transform(lambda x: x.rolling(5).mean())
)

df['PTS_MA10'] = (
    df.groupby('TEAM_ABBREVIATION')['PTS']
    .transform(lambda x: x.rolling(10).mean())
)

# ------------------------------------------------------------
# Rebotes
# ------------------------------------------------------------

df['REB_MA5'] = (
    df.groupby('TEAM_ABBREVIATION')['REB']
    .transform(lambda x: x.rolling(5).mean())
)

# ------------------------------------------------------------
# Assistências
# ------------------------------------------------------------

df['AST_MA5'] = (
    df.groupby('TEAM_ABBREVIATION')['AST']
    .transform(lambda x: x.rolling(5).mean())
)

# ------------------------------------------------------------
# Aproveitamento de arremessos
# ------------------------------------------------------------

df['FG_PCT_MA5'] = (
    df.groupby('TEAM_ABBREVIATION')['FG_PCT']
    .transform(lambda x: x.rolling(5).mean())
)

# ------------------------------------------------------------
# Aproveitamento de 3 pontos
# ------------------------------------------------------------

df['FG3_PCT_MA5'] = (
    df.groupby('TEAM_ABBREVIATION')['FG3_PCT']
    .transform(lambda x: x.rolling(5).mean())
)

# ------------------------------------------------------------
# Turnovers
# ------------------------------------------------------------

df['TOV_MA5'] = (
    df.groupby('TEAM_ABBREVIATION')['TOV']
    .transform(lambda x: x.rolling(5).mean())
)

# ============================================================
# 12. TENDÊNCIA OFENSIVA
# ============================================================

# Diferença entre média curta e média longa

df['PTS_TREND'] = (
    df['PTS_MA5'] - df['PTS_MA10']
)

# ============================================================
# 13. SEQUÊNCIA DE VITÓRIAS
# ============================================================

df['WIN_STREAK'] = (
    df.groupby('TEAM_ABBREVIATION')['WL']
    .transform(
        lambda x: (
            x.groupby(
                (x != x.shift()).cumsum()
            ).cumsum()
        )
    )
)

# ============================================================
# 14. TARGETS PARA RF1
# ============================================================

# ------------------------------------------------------------
# Média de pontos por equipe
# ------------------------------------------------------------

team_avg_pts = (
    df.groupby('TEAM_ABBREVIATION')['PTS']
    .transform('mean')
)

df['PTS_ABOVE_AVG'] = (
    df['PTS'] > team_avg_pts
).astype(int)

# ------------------------------------------------------------
# Média de rebotes por equipe
# ------------------------------------------------------------

team_avg_reb = (
    df.groupby('TEAM_ABBREVIATION')['REB']
    .transform('mean')
)

df['REB_ABOVE_AVG'] = (
    df['REB'] > team_avg_reb
).astype(int)

# ------------------------------------------------------------
# Média de assistências por equipe
# ------------------------------------------------------------

team_avg_ast = (
    df.groupby('TEAM_ABBREVIATION')['AST']
    .transform('mean')
)

df['AST_ABOVE_AVG'] = (
    df['AST'] > team_avg_ast
).astype(int)

# ============================================================
# 15. TARGETS PARA RF2
# ============================================================

# ------------------------------------------------------------
# Acima de 100 pontos
# ------------------------------------------------------------

df['PTS_100'] = (
    df['PTS'] >= 100
).astype(int)

# ------------------------------------------------------------
# Acima de 30 rebotes
# ------------------------------------------------------------

df['REB_30'] = (
    df['REB'] >= 30
).astype(int)

# ------------------------------------------------------------
# Acima de 20 assistências
# ------------------------------------------------------------

df['AST_20'] = (
    df['AST'] >= 20
).astype(int)

# ============================================================
# 16. REMOÇÃO DE VALORES NULOS
# ============================================================

# Rolling windows geram NaN nos primeiros jogos

df = df.dropna().reset_index(drop=True)

# ============================================================
# 17. DEFINIÇÃO DAS FEATURES
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
# 18. TARGETS DE REGRESSÃO
# ============================================================

targets_regression = [
    'PTS',
    'REB',
    'AST'
]

# ============================================================
# 19. TARGETS DE CLASSIFICAÇÃO
# ============================================================

targets_classification = [
    'PTS_ABOVE_AVG',
    'REB_ABOVE_AVG',
    'AST_ABOVE_AVG',
    'PTS_100',
    'REB_30',
    'AST_20'
]

# ============================================================
# 20. ESCALONAMENTO DOS DADOS
# ============================================================

# LSTM funciona melhor com dados escalonados

scaler = StandardScaler()

df_scaled = df.copy()

df_scaled[features] = scaler.fit_transform(
    df_scaled[features]
)

# ============================================================
# 21. INSPEÇÃO FINAL
# ============================================================

print("\n================ DATAFRAME FINAL =================")
print(df_scaled.head())

print("\n================ SHAPE FINAL =================")
print(df_scaled.shape)

print("\n================ FEATURES =================")
print(features)

print("\n================ TARGETS REGRESSÃO =================")
print(targets_regression)

print("\n================ TARGETS CLASSIFICAÇÃO =================")
print(targets_classification)

# ============================================================
# 22. SALVAMENTO DOS DADOS TRATADOS
# ============================================================

# Salva os dados prontos para próximas etapas

df_scaled.to_csv(
    "nba_dados_tratados.csv",
    index=False
)

print("\nDados tratados salvos com sucesso!")