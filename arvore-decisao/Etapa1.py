import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_excel("docs/arvore-decisao/crashcar.xlsx")

# Visualização inicial da base de dados
print(f'Formato do dataset: {df.shape}<br>')

# Explicação de tipo de dado em cada coluna
for col in df:
    print(df[col].head(1), "<br>")
print('Valores nulos por coluna:')
print(df.isnull().sum(), "<br>")


################################################
# Estatísticas descritivas das colunas numéricas
print('Estatísticas das colunas numéricas:')
print(df[['Year', 'Month', 'Day', 'Hour', 'Latitude']].describe())
