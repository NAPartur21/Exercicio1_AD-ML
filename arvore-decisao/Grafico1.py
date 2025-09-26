import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn.preprocessing import StandardScaler
df = pd.read_excel("docs/arvore-decisao/crashcar.xlsx")
df.info()
def collision_to_num(collision):
    if collision == '1-Car' or collision == '2-Car' or collision == '3+ Cars':
        return 1
    elif collision == 'Moped/Motorcycle':
        return 2
    elif collision == 'Bus':
        return 3
    elif collision == 'Pedestrian':
        return 4
    elif collision == 'Cyclist':
        return 5
    else:
        return 0
df['Collision Type Num'] = df['Collision Type'].apply(collision_to_num)
def injury_to_num(injury):
    if injury == 'Fatal':
        return 1
    else:
        return 0
df['Injury Type Num'] = df['Injury Type'].apply(injury_to_num)
def weekend_to_num(value):
    if str(value).lower() == 'weekend':
        return 1
    elif str(value).lower() == 'weekday':
        return 2
    else:
        return 0
df['Weekend Num'] = df['Weekend?'].apply(weekend_to_num)
numeric_cols = ['Year', 'Month', 'Day', 'Hour', 'Latitude', 'Collision Type Num', 'Weekend Num']
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


#Grafico de acidentes por tipo de veículo
collision_counts = df['Collision Type'].value_counts()
plt.figure(figsize=(10,10))
collision_counts.plot(kind='bar', color='skyblue')
plt.title('Quantidade de Acidentes por Tipo de Veículo')
plt.xlabel('Tipo de Veículo')
plt.ylabel('Número de Acidentes')
plt.xticks(rotation=45)
plt.tight_layout()
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
plt.close()
print(buffer.getvalue())

#Contagem de acidentes fatais e não fatais
injury_counts = df['Injury Type Num'].value_counts()
labels = ['Não Fatal', 'Fatal']
plt.figure(figsize=(10,12))
plt.bar(labels, injury_counts, color=['green', 'red'])
plt.title('Quantidade de Acidentes Fatais e Não Fatais')
plt.ylabel('Número de Acidentes')
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
plt.close()
print(buffer.getvalue())