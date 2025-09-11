import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn.preprocessing import StandardScaler

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

##########################################
# Transformando Collision Type em numérico
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
# Transformando Injury Type em numérico
def injury_to_num(injury):
    if injury == 'Fatal':
        return 1
    else:
        return 0
df['Injury Type Num'] = df['Injury Type'].apply(injury_to_num)
# Transformando Weekend? em numérico 
def weekend_to_num(value):
    if str(value).lower() == 'weekend':
        return 1
    elif str(value).lower() == 'weekday':
        return 2
    else:
        return 0
df['Weekend Num'] = df['Weekend?'].apply(weekend_to_num)

def primary_factor_to_num(factor):
    if factor == "FAILURE TO YIELD RIGHT OF WAY" or \
        factor == "FOLLOWING TOO CLOSELY" or \
        factor == "IMPROPER TURNING" or \
        factor == "UNSAFE BACKING" or \
        factor == "RAN OFF ROAD RIGHT" or \
        factor == "DISREGARD SIGNAL/REG SIGN" or \
        factor == "LEFT OF CENTER" or \
        factor == "IMPROPER LANE USAGE" or \
        factor == "UNSAFE LANE MOVEMENT" or \
        factor == "OVERCORRECTING/OVERSTEERING" or \
        factor == "IMPROPER PASSING" or \
        factor == "WRONG WAY ON ONE WAY" or \
        factor == "VIOLATION OF LICENSE RESTRICTION":
        return 1

    elif factor == 'SPEED TOO FAST FOR WEATHER CONDITIONS' or \
            factor == 'UNSAFE SPEED':
        return 2

    elif factor == "BRAKE FAILURE OR DEFECTIVE" or \
            factor == "TIRE FAILURE OR DEFECTIVE" or \
            factor == "ACCELERATOR FAILURE OR DEFECTIVE" or \
            factor == "STEERING FAILURE" or \
            factor == "ENGINE FAILURE OR DEFECTIVE" or \
            factor == "HEADLIGHT DEFECTIVE OR NOT ON" or \
            factor == "OTHER LIGHTS DEFECTIVE" or \
            factor == "TOW HITCH FAILURE":
        return 3

    elif factor == "ROADWAY SURFACE CONDITION" or \
            factor == "GLARE" or \
            factor == "HOLES/RUTS IN SURFACE" or \
            factor == "TRAFFIC CONTROL INOPERATIVE/MISSING/OBSC" or \
            factor == "ROAD UNDER CONSTRUCTION" or \
            factor == "SHOULDER DEFECTIVE" or \
            factor == "LANE MARKING OBSCURED" or \
            factor == "UTILITY WORK" or \
            factor == "SEVERE CROSSWINDS":
        return 4

    elif factor == "DRIVER DISTRACTED - EXPLAIN IN NARRATIVE" or \
            factor == "CELL PHONE USAGE" or \
            factor == "PASSENGER DISTRACTION":
        return 5

    elif factor == "ALCOHOLIC BEVERAGES" or \
            factor == "PRESCRIPTION DRUGS" or \
            factor == "ILLEGAL DRUGS" or \
            factor == "DRIVER ASLEEP OR FATIGUED" or \
            factor == "DRIVER ILLNESS":
        return 6

    elif factor == "ANIMAL/OBJECT IN ROADWAY" or \
            factor == "SEVERE CROSSWINDS" or \
            factor == "INSECURE/LEAKY LOAD" or \
            factor == "OVERSIZE/OVERWEIGHT LOAD":
        return 7 

    else:
        return 0

df['Primary Factor Num'] = df['Primary Factor'].apply(primary_factor_to_num)

print(df[['Primary Factor','Primary Factor Num']].head(10).to_html())

##########################################
#Limpeza
print(df.shape)
print("Tamanho do dataset antes remoção de valores ausentes")
print(df.dropna().shape)
print("Tamanho do dataset após remoção de valores ausentes")
df = df.dropna()


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

