import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np
df = pd.read_excel("docs/arvore-decisao/crashcar.xlsx")
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
##########################################
#Limpeza
df = df.dropna()
##########################################
# Preparando dados para o K-Means
# Seleciona colunas numéricas já transformadas
features = ['Year', 'Month', 'Day', 'Hour', 'Latitude',
            'Collision Type Num', 'Injury Type Num',
            'Weekend Num', 'Primary Factor Num']
X = df[features]
# Padronização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
##########################################
# Aplicando o modelo K-Means
# Definindo número de clusters (ex: 3, mas pode ajustar)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
kmeans.fit(X_scaled)
# Atribui os clusters ao dataframe
df['Cluster'] = kmeans.labels_

##########################################
# Visualização simples (2D usando Latitude x Hour)
plt.figure(figsize=(10,8))
plt.scatter(df['Latitude'], df['Hour'], c=df['Cluster'], cmap='viridis', alpha=0.5)
plt.xlabel('Latitude')
plt.ylabel('Hour')
plt.title('Clusters de Acidentes (Latitude vs Hora)')
plt.colorbar(label='Cluster')
plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
plt.close()
print(buffer.getvalue())