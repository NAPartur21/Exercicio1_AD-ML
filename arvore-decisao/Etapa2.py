import matplotlib.pyplot as plt
import pandas as pd
# Ler o arquivo corretamente (é Excel, não CSV)
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

# Etapa2primary está a normalização da coluna Primary Factor

##########################################
#Limpeza
print(df.shape)
print("Tamanho do dataset antes remoção de valores ausentes")
print(df.dropna().shape)
print("Tamanho do dataset após remoção de valores ausentes")
df = df.dropna()



