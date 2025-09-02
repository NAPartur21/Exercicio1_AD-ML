import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

df = pd.read_excel("docs/arvore-decisao/crashcar.xlsx")
#Transformando Collision Type em numérico
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
#Transformando Injury Type em numérico
def injury_to_num(injury):
    if injury == 'Fatal':
        return 1
    else:
        return 0
df['Injury Type Num'] = df['Injury Type'].apply(injury_to_num)
#Transformando Weekend? em numérico 
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

#Limpeza
print(df.shape)
df = df.dropna()


#Definindo features 
features = ['Injury Type Num', 'Weekend Num', 'Primary Factor Num', 'Year', 'Month', 'Day', 'Hour', 'Latitude']
#coluna que vamos prever com a árvore de decisão
target = 'Collision Type Num' 

x = df[features]
y = df[target]

#Dividindo em treino e teste 80% treino 20% teste
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

print(f"<br>Treino: {x_train.shape[0]} amostras")
print(f"<br>Teste: {x_test.shape[0]} amostras")
print(f"<br>Proporção: {x_train.shape[0]/x.shape[0]*100:.1f}% treino, {x_test.shape[0]/x.shape[0]*100:.1f}% teste\n")

print("Distribuição das classes - Treino:\n")
print(y_train.value_counts().to_markdown(), "\n")

print("Distribuição das classes - Teste:\n")
print(y_test.value_counts().to_markdown(), "\n")


