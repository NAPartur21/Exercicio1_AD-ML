import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier

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


# Selecionar features para o modelo
features = ['Year', 'Month', 'Day', 'Hour', 'Collision Type Num', 'Weekend Num', 'Primary Factor Num', 'Latitude', 'Longitude']
X = df[features]
y = df['Injury Type Num']  

# Verificar balanceamento das classes
print("Distribuição das classes:<br>")

print(y.value_counts())
print(f"<br>Proporção de acidentes fatais: {y.mean():.4f}<br>")

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Implementação do KNN
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # Calcular distâncias euclidianas
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        # Obter índices dos k-vizinhos mais próximos
        k_indices = np.argsort(distances)[:self.k]
        # Obter os rótulos correspondentes
        k_nearest_labels = [self.y_train.iloc[i] for i in k_indices]
        # Retornar a classe majoritária
        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        return most_common

# Treinar e avaliar o modelo

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
predictions = knn.predict(X_test_scaled)
# Métricas de avaliação
accuracy = accuracy_score(y_test, predictions)
print(f"Acurácia do modelo: {accuracy:.4f}<br>")
print("\nRelatório de Classificação:<br>")
print(classification_report(y_test, predictions, target_names=['Não Fatal', 'Fatal']))