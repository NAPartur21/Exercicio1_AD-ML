import numpy as np
import matplotlib.pyplot as plt
from io import StringIO
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

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
y = df['Injury Type Num']  # Variável alvo: 1 para fatal, 0 para não fatal
# Verificar balanceamento das classes
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

plt.figure(figsize=(12, 10))

# Gerar dataset sintético
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinar modelo KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(f"Acurácia: {accuracy_score(y_test, predictions):.2f}")

# Visualizar fronteira de decisão
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.3)
for label in np.unique(y):
    plt.scatter(X[y == label, 0], X[y == label, 1], label=f"Classe {label}", s=100)
plt.xlabel("Chance de acidente fatal")
plt.ylabel("Chance de acidente não fatal")
plt.title("Fronteira de Decisão KNN ")
plt.legend()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
plt.close()

print(buffer.getvalue())