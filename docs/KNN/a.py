import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
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


# Implementação do KNN Classifier
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
        # Compute Euclidean distances
        distances = [np.sqrt(np.sum((x - x_train)**2)) for x_train in self.X_train]
        # Get indices of k-nearest neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Get corresponding labels
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return majority class
        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        return most_common

# Preparação dos dados para o KNN
# Selecionando features relevantes
features = ['Year', 'Month', 'Day', 'Hour', 'Latitude', 'Longitude', 
            'Collision Type Num', 'Weekend Num', 'Primary Factor Num']
target = 'Injury Type Num'

X = df[features].values
y = df[target].values

# Normalização dos dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Divisão treino-teste
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Treinamento e previsão com KNN personalizado
knn = KNNClassifier(k=5)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

# Métricas de avaliação
accuracy = accuracy_score(y_test, predictions)
print(f"Acurácia do KNN personalizado: {accuracy:.4f}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, predictions))
print("\nMatriz de Confusão:")
print(confusion_matrix(y_test, predictions))

# Visualização da matriz de confusão
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusão - KNN')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()

# Gráfico de importância das features (baseado na correlação)
plt.figure(figsize=(10, 6))
correlations = df[features + [target]].corr()[target].drop(target)
correlations.sort_values().plot(kind='barh')
plt.title('Correlação das Features com Injury Type')
plt.xlabel('Coeficiente de Correlação')
plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()

# Teste com diferentes valores de K
k_values = range(1, 15)
accuracies = []

for k in k_values:
    knn_temp = KNNClassifier(k=k)
    knn_temp.fit(X_train, y_train)
    pred_temp = knn_temp.predict(X_test)
    acc = accuracy_score(y_test, pred_temp)
    accuracies.append(acc)

# Gráfico de acurácia vs valor de K
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, marker='o', linestyle='-', color='b')
plt.title('Acurácia vs Valor de K')
plt.xlabel('Valor de K')
plt.ylabel('Acurácia')
plt.grid(True)
plt.xticks(k_values)
plt.tight_layout()

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()

# Comparação com sklearn KNN
from sklearn.neighbors import KNeighborsClassifier

sklearn_knn = KNeighborsClassifier(n_neighbors=5)
sklearn_knn.fit(X_train, y_train)
sklearn_pred = sklearn_knn.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, sklearn_pred)

print(f"\nComparação com sklearn:")
print(f"Acurácia KNN personalizado: {accuracy:.4f}")
print(f"Acurácia sklearn KNN: {sklearn_accuracy:.4f}")

# Análise de distribuição das classes
plt.figure(figsize=(8, 6))
df[target].value_counts().plot(kind='bar')
plt.title('Distribuição das Classes (Injury Type)')
plt.xlabel('Tipo de Lesão (0=Não Fatal, 1=Fatal)')
plt.ylabel('Frequência')
plt.xticks(rotation=0)

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()

# Gráfico adicional: Distribuição dos tipos de colisão
plt.figure(figsize=(10, 6))
df['Collision Type Num'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribuição dos Tipos de Colisão')
plt.xlabel('Tipo de Colisão')
plt.ylabel('Frequência')
plt.xticks(rotation=45)

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()

# Gráfico adicional: Distribuição dos fatores primários
plt.figure(figsize=(12, 6))
df['Primary Factor Num'].value_counts().sort_index().plot(kind='bar')
plt.title('Distribuição dos Fatores Primários')
plt.xlabel('Categoria do Fator Primário')
plt.ylabel('Frequência')
plt.xticks(rotation=45)

buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=True)
print(buffer.getvalue())
plt.close()