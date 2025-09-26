import matplotlib.pyplot as plt
import pandas as pd
from io import StringIO, BytesIO
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import tree

df = pd.read_excel("docs/arvore-decisao/crashcar.xlsx")
print(f'Formato do dataset: {df.shape}<br>')
for col in df:
    print(df[col].head(1), "<br>")
print('Valores nulos por coluna:')
print(df.isnull().sum(), "<br>")
print('Estatísticas das colunas numéricas:')
print(df[['Year', 'Month', 'Day', 'Hour', 'Latitude']].describe())

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

def injury_to_num(injury):
    if injury == 'Fatal':
        return 1
    else:
        return 0

def weekend_to_num(value):
    if str(value).lower() == 'weekend':
        return 1
    elif str(value).lower() == 'weekday':
        return 2
    else:
        return 0

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

df['Collision Type Num'] = df['Collision Type'].apply(collision_to_num)
df['Injury Type Num'] = df['Injury Type'].apply(injury_to_num)
df['Weekend Num'] = df['Weekend?'].apply(weekend_to_num)
df['Primary Factor Num'] = df['Primary Factor'].apply(primary_factor_to_num)

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


print("Tamanho do dataset antes remoção de valores ausentes")
print(df.shape)
df = df.dropna()
print("Tamanho do dataset após remoção de valores ausentes")
print(df.shape)

numeric_cols = ['Year', 'Month', 'Day', 'Hour', 'Latitude', 'Collision Type Num', 'Weekend Num']
scaler = StandardScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])


features = ['Injury Type Num', 'Weekend Num', 'Primary Factor Num', 'Year', 'Month', 'Day', 'Hour', 'Latitude']
target = 'Collision Type Num'
x = df[features]
y = df[target]
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

clf = DecisionTreeClassifier(random_state=42, max_depth=5)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"<br>Acurácia do modelo: {accuracy:.4f}")

feature_importance = pd.DataFrame({
    "Feature": x_train.columns,
    "Importância": clf.feature_importances_
}).sort_values(by="Importância", ascending=False)
print("<br>Importância das Features:")
print(feature_importance.to_html() + "<br>")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matriz de Confusão')
plt.colorbar()
tick_marks = range(len(set(y_test)))
plt.xticks(tick_marks, tick_marks)
plt.yticks(tick_marks, tick_marks)
plt.xlabel('Predito')
plt.ylabel('Real')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
plt.tight_layout()
buffer = StringIO()
plt.savefig(buffer, format="svg", transparent=False)
plt.close()
print(buffer.getvalue())

plt.figure(figsize=(18,10), dpi=150)
plot_tree(
    clf,
    feature_names=features,
    class_names=[str(c) for c in clf.classes_],
    filled=True,
    rounded=True,
    fontsize=10
)
buffer = BytesIO()
plt.savefig(buffer, format="svg", transparent=False)
svg_data = buffer.getvalue().decode("utf-8")
print(svg_data)