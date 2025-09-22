# Exercicio 2 KNN 
Esse exercicio é baseado no dataset [Crash Car](https://www.kaggle.com/datasets/jacksondivakarr/car-crash-dataset) do Kaggle. O objetivo é analisar a base de dados, limpá-la e no final construir um modelo KNN


# Etapa 1 - Analise e instalação dos dados
O dataset foi carregado a partir de um arquivo Excel, contendo informações como tipo de colisão, tipo de lesão, dia da semana, fatores primários do acidente, data, hora e localização.
## Carregando os dados
=== "Saída"

    ```python exec="on" 
    import pandas as pd
    # Ler o arquivo corretamente (é Excel, não CSV)
    df = pd.read_excel("docs/arvore-decisao/crashcar.xlsx")

    # Visualização inicial da base de dados
    print('Formato do dataset:', df.shape)
    ```

=== "Código"

    ```python
    import pandas as pd
    # Ler o arquivo corretamente (apenas )
    df = pd.read_excel("docs/arvore-decisao/crashcar.xlsx")

    # Visualização inicial da base de dados
    print('Formato do dataset:', df.shape)
    ```

O dataset possui 53943 linha e 11 colunas.

## As bibiliotecas utilizadas
```python 
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from io import StringIO

    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
``` 

### Explicação dos tipos de dados 

=== "tipo de dados"

    ```python exec="on"
    import pandas as pd
    df = pd.read_excel("docs/arvore-decisao/crashcar.xlsx")
    for col in df:
        print(col, ":", df[col].dtype, "\n")
    ``` 
=== "Código"

    ```python 
        for col in df:
        print(col, ":", df[col].dtype, "\n")
    ``` 

!!! tip "Explicação"

    Int64: Dados numéricos inteiros, como Year, Month, Day, Hour e Latitude.<br>
    Object: Dados categóricos ou textuais, como Collision Type, Injury Type, Weekend?, Primary Factor e Time.<br>
    FLoat64: Dados numéricos com casas decimais, como Longitude, Latitude e Hora.

### Estatisticas descritivas dos numéricos


=== "Saída"

    ```python exec="on"
    import pandas as pd
    df = pd.read_excel("docs/arvore-decisao/crashcar.xlsx")
    print('Formato do dataset:', df.shape)
    # Estatísticas descritivas das colunas numéricas
    print('Estatísticas das colunas numéricas:<br>')
    print(df[['Year','Month','Day', 'Hour', 'Latitude']].sample(10).to_html())
    ``` 

=== "Código"

    ```python 
    # Estatísticas descritivas das colunas numéricas
    print('Estatísticas das colunas numéricas:')
    print(df[['Year', 'Month', 'Day', 'Hour', 'Latitude']].describe())
    ```



### Visualização dos dados

```python exec="on" html="1"
--8<-- "docs/arvore-decisao/Grafico1.py"
```

Visualizações: Foram criados gráficos de barras para o número de acidentes por tipo de veículo e por gravidade (fatal/não fatal), facilitando a compreensão dos dados.

### Exploração inicial dos dados
```python 
--8<-- "docs/arvore-decisao/Etapa1.py"
```


# Etapa 2 

## Normalização de dados categóricos em numéricos

=== "Tipo de Colisão"

    === "Saída"

        ```python exec="on"
        import pandas as pd
        df = pd.read_excel("docs/arvore-decisao/crashcar.xlsx")
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
        print(df[['Collision Type','Collision Type Num']].head(10).to_html())
        ```

    === "Código"

        ```python
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
        ```


=== "Tipo de Lesão"

    === "Saída"

        ```python exec="on"
        import pandas as pd
        df = pd.read_excel("docs/arvore-decisao/crashcar.xlsx")
        def injury_to_num(injury):
            if injury == 'Fatal':
                return 1
            else:
                return 0
        df['Injury Type Num'] = df['Injury Type'].apply(injury_to_num)
        print(df[['Injury Type','Injury Type Num']].head(10).to_html())
        ```

    === "Código"

        ```python
        def injury_to_num(injury):
            if injury == 'Fatal':
                return 1
            else:
                return 0
    
        df['Injury Type Num'] = df['Injury Type'].apply(injury_to_num)
        ```

=== "Dia da semana"


    === "Saída"

        ```python exec="on"
        import pandas as pd
        df = pd.read_excel("docs/arvore-decisao/crashcar.xlsx")
        def weekend_to_num(value):
            if str(value).lower() == 'weekend':
                return 1
            elif str(value).lower() == 'weekday':
                return 2
            else:
                return 0
        df['Weekend Num'] = df['Weekend?'].apply(weekend_to_num)
        print(df[['Weekend?','Weekend Num']].head(10).to_html())
        ```

    === "Código"

        ```python
        def weekend_to_num(value):
            if str(value).lower() == 'weekend':
                return 1
            elif str(value).lower() == 'weekday':
                return 2
            else:
                return 0
        df['Weekend Num'] = df['Weekend?'].apply(weekend_to_num)
        ```


=== "Motivo lesão"

    === "Saída"

        ```python exec="on" 
        --8<-- "docs/arvore-decisao/Etapa2primary.py"
        ```

    === "Código"

        ```python
        def primary_factor_to_num(factor):
            if factor == 'Erros de julgamento do motorista':
                return 1
            elif factor == 'Velocidade / comportamento arriscado':
                return 2
            elif factor == 'Falhas mecânicas':
                return 3
            elif factor == 'Condições da estrada / ambientais':
                return 4
            elif factor == 'Distrações':
                return 5
            elif factor == 'Uso de Substâncias':
                return 6
            elif factor == 'Fatores diversos':
                return 7
            else:
                return 0  # Outros / não especificado
        df['Primary Factor Num'] = df['Primary Factor'].apply(primary_factor_to_num)
        ```

Conversão de variáveis categóricas em numéricas:
tipo de colisão, tipo de lesão, dia da semana e fator primário foram transformados em variáveis numéricas para facilitar os proximos processos e estabelecer uma normalização.
    

## Limpeza 

=== "Saída"

    ```python exec="on"
    import pandas as pd
    df = pd.read_excel("docs/arvore-decisao/crashcar.xlsx")
    print("Tamanho do dataset antes remoção de valores ausentes")
    print(df.shape)
    print("<br>Tamanho do dataset após remoção de valores ausentes")
    df = df.dropna()
    print(df.shape)
    ```

=== "Código"

    ```python 
    print(df.shape)
    print("Tamanho do dataset antes remoção de valores ausentes")
    df = df.dropna()
    print(df.shape)
    print("Tamanho do dataset após remoção de valores ausentes")
    ```
Todos os registros com valores ausentes foram removidos, garantindo a integridade dos dados para o treinamento do modelo


# Etapa 3 

## Separação em treino e teste / KNN


=== "Saída"

    ```python exec="on" 
    --8<-- "docs/KNN/Etapa3.py"
    ```

=== "Código"

    ```python 
        # Selecionar features para o modelo
        features = ['Year', 'Month', 'Day', 'Hour', 'Collision Type Num', 'Weekend Num', 'Primary Factor Num', 'Latitude', 'Longitude']
        X = df[features]
        y = df['Injury Type Num']  # Variável alvo: 1 para fatal, 0 para não fatal

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
    ```


## Matriz

=== "Saída"

    ```python 
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Matriz de Confusão')
    plt.colorbar()
    tick_marks = range(len(cm))
    plt.xticks(tick_marks, ['Não Fatal', 'Fatal'])
    plt.yticks(tick_marks, ['Não Fatal', 'Fatal'])
    plt.xlabel('Predito')
    plt.ylabel('Real')

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')

    plt.tight_layout()
    buffer = StringIO()
    plt.savefig(buffer, format="svg", transparent=True)
    plt.close()
    print(buffer.getvalue())
    ```

=== "Código"

    ```python exec="on" html="1"
    --8<-- "docs/KNN/Matriz.py"
    ```

## Grafico de Fronteira

=== "Saída"

    ```python exec="on" html="1"
    --8<-- "docs/KNN/fronteira.py"
    ```
=== "Código"

    ```python 
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
    ```
