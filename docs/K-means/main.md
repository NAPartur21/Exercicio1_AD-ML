# Exercicio 3 - K-means 
Esse exercicio é baseado no dataset [Crash Car](https://www.kaggle.com/datasets/jacksondivakarr/car-crash-dataset) do Kaggle. O objetivo é analisar a base de dados, limpá-la e no final construir um modelo K-means


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
## Separação em treino e teste / K-means


=== "Saída"

    ```python exec="on" 
    --8<-- "docs/K-means/Etapa3.py"
    ```

=== "Código"

    ```python
    ##########################################
    # Aplicando o modelo K-Means

    # Definindo número de clusters (ex: 3, mas pode ajustar)
    kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(X_scaled)

    # Atribui os clusters ao dataframe
    df['Cluster'] = kmeans.labels_
    ```

## Grafico simples

=== "Saída"

    ```python exec="on" html="1"
    --8<-- "docs/K-means/Simples.py"
    ```

## Grafico cluster 
=== "Saída"

    ```python exec="on" html="1"
    --8<-- "docs/K-means/cluster.py"
    ```

