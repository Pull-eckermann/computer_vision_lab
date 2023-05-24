import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

# Carregar o conjunto de dados Iris
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
iris_data = pd.read_csv(url, names=names)

# Média
mean_values = iris_data.mean()
print("Média geral:\n", mean_values)
mean_values_by_category = iris_data.groupby('class').mean()
print("\nMédia por categoria:\n", mean_values_by_category)

# Desvio padrão
std_values = iris_data.std()
print("\nDesvio padrão geral:\n", std_values)
std_values_by_category = iris_data.groupby('class').std()
print("\nDesvio padrão por categoria:\n", std_values_by_category)

# Moda
mode_values = iris_data.mode().iloc[0]
print("\nModa geral:\n", mode_values)
mode_values_by_category = iris_data.groupby('class').apply(lambda x: x.mode().iloc[0])
print("\nModa por categoria:\n", mode_values_by_category)

# Frequência das categorias
category_counts = iris_data['class'].value_counts()
print("\nFrequência das categorias:\n", category_counts)

# Gráfico de dispersão da relação entre comprimento e largura da sépala
plt.figure(figsize=(10, 6))
for category in iris_data['class'].unique():
    subset = iris_data[iris_data['class'] == category]
    plt.scatter(subset['sepal_length'], subset['sepal_width'], label=category)
plt.xlabel('Comprimento da Sépala')
plt.ylabel('Largura da Sépala')
plt.legend()
plt.show()

# Gráfico de dispersão da relação entre comprimento e largura da pétala
plt.figure(figsize=(10, 6))
for category in iris_data['class'].unique():
    subset = iris_data[iris_data['class'] == category]
    plt.scatter(subset['petal_length'], subset['petal_width'], label=category)
plt.xlabel('Comprimento da Pétala')
plt.ylabel('Largura da Pétala')
plt.legend()
plt.show()

# Gráficos de caixa das dimensões
plt.figure(figsize=(10, 6))
iris_data.boxplot(column=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'], by='class', grid=False)
plt.suptitle

# Separar os atributos das classes
X = iris_data.iloc[:, :-1]
y = iris_data['class']

# Aplicar o PCA com dois componentes
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Criar um DataFrame para os resultados do PCA
pca_df = pd.DataFrame(data=X_pca, columns=['PC1', 'PC2'])
pca_df['class'] = y

# Gráfico de dispersão dos componentes principais
plt.figure(figsize=(10, 6))
sns.scatterplot(data=pca_df, x='PC1', y='PC2', hue='class')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.title('PCA - Conjunto de Dados Iris')
plt.show()

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Verificar as dimensões dos conjuntos de treinamento e teste
print("Dimensões do conjunto de treinamento:", X_train.shape, y_train.shape)
print("Dimensões do conjunto de teste:", X_test.shape, y_test.shape)

# Treinar o classificador K-NN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# Classificar as amostras do conjunto de teste
y_pred = knn.predict(X_test)

# Calcular a acurácia do modelo
accuracy = accuracy_score(y_test, y_pred)
print("Acurácia do modelo K-NN:", accuracy)
