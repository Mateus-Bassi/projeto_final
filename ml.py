import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


"""
Função para carregar dados do Titanic em formato CSV.
Seleciona colunas relevantes para o Machine Learning.

Retorna o conjunto de características (X) e o alvo (y).
"""
def carregar_dados():
    #url = 'https://learnenough.s3.amazonaws.com/titanic.csv'
    #titanic = pd.read_csv(url)
    titanic = pd.read_csv('titanic.csv')

    columns_to_drop = ['Name', 'PassengerId', 'Cabin', 'Embarked', 
                   'SibSp', 'Parch', 'Ticket', 'Fare']

    for column in columns_to_drop:
        titanic = titanic.drop(column, axis=1)

    for column in ['Age', 'Sex', 'Pclass']:
        titanic = titanic[titanic[column].notna()]

    sex_int = {'male': 0, 'female': 1}
    titanic['Sex'] = titanic['Sex'].map(sex_int)

    titanic.head()

    X = titanic.drop('Survived', axis=1)
    y = titanic['Survived']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    return X, y

"""
Realiza o treinamento de acordo com o classificador recebido no parâmetro
Realiza plotagem de matrix de confusão e transforma imagem
Salva imagem na pasta static com o nome de acordo com o classificador

Retorna o nome do arquivo de imagem salvo
"""
def treinar_e_avaliar(classificador, parametros, X_train, y_train, X_test, y_test):
    
    if classificador == 'SVM':
        clf = SVC(kernel=parametros['kernel'], degree=parametros['degree'])
        #clf = SVC(kernel=parametros['kernel'], degree=parametros['degree'], leaf_size=parametros['leaf_size'])
    elif classificador == 'DT':
        clf = DecisionTreeClassifier(max_depth=parametros['max_depth'])
        #clf = DecisionTreeClassifier(max_depth=parametros['max_depth'])
    elif classificador == 'MLP':
        clf = MLPClassifier(hidden_layer_sizes=parametros['hidden_layer_sizes'], max_iter=parametros['max_iter'])
        #clf = MLPClassifier(hidden_layer_sizes=parametros['hidden_layer_sizes'], max_iter=parametros['max_iter'])
    elif classificador == 'RF':
        clf = RandomForestClassifier(n_estimators=parametros['n_estimators'], max_depth=parametros['max_depth'])
        #clf = RandomForestClassifier(n_estimators=parametros['n_estimators'], max_depth=parametros['max_depth'])
    else:
        raise ValueError("Classificador não encontrado.")
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    matriz_confusao = confusion_matrix(y_test, y_pred)

    acuracia = accuracy_score(y_test, y_pred)
    print(f'Acurácia do {classificador}: {acuracia}')
    
    # Salvando a matriz de confusão como imagem
    nome_imagem = f'confusion_matrix_{classificador}.png'
    caminho_imagem = f'static/{nome_imagem}'
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz_confusao)
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax)
    plt.savefig(caminho_imagem)
    plt.close(fig)

    return nome_imagem
