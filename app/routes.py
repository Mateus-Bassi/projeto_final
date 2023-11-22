from flask import flash, redirect, render_template, request, url_for
from sklearn.model_selection import train_test_split

from app import app
from .forms import ClassifierForm, SVMForm, DTForm, RFForm, MLPForm
from ml import carregar_dados, treinar_e_avaliar



"""
Recebe um classificador e seleciona o formulário de acordo com ele
Possui um dicionário de parâmetros que é modificado de acordo com o classificador
    Esse dicionário possi os argumentos necessários para o formulário de cada classificador

Chama a função de treinamento depois do formulário dinâmico ser preenchido
Renderiza a template classifier_form.html
"""
@app.route('/select_classifier', methods=['GET', 'POST'])
def select_classifier():
    classifier = request.args.get('classifier', 'DT')  # DT como padrão
    form = None
    parametros = {}

    if classifier == 'SVM':
        form = SVMForm(request.form)
        if form.validate_on_submit():
            parametros = {'kernel': form.kernel.data, 'degree': form.degree.data}
            #ADD leaf_size : int, default=30
            """
            leaf_sizeint, padrão = 30
            Tamanho da folha passado para BallTree ou KDTree. Isso pode afetar a velocidade de construção e consulta, bem como a memória necessária para armazenar a árvore. O valor ideal depende da natureza do problema.

            pfloat, padrão = 2
            Parâmetro de potência para a métrica Minkowski. Quando p = 1, isso é equivalente a usar distância_manhattan (l1) e distância_euclidiana (l2) para p = 2. Para p arbitrário, distância_minkowski (l_p) é usado.
            """

    elif classifier == 'DT':
        form = DTForm(request.form)
        if form.validate_on_submit():
            parametros = {'max_depth': form.max_depth.data}
            #ADD min_samples_leafint or float, default=1
            #ADD max_leaf_nodesint, default=None

            """
            The minimum number of samples required to split an internal node:
            If int, then consider min_samples_split as the minimum number.
            If float, then min_samples_split is a fraction and ceil(min_samples_split * n_samples) are the minimum number of samples for each split.

            Grow a tree with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
            """
    elif classifier == 'MLP':
        form = MLPForm(request.form)
        if form.validate_on_submit():
            hidden_layer_sizes = tuple(map(int, form.hidden_layer_sizes.data.split(',')))
            parametros = {'hidden_layer_sizes': hidden_layer_sizes, 'max_iter': form.max_iter.data}


    elif classifier == 'RF':
        form = RFForm(request.form)
        if form.validate_on_submit():
            parametros = {'n_estimators': form.n_estimators.data, 'max_depth': form.max_depth.data}
            #max_leaf_nodesint, default=None
            """
            Grow trees with max_leaf_nodes in best-first fashion. Best nodes are defined as relative reduction in impurity. If None then unlimited number of leaf nodes.
            """

    # Treinamento com os dados do formulário
    if request.method == 'POST' and form.validate():
        X, y = carregar_dados()
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
        caminho_imagem = treinar_e_avaliar(classifier, parametros, X_train, y_train, X_test, y_test)

        flash('Modelo treinado com sucesso!')
        return render_template('classifier_form.html', form=form, classifier=classifier, matrix_image=caminho_imagem)#, resultado=acuracia)

    return render_template('classifier_form.html', form=form, classifier=classifier)


@app.route('/', methods=['GET', 'POST'])
def index():
    form = ClassifierForm()
    if form.validate_on_submit():
        classificador_escolhido = form.classifier.data
        return redirect(url_for('select_classifier', classifier=classificador_escolhido))

    return render_template('index.html', form=form)

if __name__ == '__main__':
    app.run(debug=True)
