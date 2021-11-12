import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, recall_score, confusion_matrix, classification_report, plot_confusion_matrix, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import f_classif, SelectKBest
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


##########################################################################
#                            Dataset Loading                             #
##########################################################################
data = pd.read_csv('dataset.csv')
# print('Initialisation :')
# print(data.shape)
# print(data.head())
# print()

##########################################################################
#                       Data Exploratory Analysis                        #
##########################################################################
df = data.copy()
# print('Data Exploratory Analysis :')
#
# 3D graph
species = df['Species']
transformer = LabelEncoder()
colors = transformer.fit_transform(species)

hornet = df['Species'] == 'frelon'
wasp = df['Species'] == 'guepe'

black = df['Black_proportion']
orange = df['Orange_proportion']
ratio = df['Ratio_orange/black']

black_hornet = hornet['Black_proportion']
orange_hornet = hornet['Orange_proportion']
ratio_hornet = hornet['Ratio_orange/black']

plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(black, orange, ratio, c=colors, label=species)
plt.title('Représentation 3D du dataset')

plt.show()
#  Hist
# plt.figure()
# plt.hist(df['Orange_proportion'])

# Pair plot
# sns.pairplot(df, hue="Species", diag_kws={'bw': 0.2})

# # Heatmap
# sns.heatmap(df.corr())
# plt.title('Matrice de correlation')
#
# # Joint plot
# plt.figure()
# sns.jointplot('Black_proportion', 'Orange_proportion', df, kind='kde', cmap="YlOrBr", color='y')
#
# # Box plot
# plt.figure()
# sns.boxplot(x="Species", y="Black_proportion", data=df)
# plt.title('Répartition des individus en fonction de leur espèce et du taux de couleur noire')
# plt.figure()
# sns.boxplot(x="Species", y="Orange_proportion", data=df)
# plt.title('Répartition des individus en fonction de leur espèce et du taux de couleur orange')
# plt.figure()
# sns.boxplot(x="Species", y="Ratio_orange/black", data=df)
# plt.title('Répartition des individus en fonction de leur espèce et du ratio des couleurs')
#
# # Violin plot
# plt.figure()
# sns.violinplot(x="Species", y="Black_proportion", data=df)
# plt.figure()
# sns.violinplot(x="Species", y="Orange_proportion", data=df)
# plt.figure()
# sns.violinplot(x="Species", y="Ratio_orange/black", data=df)

print()
plt.show()

##########################################################################
#                             Preprocessing                              #
##########################################################################
df = data.copy()
print('Preprocessing :')


def encoding(_df):
    code = {'frelon': 1, 'guepe': 0}
    for col in _df.select_dtypes('object').columns:
        _df.loc[:, col] = _df[col].map(code)

    return _df


def impute(_df):
    return _df.dropna(axis=0)


def preprocessing(_df):
    _df = encoding(_df)
    _df = impute(_df)

    X = _df.drop('Species', axis=1)
    y = _df['Species']

    return X, y


def evaluation(_model):
    _model.fit(X_train, y_train)
    y_pred = _model.predict(X_test)

    # print('Evaluation report : ')
    # print(confusion_matrix(y_test, y_pred))
    # print(classification_report(y_test, y_pred))

    N, train_score, val_score = learning_curve(_model, X_train, y_train, cv=4, train_sizes=np.linspace(0.1, 1, 10))

    # plt.figure()
    # plt.plot(N, train_score.mean(axis=1), label='train score')
    # plt.plot(N, val_score.mean(axis=1), label='val score')
    # plt.legend()
    # plot_confusion_matrix(_model, X_test, y_test)
    return N, train_score, val_score


# Ensembles
color_columns = ['Black_proportion', 'Orange_proportion']
ratio_columns = ['Ratio_orange/black']
morph_columns = ['Length']
target = ['Species']

# Filtred Dataset
df = df[target + color_columns + ratio_columns]
print(df.head())

# Train/Test split
train_set, test_set = train_test_split(df, test_size=0.2, random_state=0)
print(train_set['Species'].value_counts())
print(test_set['Species'].value_counts())

X_train, y_train = preprocessing(train_set)
X_test, y_test = preprocessing(test_set)

# first test model
# model = DecisionTreeClassifier(random_state=0)
#
# evaluation(model)

# features_names = ['Black proportion', 'Orange proportion', 'Ratio Orange/Black', 'Length']
# class_names = ['Wasp', 'Hornet']

# fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(4, 4), dpi=300)
# plot_tree(model, feature_names=features_names, class_names=class_names, filled=True)

##########################################################################
#                              Modelisation                              #
##########################################################################
preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=False))
#
RandomForest = make_pipeline(preprocessor, RandomForestClassifier(random_state=0))
AdaBoost = make_pipeline(preprocessor, AdaBoostClassifier(random_state=0))
SVM = make_pipeline(preprocessor, StandardScaler(), SVC(random_state=0))
KNN = make_pipeline(preprocessor, StandardScaler(), KNeighborsClassifier())
#
# model_list = [RandomForest, AdaBoost, SVM, KNN]

# for models in model_list:
#     evaluation(models)

#################### Amelioration model Random Forest ####################
hyper_params_RF = {'randomforestclassifier__n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9],
                   'randomforestclassifier__criterion': ['gini', 'entropy']}

grid_RF = GridSearchCV(RandomForest, hyper_params_RF, scoring='recall', cv=4)
grid_RF.fit(X_train, y_train)

print(grid_RF.best_params_)

evaluation(grid_RF.best_estimator_)

###################### Amelioration model AdaBoost #######################
hyper_params_Ada = {'adaboostclassifier__algorithm': ['SAMME', 'SAMME.R'],
                    'adaboostclassifier__n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                    'adaboostclassifier__base_estimator': [DecisionTreeClassifier(random_state=0), KNeighborsClassifier()]}

grid_Ada = GridSearchCV(AdaBoost, hyper_params_Ada, scoring='recall', cv=4)
grid_Ada.fit(X_train, y_train)

print(grid_Ada.best_params_)

evaluation(grid_Ada.best_estimator_)

######################### Amelioration model SVM #########################
hyper_params_SVM = {'svc__gamma': [0.001, 0.0001],
                    'svc__C': [1, 10, 100, 1000],
                    'svc__kernel': ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed'],
                    'svc__degree': [1, 2, 3, 4, 5]}

grid_SVM = GridSearchCV(SVM, hyper_params_SVM, scoring='recall', cv=4)
grid_SVM.fit(X_train, y_train)

print(grid_SVM.best_params_)

evaluation(grid_SVM.best_estimator_)

######################### Amelioration model KNN #########################
# hyper_params_KNN = {'kneighborsclassifier__n_neighbors': [2, 3, 4, 5, 6, 7, 8, 9, 10],
#                     'kneighborsclassifier__leaf_size': [1, 2, 3, 4, 5, 6, 7, 8, 9],
#                     'kneighborsclassifier__weights': ['uniform', 'distance'],
#                     'kneighborsclassifier__algorithm': ['ball_tree', 'kd_tree', 'brute']}
#
# grid_KNN = GridSearchCV(KNN, hyper_params_KNN, scoring='recall', cv=4)
# grid_KNN.fit(X_train, y_train)
#
# print(grid_KNN.best_params_)
#
# evaluation(grid_KNN.best_estimator_)

############################### Comparaison ##############################
# NRF, train_score_RF, val_score_RF = evaluation(grid_RF.best_estimator_)
# NAda, train_score_Ada, val_score_Ada = evaluation(grid_Ada.best_estimator_)
# NKNN, train_score_KNN, val_score_KNN = evaluation(grid_KNN.best_estimator_)
# NSVM, train_score_SVM, val_score_SVM = evaluation(grid_SVM.best_estimator_)
#
# # Learning curve
# plt.figure()
# plt.subplot(221)
# plt.plot(NRF, train_score_RF.mean(axis=1), label='train score'), plt.plot(NRF, val_score_RF.mean(axis=1), label='val score')
# plt.legend(), plt.title("Random Forest Classifier")
# plt.subplot(222)
# plt.plot(NAda, train_score_Ada.mean(axis=1), label='train score'), plt.plot(NAda, val_score_Ada.mean(axis=1), label='val score')
# plt.legend(), plt.title("AdaBoost Classifier")
# plt.subplot(223)
# plt.plot(NKNN, train_score_KNN.mean(axis=1), label='train score'), plt.plot(NKNN, val_score_KNN.mean(axis=1), label='val score')
# plt.legend(), plt.title("K Nearest Neighbour Classifier")
# plt.subplot(224)
# plt.plot(NSVM, train_score_SVM.mean(axis=1), label='train score'), plt.plot(NSVM, val_score_SVM.mean(axis=1), label='val score')
# plt.legend(), plt.title("Support Vector Classifier")
#
# # Confusion matrix
# plot_confusion_matrix(grid_RF.best_estimator_, X_test, y_test)
# plt.title("Random Forest Classifier")
#
# plot_confusion_matrix(grid_Ada.best_estimator_, X_test, y_test)
# plt.title("AdaBoost Classifier")
#
# plot_confusion_matrix(grid_KNN.best_estimator_, X_test, y_test)
# plt.title("K Nearest Neighbour Classifier")
#
# plot_confusion_matrix(grid_SVM.best_estimator_, X_test, y_test)
# plt.title("Support Vector Classifier")
##########################################################################
#                                Final model                             #
##########################################################################
finalModel = make_pipeline(preprocessor, AdaBoostClassifier(algorithm='SAMME', n_estimators=3))
finalModel.fit(X_train, y_train)
precision, recall, thresholds = precision_recall_curve(y_test, finalModel.decision_function(X_test))

plt.figure()
plt.plot(thresholds, precision[:-1], label='precision')
plt.plot(thresholds, recall[:-1], label='recall')
plt.ylabel("score")
plt.xlabel("threshold")
plt.legend()

# # print(X_train.shape)
# X = np.array([81.62866124, 0.0, 0.0])
# X = X.reshape(1, 3)
# print(finalModel.predict(X))
##########################################################################

plt.show()
