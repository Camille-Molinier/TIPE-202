print("Importation des modules en cours...")
print("    modules python...", end="")
import numpy as np
import pandas as pd
print("ok")
print("    Scikit Learn...", end="")
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
print("ok")
print("    Image processor...", end="")
from Image_Processing.Image_Processor import Processor
print("ok")
print("    Targeter...", end="")
from targeter.carreV1 import CarreV1
print("ok")
print("Importation des modules terminée !", "\n")
################################################################################
#                               Machine Learning                               #
################################################################################
############################ Usefull functions ############################


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

def splitvalues(content):
    stringValues = content.split(" ")
    return float(stringValues[0]), float(stringValues[1]), float(stringValues[2])


############################## Init function ##############################
def model_init():
    # Loading data
    df = pd.read_csv('Machine_Learning/dataset.csv')

    color_columns = ['Black_proportion', 'Orange_proportion']
    ratio_columns = ['Ratio_orange/black']
    target = ['Species']
    # Filtred Dataset
    df = df[target + color_columns + ratio_columns]

    X, y = preprocessing(df)
    # # Making the train set and the test set
    # train_set, test_set = train_test_split(df, test_size=0.2, random_state=0)
    # X_train, y_train = preprocessing(train_set)
    # X_test, y_test = preprocessing(test_set)

    preprocessor = make_pipeline(PolynomialFeatures(2, include_bias=False))
    model_ = make_pipeline(preprocessor, RandomForestClassifier(criterion='gini', n_estimators=3))
    model_.fit(X, y)
    # plot_confusion_matrix(model_, X, y)
    # plt.show()
    return model_


########################### Prediction function ###########################
def machine_learning(name, _model):
    prc = Processor()
    prc.processing_demo(name)
    content = open("imagesDEMO/Bin/label.txt", 'r').read()
    b, o, r = splitvalues(content)
    X = np.array([b, o, r])
    X = X.reshape(1, 3)
    Pred = _model.predict(X)
    if Pred[0] == 1:
        return True
    else:
        return False


################################################################################
#                                    Website                                   #
################################################################################
def update_website(image_number):
    preImage = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta http-equiv='refresh' content='5' />
        <meta charset=\"UTF-8\">
        <title>A-Hive : Solution de detection de frelons</title>
        <link rel="icon" type="Images/png" href="images/bee.png" />
        <link rel="stylesheet" type="text/css" href="Style.css" />
    </head>
    <body>
        <img src="images/logo.png" class = "center">
    <header>
        <section>
            <h1>A-Hive, votre solution de detection des frelons asiatiques</h1>
            <div class = "rec-title">
                <a class="part-title">Navigation |</a>
                <a class = "part-title" href="Carte.html">Carte de la presence des frelons en France  |</a>
                <a class = "part-title" href="Presentation.html">Presentation de notre demarche  |</a>
            </div>
            <div id="rectangle">
                <p>Image courante prise par le dispositif :</p>
            </div>
            <img src=
    """
    postImage = """ class = "center"/>
            <div id="rectangleTitre">
            <p>"""

    postString = """</p>
            </div>
        </section>
        </header>
    </body>
    </html>"""

    # if machine_learning(f"imgTest{image_number}.jpg", model):
    #     targeter.trace_carre(image_number)
    #     adresseImage = f"""imagesDEMO/imgTest{image_number}.jpg"""
    #     indexString = preImage + adresseImage + postImage + "Reponse de l'algorithme : Attention ! Il y a un frelon devant votre ruche." + postString
    # else:
    #     adresseImage = f"""imagesDEMO/imgTest{image_number}.jpg"""
    #     indexString = preImage + adresseImage + postImage + "Reponse de l'algorithme : Aucune menace detectee." + postString

    hornet = machine_learning(f"imgTest{image_number}.jpg", model)
    targeter.trace_carre_V2(image_number, hornet)
    adresseImage = "../imagesDemo/Bin/carre.jpg"

    if hornet:
        rep = "Reponse de l'algorithme : Attention ! Il y a un frelon devant votre ruche."
    else:
        rep = "Reponse de l'algorithme : Aucune menace detectee."

    indexString = preImage + adresseImage + postImage + rep + postString

    with open("Website/Index.html", "w") as file:
        file.write(indexString)


################################################################################
#                                   Main loop                                  #
################################################################################
print("Initialisation du modèle en cours...")
model = model_init()
print("Initalisation du modèle avec sucsès", "\n")


image = ""
answers = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17"]
end = False
targeter = CarreV1()

while not end:
    image = input("Choisir un numéro d'image : ")
    if answers.__contains__(image):
        update_website(int(image))
        print("    Site modifié avec succses !")
    elif image == "Stop" or image == "stop":
        end = True
    else:
        print("Entrée invalide, réesayez.")
    print()
