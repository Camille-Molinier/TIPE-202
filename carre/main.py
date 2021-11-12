from PIL import Image
import matplotlib.pyplot as plt
from src.implements.carreV1 import CarreV1

carre = CarreV1

im000 = Image.open("src/images/000.jpg")
imc000 = Image.open("src/images/000_cropped.jpg")
im022 = Image.open("src/images/022.jpg")
imc022 = Image.open("src/images/022_cropped.jpg")
im108 = Image.open("src/images/108.jpg")
imc108 = Image.open("src/images/108_cropped.jpg")

carre.trace_carre(carre, imc108, im108)
carre.trace_carre(carre, imc000, im000)
carre.trace_carre(carre, imc022, im022)

plt.figure()
plt.imshow(im000)

plt.figure()
plt.imshow(im022)

plt.figure()
plt.imshow(im108)

plt.show()
