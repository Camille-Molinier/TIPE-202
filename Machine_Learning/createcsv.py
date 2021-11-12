import glob
import csv
import pandas as pd

filelisthornet = glob.glob("../image_dataset/hornet_images_and_labels/labels/*.txt")
filelistwasp = glob.glob("../image_dataset/wasp_images_and_labels/labels/*.txt")


def splitvalues(content):
    stringValues = content.split(" ")
    return stringValues[0], float(stringValues[1]), float(stringValues[2]), float(stringValues[3])


with open('dataset.csv', 'w', newline='') as csvfile:
    feildnames = ["Species", "Black_proportion", "Orange_proportion", "Ratio_orange/black"]
    writer = csv.DictWriter(csvfile, fieldnames=feildnames)
    writer.writeheader()
    for path in filelisthornet:
        content = open(path, 'r').read()

        specie, black, orange, ratio = splitvalues(content)
        writer.writerow(
            {"Species": specie, "Black_proportion": black, "Orange_proportion": orange, "Ratio_orange/black": ratio})

    for path in filelistwasp:
        content = open(path, 'r').read()

        specie, black, orange, ratio = splitvalues(content)
        writer.writerow(
            {"Species": specie, "Black_proportion": black, "Orange_proportion": orange, "Ratio_orange/black": ratio})

df = pd.read_csv('dataset.csv')
df = df.sample(frac=1)
df.to_csv('dataset.csv', index=False)
