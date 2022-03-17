import os

path = "C:\\Users"


def listdir(dir):
    filenames = os.listdir(dir)
    for files in filenames:
        print(files)


listdir(path)