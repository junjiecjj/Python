from os import listdir
from os.path import isfile, join


def printnames(dir):
    # loop through every file and folder in the current folder
    for file in sorted(listdir(dir)):
        fullpath = join(dir, file)
        if isfile(fullpath):
            # if it is a file, print out the name
            print(file)
        else:
            # if it is a folder, call this function recursively on it
            # to look for files and folders
            printnames(fullpath)


printnames("pics")
