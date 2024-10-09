from io import open
import glob
import os
import unicodedata
import string



def conversion(path):

    def findFiles(path): return glob.glob(path)

    all_letters = string.ascii_letters + " .,;'"
    n_letters = len(all_letters)

    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
        )

    category_lines = {}
    all_categories = []

    def readLines(filename):
        lines = open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicodeToAscii(line) for line in lines]


    for filename in findFiles(path):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        lines = readLines(filename)
        category_lines[category] = lines

    return  category_lines, all_categories