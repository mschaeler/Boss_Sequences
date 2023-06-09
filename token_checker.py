import os 
import numpy as np
from numpy.linalg import norm
import goslate

EN_DATA_FILE = "./data/en/matches.en.min.tsv"
DE_DATA_FILE = "./data/de/matches.de.min.tsv"


def load_tsv_file(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    file.close()
    """
        data is a dictionary
        syntactic token --> [embedding token, syntactic token score, 300 dimenstinoal [vector]]
    """
    data = {}
    del lines[0]
    for line in lines:
        l = line.split('\t')
        l = [l1.strip() for l1 in l]
        # l = [embedding token, syntactic token, score, vector]
        token = l[0]
        syn_token = l[1]
        syn_score = float(l[2])
        vector = np.array([float(v) for v in l[3:]])
        data[syn_token] = [token, syn_score, vector / norm(vector)]
    
    # print(data["ahasuerus"])
    return data
        


def compare_vectors(en_token, de_token, en_data, de_data):
    en_token_vector = en_data[en_token][2]
    de_token_vector = de_data[de_token][2]
    cosine = np.dot(en_token_vector,de_token_vector)/(norm(en_token_vector)*norm(de_token_vector))
    print("Cosine Similarity:", cosine)



def translate():
    gs = goslate.Goslate()
    new_word = gs.translate('provinzen', 'en')
    print(new_word)

if __name__ == "__main__":
    # en_data = load_tsv_file(EN_DATA_FILE)
    # de_data = load_tsv_file(DE_DATA_FILE)
    # compare_vectors("ethiopia", "kusch", en_data, de_data)
    translate()