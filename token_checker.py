import os 
import numpy as np
from numpy.linalg import norm
import sqlite3
from sqlite3 import Error
import pickle
import faiss
from sklearn.metrics.pairwise import cosine_similarity as cosine

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
        data[syn_token] = [token, syn_score, vector]
    
    # print(data["ahasuerus"])
    return data
        


def compare_vectors(en_token, de_token, en_data, de_data):
    # en_token_vector = en_data[en_token][2]
    en_token_vector = de_data[en_token][2]
    de_token_vector = de_data[de_token][2]
    cosine = np.dot(en_token_vector,de_token_vector)/(norm(en_token_vector)*norm(de_token_vector))
    print("Cosine Similarity:", cosine)


def data_to_db(data):
    endb = "/root/vectors_v2.sqlite3"
    conn = None
    try:
        conn = sqlite3.connect(endb)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS en_vectors(word varchar(100), vec blob, idx int)
        ''')
        i = 0
        for k, v in data.items():
            word = k
            vector = v[2]
            vector_blob = pickle.dumps(vector)
            c.execute("INSERT INTO en_vectors(word, vec, idx) VALUES(?,?,?)", (word, vector_blob, i))
            conn.commit()
            i += 1

        conn.commit()
    except Error as e:
        print(e)
    
    finally:
        if conn:
            conn.close()


def all_pair_similarity(data):
    id_to_word = []
    A = np.zeros((len(data), 300))
    B = np.zeros((len(data), 300))
    i = 0
    for k, v in data.items():
        word = k
        vector = v[2]
        id_to_word[i] = word
        A[i] = vector
        B[i] = vector
        i += 1
    
    results = []
    # for i in range(A.shape[0]):
    #     # results.append(np.max)



def faiss_test(data):
    nb = len(data)
    d = 300
    id_to_word = [None] * nb
    A = np.zeros((len(data), 300))
    B = np.zeros((len(data), 300))
    i = 0
    for k, v in data.items():
        word = k
        vector = v[2]
        id_to_word[i] = word
        A[i] = vector
        B[i] = vector
        i += 1
    
    A = A.astype('float32')
    A[:, 0] += np.arange(nb) / 1000.

    B = B.astype('float32')
    B[:, 0] += np.arange(nb) / 1000.

    index = faiss.IndexFlatL2(d)

    print(index.is_trained)
    index.add(A)
    print(index.ntotal)

    D, I = index.search(B, nb)     # actual search
    for idx, i in enumerate(I):
        print("query word : {0}".format(id_to_word[idx]))
        for n in i:
            print('similar words:')
            print(id_to_word[n])
        break









if __name__ == "__main__":
    en_data = load_tsv_file(EN_DATA_FILE)
    # de_data = load_tsv_file(DE_DATA_FILE)
    # compare_vectors("zeiten", "tagen", en_data, de_data)
    # translate()
    # data_to_db(en_data)
    faiss_test(en_data)