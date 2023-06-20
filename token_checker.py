import os 
import numpy as np
from numpy.linalg import norm
import sqlite3
from sqlite3 import Error

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
    endb = "/root/vectors.sqlite3"
    conn = None
    try:
        conn = sqlite3.connect(endb)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS de_vectors(word varchar(100), vec blob, idx int)
        ''')
        i = 0
        for k, v in data.items():
            word = k
            vector = v[2]
            c.execute("INSERT INTO de_vectors(word, vec, idx) VALUES(?,?,?)", (word, sqlite3.Binary(vector), i))
            conn.commit()
            i += 1

        conn.commit()
    except Error as e:
        print(e)
    
    finally:
        if conn:
            conn.close()



if __name__ == "__main__":
    # en_data = load_tsv_file(EN_DATA_FILE)
    de_data = load_tsv_file(DE_DATA_FILE)
    # compare_vectors("zeiten", "tagen", en_data, de_data)
    # translate()
    data_to_db(de_data)