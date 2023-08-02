import os 
from os.path import isfile
import re
import numpy as np




def load_data(fileloc):
    file = open(fileloc, 'r')
    lines = file.readlines()
    file.close()
    data = {}
    del lines[0]
    verse = -1
    all_lines = []
    for line in lines:
        if line.startswith('--'):
            verse = int(line.replace('--', '').strip())
            data[verse] = list()
        else:
            line = line.replace(u'\xa0', u' ')
            sequence = line.split(' ')
            # print(line)
            # print(sequence)
            sequence = [s.strip() for s in sequence]
            # print(sequence)
            sequence_id = int(sequence[0])
            formatted_seq = []
            for s in sequence[1:]:
                res = re.sub(r'[^\w\s]', '', s)
                formatted_seq.append(res)
            # print(verse)
            if verse > 0:
                data[verse].insert(sequence_id - 1,formatted_seq)
                all_lines.extend(formatted_seq)
    # return data
    return all_lines



def write_book(data, outfolder):
    os.makedirs(outfolder, exist_ok=True)
    print(outfolder)
    file_nmae = '1.txt'
    with open(os.path.join(outfolder, file_nmae), 'w') as out:
        for word in data:
            out.write('{0}\n'.format(word.lower()))
    out.close()

def write_data(data, outfolder):
    os.makedirs(outfolder, exist_ok=True)
    print(outfolder)
    # file_name = '{0}_{1}.txt'
    # file_name = '{0}/{1}.txt'
    file_name = '{0}.txt'
    for k, v in data.items():
        for idx, v1 in enumerate(v):
            # outfolder_1 = outfolder + '{0}_{1}'.format(k, idx)
            outfolder_1 = outfolder + '{0}'.format(k)
            # outfolder_1 = outfolder
            os.makedirs(outfolder_1, exist_ok=True)
            # f = file_name.format(k, idx+1)
            f = file_name.format(k)
            with open(os.path.join(outfolder_1, f), 'a') as out:
                for word in v1:
                    out.write('{0}\n'.format(word.lower()))
            out.close()



def normalize_vector(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm

def cosine_sim(a, b):
    dot_product = np.dot(a, b)
    return dot_product

def vectort_test():
    a1 = np.random.uniform(low=0, high=1, size=300)
    a2 = np.random.uniform(low=0, high=1, size=300)
    a3 = np.random.uniform(low=0, high=1, size=300)

    b1 = np.random.uniform(low=0, high=1, size=300)
    b2 = np.random.uniform(low=0, high=1, size=300)
    b3 = np.random.uniform(low=0, high=1, size=300)
    
    a1 = normalize_vector(a1)
    a2 = normalize_vector(a2)
    a3 = normalize_vector(a3)

    b1 = normalize_vector(b1)
    b2 = normalize_vector(b2)
    b3 = normalize_vector(b3)

    A = np.concatenate((a1, a2, a3))
    B = np.concatenate((b1, b2, b3))

    cos_sim = cosine_sim(a1, b1) + cosine_sim(a2, b2) + cosine_sim(a3, b3)
    print(cos_sim / 3)
    print(cosine_sim(A, B) / 3)
    print(cosine_sim(normalize_vector(A), normalize_vector(B)))






if __name__ == "__main__":
    fileloc = "./data/en/esv.txt"
    data = load_data(fileloc)
    outfolder ="/root/data/en/esv_book/"
    write_book(data, outfolder)
    # write_data(data, outfolder)
    # vectort_test()