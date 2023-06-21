import os 
from os.path import isfile
import re





def load_data(fileloc):
    file = open(fileloc, 'r')
    lines = file.readlines()
    file.close()
    data = {}
    del lines[0]
    verse = -1
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
            print(sequence)
            sequence_id = int(sequence[0])
            formatted_seq = []
            for s in sequence[1:]:
                res = re.sub(r'[^\w\s]', '', s)
                formatted_seq.append(res)
            # print(verse)
            if verse > 0:
                data[verse].insert(sequence_id - 1,formatted_seq)
    return data


def write_data(data, outfolder):
    os.makedirs(outfolder, exist_ok=True)
    file_name = '{0}_{1}.txt'
    for k, v in data.items():
        for idx, v1 in enumerate(v):
            f = file_name.format(k, idx+1)
            with open(os.path.join(outfolder, f), 'w') as out:
                for word in v1:
                    out.write('{0}\n'.format(word))
            out.close()





if __name__ == "__main__":
    fileloc = "./data/en/esv.txt"
    data = load_data(fileloc)
    outfolder ="./data/en/esv/"
    write_data(data, outfolder)