import os 
import subprocess




CHAPTER_PARAS = {
    1 : 22,
    2 : 23,
    3 : 15,
    4 : 17,
    5 : 14,
    6 : 14,
    7 : 10,
    8 : 17,
    9 : 32,
    10 : 3
}


logfolder_para2para = './results/esv_king_james_{0}_{1}_k{2}_0.7_log_para_permIndex2.txt'
logfolder_para2chapter = './results/esv_king_james_{0}_{1}_{2}_k{3}_0.7_log_para2chapter_permIndex2.txt'
logfolder_chapter2chapter = './results/esv_king_james_{0}_k{1}_0.7_log_chapter2chapter_permIndex2.txt'
results_para2para = './para2para_k{0}_permIndex2.csv'
results_para2chapter = './para2chapter_k{0}_permIndex2.csv'
results_chapter2chapter = './chapter2chapter_k{0}_permIndex2.csv'


def get_data(fileloc):
    file = open(fileloc, 'r')
    lines = file.readlines()
    lines = [line.strip() for line in lines]
    file.close()
    result_data = {}
    for line in lines:
        if line.startswith('Inverted Index Size:'):
            s = line.split(':')
            result_data['invertedIndex'] = int(s[1].strip())
        
        if line.startswith('Size of valid edges:'):
            s = line.split(':')
            result_data['validEdges'] = int(s[1].strip())
        
        if line.startswith('Number of Graph Matching Computed:'):
            s = line.split(':')
            result_data['alignmentMatrixSize'] = int(s[1].strip())

        if line.startswith('Number of Zero Entries Cells:'):
            s = line.split(':')
            result_data['belowThresholdCells'] = int(s[1].strip())

        if line.startswith('Env Time:'):
            s = line.split(':')
            result_data['envTime'] = float(s[1].strip())

        if line.startswith('Dataloader Time:'):
            s = line.split(':')
            result_data['dataloaderTime'] = float(s[1].strip())

        if line.startswith('FaissIndex Time:'):
            s = line.split(':')
            result_data['faissTime'] = float(s[1].strip())

        if line.startswith('Main Loop Time:'):
            s = line.split(':')
            result_data['algoTime'] = float(s[1].strip())

    return result_data


def get_paths(granularity):
    if granularity == 'para2para':
        return [logfolder_para2para, results_para2para]
    elif granularity == 'para2chapter':
        return [logfolder_para2chapter, results_para2chapter]
    elif granularity =='chapter2chapter':
        return [logfolder_chapter2chapter, results_chapter2chapter]
    else:
        return None

def baseline_runner(granularity, theta=0.7, k=3):
    paths = get_paths(granularity)
    outfolder = '../results/'
    embedding_file = "./en.tsv"
    if granularity == 'para2para':
        for i in range(1, 11):
            chapter = i
            for p in range(CHAPTER_PARAS[chapter]):
                text1 = '/root/data/en/king_james_bible_para/{0}_{1}/'.format(chapter, p)
                text2 = '/root/data/en/esv_para/{0}_{1}/'.format(chapter, p)
                local_logfolder = paths[0].format(chapter, p, k)
                f = open(local_logfolder, 'w')
                completed = subprocess.run(['./build/baseline', text1, text2, str(k), str(theta), outfolder, embedding_file], stdout=f)
                print('returncode: ', completed.returncode)
                f.close()
    elif granularity == 'para2chapter':
        for i in range(1, 11):
            chapter = i
            for p in range(CHAPTER_PARAS[chapter]):
                text1 = '/root/data/en/king_james_bible_para/{0}_{1}/'.format(chapter, p)
                text2 = '/root/data/en/esv_chapter/{0}/'.format(chapter)
                local_logfolder = paths[0].format(chapter, p, chapter, k)
                f = open(local_logfolder, 'w')
                completed = subprocess.run(['./build/baseline', text1, text2, str(k), str(theta), outfolder, embedding_file], stdout=f)
                print('returncode: ', completed.returncode)
                f.close()
    elif granularity == 'chapter2chapter':
        for i in range(1, 11):
            chapter = i
            text1 = '/root/data/en/king_james_bible_chapter/{0}/'.format(chapter) 
            text2 = '/root/data/en/esv_chapter/{0}/'.format(chapter)
            local_logfolder = paths[0].format(chapter, k)
            f = open(local_logfolder, 'w')
            completed = subprocess.run(['./build/baseline', text1, text2, str(k), str(theta), outfolder, embedding_file], stdout=f)
            print('returncode: ', completed.returncode)
            f.close()
    else:
        print('Unknown Granularity')


def baseline_results_to_csv(granularity, k):
    paths = get_paths(granularity)
    if granularity in ['para2para', 'para2chapter'] :
        outfile = paths[1].format(k)
        with open(outfile, 'w') as f:
            f.write('Paragraph, AlgoTime(ms), Result Matrix, Chapter, Zero Entries\n')
            idx = 0
            for i in range(1, 11):
                for p in range(CHAPTER_PARAS[i]):
                    if granularity == 'para2chapter':
                        data = get_data(paths[0].format(i, p, i, k))
                    else:
                        data = get_data(paths[0].format(i, p, k))
                    algoTime_ms = data['algoTime'] * 1000
                    matrixSize = data['alignmentMatrixSize']
                    zeroEntries = data['belowThresholdCells']
                    f.write('{0}, {1}, {2}, {3}, {4}\n'.format(idx, algoTime_ms, matrixSize, i, zeroEntries))
                    idx += 1
        f.close()
    
    elif granularity == 'chapter2chapter':
        outfile = paths[1].format(k)
        with open(outfile, 'w') as f:
            f.write('AlgoTime(ms), Result Matrix, Chapter, Zero Entries\n')
            for i in range(1, 11):
                data = get_data(paths[0].format(i, k))
                try:
                    algoTime_ms = data['algoTime'] * 1000
                    matrixSize = data['alignmentMatrixSize']
                    zeroEntries = data['belowThresholdCells']
                    f.write('{0}, {1}, {2}, {3}\n'.format(algoTime_ms, matrixSize, i, zeroEntries))
                except KeyError:
                    print(paths[0].format(i, k))
        f.close()

if __name__ == '__main__':
    for gran in ['para2chapter', 'para2para', 'chapter2chapter']:
        for k in [3]:
            print('Experiment for gran:{0}\tk:{1}'.format(gran, k))
            baseline_runner(gran, 0.7, k)
            baseline_results_to_csv(gran, k)