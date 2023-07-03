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




def start_experiments():
    outfolder = '../results/'
    logfolder = './results/esv_king_james_{0}_{1}_k{2}_0.7_log_para.txt'

    for i in range(1, 11):
        chapter = i
        paras = [p for p in range(CHAPTER_PARAS[chapter])]
        for p in paras:
            text1 = '/root/data/en/king_james_bible_para/{0}_{1}/'.format(chapter, p)
            text2 = '/root/data/en/esv_para/{0}_{1}/'.format(chapter, p)
            local_logfolder = logfolder.format(chapter, p, 3)
            try:
                completed = subprocess.run(['./build/baseline', text1, text2, "3", "0.7", outfolder, "./en.tsv"], stdout=local_logfolder)
                print('returncode: ', completed.returncode)
            except:
                print("Error")


if __name__ == '__main__':
    start_experiments()