import subprocess
import argparse




log_file = './results/esv_king_james_{0}_theta_{1}_method_{2}_v2.txt'


#  0 : baseline, 1 : pruning_column_row_sum, 2 : zick_zak, 3 : pruning_max_matrix_value, 4 : candidates

def start_experiments(gran, theta, method, stop_words=False):
    if stop_words:
        data_file = '../data/en/matches_stopwords.en.min.tsv'
    else:
        data_file = '../data/en/matches.en.min.tsv'
    
    if gran == 'book':
        text1_location = '/root/data/en/king_james_bible_book_martin/'
        text2_location = '/root/data/en/esv_book_martin/'
    
    
    if gran == 'chapter':
        text1_location = '/root/data/en/king_james_bible_chapter/1/'
        text2_location = '/root/data/en/esv_chapter/1/'
    
    if gran == 'para':
        text1_location = '/root/data/en/king_james_bible_para/1_1/'
        text2_location = '/root/data/en/esv_para/1_1'
    
    local_logfile = log_file.format(gran, theta, method)

    f = open(local_logfile, 'w')

    completed = subprocess.run(['./build/baseline', text1_location, text2_location, str(theta), data_file, str(method)], stdout=f)
    print('returncode: ', completed.returncode)
    f.close()






if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--gran', type=str, default='book')
    parser.add_argument('--theta', type=float, default=0.7)
    parser.add_argument('--method', type=int, default=4)

    params = parser.parse_args()
    start_experiments(params.gran, params.theta, params.method, stop_words=False)