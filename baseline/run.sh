#!/bin/sh
make

for i in 1 2 3 4 5 6 7 8 9 10
do
    for k in 3 
    do 
        ./build/baseline /root/data/en/king_james_bible/${i}/ /root/data/en/esv_chapter/${i}/ ${k} 0.7 ../results/ ./en.tsv > ./results/esv_king_james_${i}_k${k}_0.7_log_chapterwise.txt
    done 
done