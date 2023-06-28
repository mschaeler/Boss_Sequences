#!/bin/sh
make

for i in 1 2 3 4 5 6 7 8 9 10
do
    ./build/baseline /root/data/en/king_james_bible/${i}/ /root/data/en/esv/${i}/ 7 0.7 ../results/ ./en.tsv > ./results/esv_king_james_${i}_k7_0.7.txt
done