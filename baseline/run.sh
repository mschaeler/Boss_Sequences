#!/bin/sh
make

./build/baseline ../data/en/king_james_bible/ ../data/en/esv/ 3 0.7 ../results/ ~/vectors_v2.sqlite3 ./en.tsv > ./results/esv_king_james_k3_0.7.txt