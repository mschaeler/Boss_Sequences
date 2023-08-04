CC = clang++
BUILD = ./build
SRC = ./src_cpp
CFLAGS = -O3 -fPIC -std=c++20 -Wno-deprecated 

baseline: $(BUILD)/baseline-main.o 
	$(CC) -o $(BUILD)/baseline $(BUILD)/baseline-main.o $(CFLAGS)

$(BUILD)/baseline-main.o : $(SRC)/main.cpp $(SRC)/Experiment.h $(SRC)/Environment.h $(SRC)/HungarianKevinStern.h $(SRC)/PermutationSolver.h
	$(CC) -c $(SRC)/main.cpp -o $(BUILD)/baseline-main.o $(CFLAGS)

