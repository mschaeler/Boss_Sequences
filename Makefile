CC = clang++
BUILD = ./build
SRC = ./src_cpp
CFLAGS = -O3 -fPIC -std=c++20 -Wno-deprecated 

baseline: $(BUILD)/baseline-main.o $(BUILD)/hung.o 
	$(CC) -o $(BUILD)/baseline $(BUILD)/baseline-main.o $(BUILD)/hung.o $(CFLAGS)

$(BUILD)/baseline-main.o : $(SRC)/main.cpp $(SRC)/Experiment.h $(SRC)/Environment.h $(SRC)/Hungarian.h $(SRC)/PermutationSolver.h
	$(CC) -c $(SRC)/main.cpp -o $(BUILD)/baseline-main.o $(CFLAGS)

$(BUILD)/hung.o: $(SRC)/Hungarian.cpp $(SRC)/Hungarian.h
	$(CC) -c $(SRC)/Hungarian.cpp -o $(BUILD)/hung.o $(CFLAGS)
