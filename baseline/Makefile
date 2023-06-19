CC = g++
BUILD = ./build
SRC = ./src
OPT_LINK = ./opt/usr/local/lib
HUNG = $(SRC)/hungarian-algorithm-cpp-master
MOD = ./modules
USER_INC = ./opt/usr/local/include
INC_DIR = $(HUNG) $(USER_INC) $(MOD) 
LINK_DIR = $(OPT_LINK)
INC_PARAMS = $(foreach d, $(INC_DIR), -I$d)
LINK_PARAMS = $(foreach l, $(LINK_DIR), -L$l)
CFLAGS = -lstdc++fs -fPIC -std=c++17 -Wno-deprecated -fopenmp $(LINK_PARAMS) $(INC_PARAMS) -lortools -lsqlite3 -lfaiss -lopenblas -llapack -ltbb

baseline: $(BUILD)/baseline-main.o $(BUILD)/hung.o $(BUILD)/timing.o
	$(CC) -g -o $(BUILD)/baseline $(BUILD)/baseline-main.o $(BUILD)/hung.o $(BUILD)/timing.o $(CFLAGS)

$(BUILD)/baseline-main.o : $(SRC)/main.cpp
	$(CC) -g -c $ $(SRC)/main.cpp $(CFLAGS) -o $(BUILD)/baseline-main.o
	
$(BUILD)/hung.o: $(HUNG)/Hungarian.cpp $(HUNG)/Hungarian.h
	$(CC) -g -c $(HUNG)/Hungarian.cpp -o $(BUILD)/hung.o

$(BUILD)/timing.o: $(MOD)/timing.cxx $(MOD)/timing.h
	$(CC) -g -c $(MOD)/timing.cxx -o $(BUILD)/timing.o
