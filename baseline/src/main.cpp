/**

    Implement the baseline:
    - read the data from the txt file 
    - read the vectors for each token
    - solve the sliding window bipartite graph matching 
      to compute the alignment matrix 
*/

#include <iostream>
#include <string>
#include <vector>
#include <set>
#include <map>
#include <filesystem>
#include <fstream>
#include <future>
#include <sqlite3.h>
#include <chrono>
#include <cmath>
#include <ctime>
#include <iomanip>
#include <cstring>
#include <cmath>
#include <sstream>
#include <numeric>
#include <queue>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <faiss/IndexFlat.h>
#include <faiss/utils/distances.h>
#include <faiss/index_io.h>
#include <thread>
#include <future>
#include <omp.h>
#include <regex>
#include "thread_pool.hpp"
#include "../modules/timing.h"
#include "hungarian-algorithm-cpp/Hungarian.h"

std::mutex gmtx;
using namespace std;
// using idx_t = faiss::Index::index_t;



/**
 * Main Function
*/
int main(int argc, char const *argv[]) {
  // arguments: text1_location, text2_location, window_width, threshold, result_folder
  string text1_location = argv[1];
  string text2_location = argv[2];
  int k = stoi(argv[3]);
  double theta = stod(argv[4]);
  string result_folder = argv[5];
  
  
  
  return 0;
}