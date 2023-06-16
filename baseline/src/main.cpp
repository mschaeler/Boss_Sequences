/*

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
#include "modules/timing.h"
#include "hungarian-algorithm-cpp-master/Hungarian.h"

// #include "hungarian-algorithm-cpp-master/Hungarian.h"
// #include "absl/flags/parse.h"
// #include "absl/flags/usage.h"
// #include "ortools/algorithms/hungarian.h"
// #include "absl/container/flat_hash_map.h"
// #include "modules/timing.h"
// #include <sys/wait.h>
// #include <oneapi/tbb.h>

