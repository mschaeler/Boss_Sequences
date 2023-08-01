/**
 * @author: Pranay Mundra
 * @package: Semantic Alignment Calculator
 * @version: 1.3
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
#include <deque>
#include <numeric>
#include <queue>
#include <limits>
#include <random>
#include <algorithm>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <future>
// #include <omp.h>
#include <regex>
#include "thread_pool.hpp"
#include "../modules/timing.h"
#include "./hungarian-algorithm-cpp-master/Hungarian.h"

std::mutex gmtx;
using namespace std;
namespace fs = std::filesystem;
inline size_t key(int i,int j) {return (size_t) i << 32 | (unsigned int) j;} // concat unsigned int with two integer set id as an edge's integer id

std::vector<std::vector<int>> slidingWindows(vector<int>& nums, int k) {
    std::vector<std::vector<int>> result;
    if (nums.empty() || k <= 0 || k > nums.size()) {
        return result;
    }

    auto it = nums.begin();
    std::vector<int> window(it, std::next(it, k));
    result.push_back(window);

    while (std::next(it, k) != nums.end()) {
        window.erase(window.begin());
        window.push_back(*std::next(it, k));
        result.push_back(window);
        ++it;
    }

    return result;
}

/**
 * Environment
*/

class Environment {
	private:
		std::unordered_map<int, vector<int>> sets;
		set<int> text1sets;
		set<int> text2sets;
		std::unordered_set<int> wordSet;
		vector<set<int>> invertedIndex;
		std::unordered_map<int, string> int2word;
		std::unordered_map<string, int> word2int;
		std::unordered_map<int, string> int2set;
		std::unordered_map<string, int> set2int;
		int text1_average_cardinality = 0;
		int text2_average_cardinality = 0;
		
	public:
		Environment(string text1location, string text2location) {
			cout << "creating environment with lakes: " << text1location << ", " << text2location << endl;
			// int k = windowWidth;
			vector<string> text1_files;
			vector<string> text2_files;
			int id = 0;
			int tokens = 0;
			int sid = 0;
			for (const auto &entry: fs::directory_iterator(text1location)) {
				text1_files.push_back(entry.path());
			}

			for (const auto &entry: fs::directory_iterator(text2location)) {
				text2_files.push_back(entry.path());
			}

			cout << text1_files.size() << " text1_files listed" << endl;
			cout << text2_files.size() << " text2_files listed" << endl;

			for (size_t i = 0; i < text1_files.size(); ++i) {
				string f = text1_files[i];
				set2int[f] = sid;
				int2set[sid] = f;
				text1sets.insert(sid);
				string line;
				ifstream infile(f);
				if (infile.is_open()) {
					string line;
					while (getline(infile, line)) {
						tokens += 1;
						line.erase(line.find_last_not_of(" \n\r\t")+1);
						if (line.size() > 0) {
							if (word2int.find(line) == word2int.end()) {
								word2int[line] = id;
								wordSet.insert(id);
								int2word[id] = line;
								sets[sid].push_back(id);
								std::set<int> nset = {sid};
								invertedIndex.push_back(nset);
								id += 1;
							} else {
								sets[sid].push_back(word2int[line]);
								invertedIndex[word2int[line]].insert(sid);
							}
						}
					}
				}
				sid += 1;
			}
			int text1_tokens = tokens;
			int text1_sets = sets.size();
			text1_average_cardinality = text1_tokens / text1_sets;

			for (size_t i = 0; i < text2_files.size(); ++i) {
				string f = text2_files[i];
				set2int[f] = sid;
				int2set[sid] = f;
				text2sets.insert(sid);
				ifstream infile(f);
				if (infile.is_open()) {
					string line;
					while (getline(infile, line)) {
						tokens += 1;
						line.erase(line.find_last_not_of(" \n\r\t")+1);
						// cout << "Line for File: " << f << " " << line << endl;
						if (line.size() > 0) {
							if (word2int.find(line) == word2int.end()) {
								word2int[line] = id;
								wordSet.insert(id);
								int2word[id] = line;
								sets[sid].push_back(id);
								std::set<int> nset = {sid};
								invertedIndex.push_back(nset);
								id += 1;
							} else {
								sets[sid].push_back(word2int[line]);
								invertedIndex[word2int[line]].insert(sid);
							}
						}
					}
				}
				sid += 1;
				
			}

			text2_average_cardinality = (tokens - text1_tokens) / (sets.size() - text1_sets);

		}

		std::unordered_map<int, vector<int>> getSets() {
			return sets;
		}

		std::unordered_map<int, vector<vector<int>>> computeSlidingWindows(int windowWidth) {
			std::unordered_map<int, vector<vector<int>>> windows;
			for (auto i = sets.begin(); i != sets.end(); i++) {
				windows[i->first] = slidingWindows(i->second, windowWidth);
			}
			return windows;
		}
		
		std::unordered_set<int>& getWordSet() {
			return wordSet;
		}

		vector<set<int>>& getInvertedIndex() {
			return invertedIndex;
		}

		int getText1Avg() {
			return text1_average_cardinality;
		}

		int getText2Avg() {
			return text2_average_cardinality;
		}

		int toInt(string t) {
			if (word2int.find(t) == word2int.end()) {
				return -1;
			} else {
				return word2int[t];
			}
		}

		string toWord(int i) {
			return int2word[i];
		}

		int getSetId(string s) {
			if (set2int.find(s) == set2int.end()) {
				return -1;
			} else {
				return set2int[s];
			}
		}

		string getSetName(int i) {
			return int2set[i];
		}

		set<int>& getText1SetIds() {
			return text1sets;
		}

		set<int>& getText2SetIds() {
			return text2sets;
		}
};


/**
 * Data loader to read vectors from a tsv file
*/
class DataLoader {
	private:
		Environment *env;
		std::unordered_map<int, vector<float>> vectors;
		vector<int> dictionary;
		std::unordered_map<size_t, double> cache;


		std::vector<string> splitString(const string& line, char del) {
			std::vector<string> result;
			stringstream ss(line);
			string item;

			while (getline(ss, item, del)) {
				result.push_back(item);
			}
			return result;
		}

		void normalizeVector(vector<float>& vec) {
			float norm = 0.0f;

			// Compute the L2 norm
			for (float value : vec) {
				norm += value * value;
			}
			norm = sqrt(norm);

			// Normalize the vector using the L2 norm
			for (float& value : vec) {
				value /= norm;
			}
		}
	public:
		DataLoader(string location, Environment *e){
			// read TSV file
			env = e;
			ifstream file(location);
			if (file.is_open()) {
				string line;
				while (getline(file, line)) {
					vector<string> values = splitString(line, '\t');
					// token, lem_token, score between token & lem_token, vector values
					int token_id = env->toInt(values[1]);
					if (token_id != -1) {
						vector<float> embedding;
						for (int i = 3; i < values.size(); i++) {
							embedding.push_back(stof(values[i]));
						}
						// float* r = new float[300];
						// r = embedding.data();
						// faiss::fvec_renorm_L2(300, 1, r);
						// vector<float> vr(r, r + 300);
						// vectors[token_id] = vr;
						normalizeVector(embedding);
						vectors[token_id] = embedding;
						dictionary.push_back(token_id);
					}
				}
			}
			file.close();
		}

		vector<float> get_vector(int id){
			if (vectors.find(id) == vectors.end()) {
				vector<float> emptyVector(300, 0.0f);
				return emptyVector;
			} else {
				return vectors[id];
			}
		}

		set<int> getWords(){
			set<int> keys;
			for (auto kv : vectors) {
				keys.insert(kv.first);
			}
			return keys;
		}

		vector<vector<float>> getAllVectors() {
			vector<vector<float>> result;
			for (auto kv : vectors) {
				result.push_back(kv.second);
			}
			return result;
		}

		vector<int> getDictionary() {
			return dictionary;
		}

		double calculate_similarity(int token1_ID, int token2_ID) {
			// if same tokens similarity is 1
			if (token1_ID == token2_ID) {
				return 1.0;
			}
			// if either token doesn't have a vector then similarity is 0
			if ((vectors.find(token1_ID) == vectors.end()) || (vectors.find(token2_ID) == vectors.end())) {
				return 0.0;
			}
			// if similarity was calculated before we simply return
			if (cache.find(key(token1_ID, token2_ID)) != cache.end()) {
				return cache[key(token1_ID, token2_ID)];
			}
			// we calculate the similarity and cache it
			float dot = 0.0, denom_a = 0.0, denom_b = 0.0;
			vector<float> A = vectors[token1_ID];
			vector<float> B = vectors[token2_ID];
			for(int i = 0; i < A.size(); ++i) {
				dot += A[i] * B[i];
				denom_a += A[i] * A[i];
				denom_b += B[i] * B[i];
			}
			// double sim = double(dot / (sqrt(denom_a) * sqrt(denom_b)));
			double sim = static_cast<double>(dot);
			cache[key(token1_ID, token2_ID)] = sim;
			return sim;
		}
};


/**
 * @author Pranay Mundra
 * @package C++ Implementation of the Hungarian Algorithm orginally by Kevin L. Stern
 * @copyright Kevin L. Stern
*/
class HungarianKevinStern {
private:
    const int dim;
    std::vector<std::vector<double>> costMatrix;
    std::vector<double> labelByWorker, labelByJob, minSlackValueByJob;
    std::vector<int> minSlackWorkerByJob, matchJobByWorker, matchWorkerByJob;
    std::vector<int> parentWorkerByCommittedJob;
    std::vector<bool> committedWorkers;

    void computeInitialFeasibleSolution() {
        for (int j = 0; j < dim; j++) {
            labelByJob[j] = std::numeric_limits<double>::infinity();
        }

        for (int w = 0; w < dim; w++) {
            for (int j = 0; j < dim; j++) {
                if (costMatrix[w][j] < labelByJob[j]) {
                    labelByJob[j] = costMatrix[w][j];
                }
            }
        }
    }
    /**
	 * Execute a single phase of the algorithm. A phase of the Hungarian algorithm
	 * consists of building a set of committed workers and a set of committed jobs
	 * from a root unmatched worker by following alternating unmatched/matched
	 * zero-slack edges. If an unmatched job is encountered, then an augmenting path
	 * has been found and the matching is grown. If the connected zero-slack edges
	 * have been exhausted, the labels of committed workers are increased by the
	 * minimum slack among committed workers and non-committed jobs to create more
	 * zero-slack edges (the labels of committed jobs are simultaneously decreased
	 * by the same amount in order to maintain a feasible labeling).
	 * <p>
	 * 
	 * The runtime of a single phase of the algorithm is O(n^2), where n is the
	 * dimension of the internal square cost matrix, since each edge is visited at
	 * most once and since increasing the labeling is accomplished in time O(n) by
	 * maintaining the minimum slack values among non-committed jobs. When a phase
	 * completes, the matching will have increased in size.
	 */
    void executePhase() {
        while (true) {
            int minSlackWorker = -1, minSlackJob = -1;
            double minSlackValue = std::numeric_limits<double>::infinity();

            for (int j = 0; j < dim; j++) {
                if (parentWorkerByCommittedJob[j] == -1) {
                    if (minSlackValueByJob[j] < minSlackValue) {
                        minSlackValue = minSlackValueByJob[j];
                        minSlackWorker = minSlackWorkerByJob[j];
                        minSlackJob = j;
                    }
                }
            }

            if (minSlackValue > 0) {
                updateLabeling(minSlackValue);
            }

            parentWorkerByCommittedJob[minSlackJob] = minSlackWorker;

            if (matchWorkerByJob[minSlackJob] == -1) {
                /*
				 * An augmenting path has been found.
				 */
                int committedJob = minSlackJob;
                int parentWorker = parentWorkerByCommittedJob[committedJob];

                while (true) {
                    int temp = matchJobByWorker[parentWorker];
                    match(parentWorker, committedJob);
                    committedJob = temp;

                    if (committedJob == -1) {
                        break;
                    }

                    parentWorker = parentWorkerByCommittedJob[committedJob];
                }

                return;
            } else {
                /*
				 * Update slack values since we increased the size of the committed workers set.
				 */
                int worker = matchWorkerByJob[minSlackJob];
                committedWorkers[worker] = true;

                for (int j = 0; j < dim; j++) {
                    if (parentWorkerByCommittedJob[j] == -1) {
                        double slack = costMatrix[worker][j] - labelByWorker[worker] - labelByJob[j];

                        if (minSlackValueByJob[j] > slack) {
                            minSlackValueByJob[j] = slack;
                            minSlackWorkerByJob[j] = worker;
                        }
                    }
                }
            }
        }
    }

    /**
	 * 
	 * @return the first unmatched worker or {@link #dim} if none.
	 */
    int fetchUnmatchedWorker() {
        for (int w = 0; w < dim; w++) {
            if (matchJobByWorker[w] == -1) {
                return w;
            }
        }

        return dim;
    }

    /**
	 * Find a valid matching by greedily selecting among zero-cost matchings. This
	 * is a heuristic to jump-start the augmentation algorithm.
	 */
    void greedyMatch() {
        for (int w = 0; w < dim; w++) {
            for (int j = 0; j < dim; j++) {
                if (matchJobByWorker[w] == -1 && matchWorkerByJob[j] == -1 &&
                    costMatrix[w][j] - labelByWorker[w] - labelByJob[j] == 0) {
                    match(w, j);
                }
            }
        }
    }

    /**
	 * Initialize the next phase of the algorithm by clearing the committed workers
	 * and jobs sets and by initializing the slack arrays to the values
	 * corresponding to the specified root worker.
	 * 
	 * @param w the worker at which to root the next phase.
	 */
    void initializePhase(int w) {
        committedWorkers.assign(dim, false);
        parentWorkerByCommittedJob.assign(dim, -1);
        committedWorkers[w] = true;

        for (int j = 0; j < dim; j++) {
            minSlackValueByJob[j] = costMatrix[w][j] - labelByWorker[w] - labelByJob[j];
            minSlackWorkerByJob[j] = w;
        }
    }

    /**
	 * Helper method to record a matching between worker w and job j.
	 */
    void match(int w, int j) {
        matchJobByWorker[w] = j;
        matchWorkerByJob[j] = w;
    }

    /**
	 * Reduce the cost matrix by subtracting the smallest element of each row from
	 * all elements of the row as well as the smallest element of each column from
	 * all elements of the column. Note that an optimal assignment for a reduced
	 * cost matrix is optimal for the original cost matrix.
	 */
    void reduce() {
        for (int w = 0; w < dim; w++) {
            double min = std::numeric_limits<double>::infinity();

            for (int j = 0; j < dim; j++) {
                if (costMatrix[w][j] < min) {
                    min = costMatrix[w][j];
                }
            }

            for (int j = 0; j < dim; j++) {
                costMatrix[w][j] -= min; //XXX here we indeed modify the matrix
            }
        }

        std::vector<double> min(dim, std::numeric_limits<double>::infinity());

        for (int j = 0; j < dim; j++) {
            for (int w = 0; w < dim; w++) {
                if (costMatrix[w][j] < min[j]) {
                    min[j] = costMatrix[w][j];
                }
            }
        }

        for (int w = 0; w < dim; w++) {
            for (int j = 0; j < dim; j++) {
                costMatrix[w][j] -= min[j]; //XXX here we indeed modify the matrix
            }
        }
    }

    /**
	 * Update labels with the specified slack by adding the slack value for
	 * committed workers and by subtracting the slack value for committed jobs. In
	 * addition, update the minimum slack values appropriately.
	 */
    void updateLabeling(double slack) {
        for (int w = 0; w < dim; w++) {
            if (committedWorkers[w]) {
                labelByWorker[w] += slack;
            }
        }

        for (int j = 0; j < dim; j++) {
            if (parentWorkerByCommittedJob[j] != -1) {
                labelByJob[j] -= slack;
            } else {
                minSlackValueByJob[j] -= slack;
            }
        }
    }

public:
    /**
	 * Construct an instance of the algorithm.
	 * 
	 * @param costMatrix the cost matrix, where matrix[i][j] holds the cost of
	 *                   assigning worker i to job j, for all i, j. The cost matrix
	 *                   must not be irregular in the sense that all rows must be
	 *                   the same length; in addition, all entries must be
	 *                   non-infinite numbers.
	 */
    HungarianKevinStern(int k) : dim(k), costMatrix(k, std::vector<double>(k, 0.0)),
                                 labelByWorker(k, 0.0), labelByJob(k, 0.0),
                                 minSlackValueByJob(k, 0.0),
                                 minSlackWorkerByJob(k, 0),
                                 matchJobByWorker(k, -1), matchWorkerByJob(k, -1),
                                 parentWorkerByCommittedJob(k, -1),
                                 committedWorkers(k, false) {}

    /**
	 * Execute the algorithm.
	 * 
	 * @return the minimum cost matching of workers to jobs based upon the provided
	 *         cost matrix. A matching value of -1 indicates that the corresponding
	 *         worker is unassigned.
	 */

    double solve(const std::vector<std::vector<double>>& org_cost_matrix, double threshold) {
        // Note: we need to copy the matrix, as we'll modify the values in between
        for (int w = 0; w < dim; w++) {
            costMatrix[w] = org_cost_matrix[w];
        }

        reduce();
        computeInitialFeasibleSolution();
        greedyMatch();

        int w = fetchUnmatchedWorker();

        while (w < dim) {
            initializePhase(w);
            executePhase();
            w = fetchUnmatchedWorker();
        }

        // DONE - Collect the result
        double cost = 0.0;
        double cost_2 = 0.0;

        for (w = 0; w < matchJobByWorker.size(); w++) {
            cost += org_cost_matrix[w][matchJobByWorker[w]];
            cost_2 += org_cost_matrix[w][matchWorkerByJob[w]];
        }

        return cost;
    }
};


/**
* BIPARTITE GRAPH MATRIX
*/
class ValidMatrix {
	private:
		vector<vector<double>> M; // the matrix
		vector<vector<pair<int, int>>> M_tokens; // the matrix of tokens
		vector<int> non_exact_token_indices;
		vector<int> assignment_internal;
		int matching;
		int realsize;
	public:
		// building a valid matrix with query set and target set and valid edges valid edges stand 
        // for edges popped from the monotonically decreasing priority queue by edge weights 
        // (cosine similarity) by leveraging valid edges, we save computing resources by do not 
        // have to re-calculate cosine similarity for edges retrieved from the Faiss index (which is expensive)
		ValidMatrix(vector<int> query_set, vector<int> target_set, std::unordered_map<size_t, double> validedge) {
			int i = 0;
			vector<int> query_set_pruned;
			vector<int> target_set_pruned;
			matching = min(query_set.size(), target_set.size());
			for(vector<int>::iterator itq = query_set.begin(); itq != query_set.end(); ++itq) {
				vector<double> temp;
				int qword = *itq;
				for(vector<int>::iterator itt = target_set.begin(); itt != target_set.end(); ++itt) {
					int tword = *itt;
					if (validedge.find(key(qword, tword)) != validedge.end()) {
						query_set_pruned.push_back(qword);
						target_set_pruned.push_back(tword);
					} else {
						if (qword == tword) {
							query_set_pruned.push_back(qword);
							target_set_pruned.push_back(tword);
						}
					}// bucket upperbound not considering this
				}
			}
			realsize = query_set_pruned.size() * target_set_pruned.size();
			// vector<int> query_tokens;
			for(vector<int>::iterator itq = query_set_pruned.begin(); itq != query_set_pruned.end(); ++itq) {
				vector<double> temp;
				vector<pair<int, int>> temp_tokens;
				int qword = *itq;
				for(vector<int>::iterator itt = target_set_pruned.begin(); itt != target_set_pruned.end(); ++itt) {
					int tword = *itt;
					if (validedge.find(key(qword, tword)) != validedge.end()) {
						temp.push_back(0.0 - validedge[key(qword, tword)]);
					} else {
						if (qword == tword) {
							temp.push_back(-1.0);
						} else {
							temp.push_back(0.0);
						}
					}
					temp_tokens.push_back(make_pair(qword, tword));
				}
				M.push_back(temp);
				M_tokens.push_back(temp_tokens);
			}
		}

		ValidMatrix(vector<int> query_set, vector<int> target_set, DataLoader *dl) {
			int i = 0;
			matching = min(query_set.size(), target_set.size());
			for(vector<int>::iterator itq = query_set.begin(); itq != query_set.end(); ++itq) {
				vector<double> temp;
				vector<pair<int, int>> temp_tokens;
				int qword = *itq;
				for(vector<int>::iterator itt = target_set.begin(); itt != target_set.end(); ++itt) {
					int tword = *itt;
					temp.push_back(0.0 - dl->calculate_similarity(qword, tword));
					temp_tokens.push_back(make_pair(qword, tword));
				}
				M.push_back(temp);
				M_tokens.push_back(temp_tokens);
			}
		}

		const vector<vector<double>> reveal() {
			return M;
		}

		double solveQ(int q) {
			HungarianAlgorithm HungAlgo;
			vector<int> assignment;
			if (matching == 0 || M.size() == 0) {
				return 0.0;
			}
			double cost = HungAlgo.Solve(M, assignment);
			return -cost/q;
			// return 1.0;
		}

		// vector<pair<pair<int, int>, double>> get_non_exact_tokens() {
		// 	vector<pair<pair<int, int>, double>> non_exact_tokens;
		// 	int nRows = M.size();
		// 	int nCols = M[0].size();
		// 	for (int i = 0; i < nRows; i++) {
		// 		if (std::find(non_exact_token_indices.begin(), non_exact_token_indices.end(), i) != non_exact_token_indices.end()) {
		// 			int j = assignment_internal[i];
		// 			double asim = -M[i][j];
		// 			pair<int, int> qt_pair = M_tokens[i][j];
		// 			non_exact_tokens.push_back(make_pair(qt_pair, asim));
		// 		}
		// 	}
		// 	return non_exact_tokens;
		// }
};


class GlobalAMatrix {
	private:
		vector<vector<double>> costMatrix;
		vector<vector<pair<double, double>>> rowStats;
		vector<vector<pair<double, double>>> colStats;
		double* alignmentMatrix;
		vector<int> *set1Tokens;
		vector<int> *set2Tokens;
		double theta;
		DataLoader *dl;
		int width;
		int height;
		int zero_entries = 0;
		int num_cells_pruned = 0;
		int num_cells_geq_threshold_estimate = 0;
		int num_cells_geq_threshold = 0;

	public:
		GlobalAMatrix(vector<int>* _set1Tokens, vector<int>* _set2Tokens, double _theta, DataLoader* _dl, int k) {
			set1Tokens = _set1Tokens;
			set2Tokens = _set2Tokens;
			theta = _theta;
			dl = _dl;
			// calculate the cost matrix
			for (vector<int>::iterator it1 = set1Tokens->begin(); it1 != set1Tokens->end(); ++it1) {
				vector<double> temp;
				int word1 = *it1;
				for (vector<int>::iterator it2 = set2Tokens->begin(); it2 != set2Tokens->end(); ++it2) {
					int word2 = *it2;
					temp.push_back(0.0 - dl->calculate_similarity(word1, word2));
				}
				costMatrix.push_back(temp);
			}
            width = set1Tokens->size() - k + 1;
			height = set2Tokens->size() - k + 1;
			alignmentMatrix = new double[width * height];
		}


		// GlobalAMatrix(vector<int>* _set1Tokens, vector<int>* _set2Tokens, double _theta, DataLoader* _dl, int k) : 
		// 	rowStats(_set1Tokens->size(), std::vector<std::pair<double, double>>(_set2Tokens->size())), colStats(_set1Tokens->size(), std::vector<std::pair<double, double>>(_set2Tokens->size())),
		// 	set1Tokens(_set1Tokens), set2Tokens(_set2Tokens), theta(_theta), dl(_dl) {
		// 	/**
		// 	 * @todo : How to populate colStats on the fly?
		// 	*/
		// 	// calculate the cost matrix
		// 	for (int i = 0; i < set1Tokens->size(); i++) {
		// 		vector<double> temp;
		// 		int word1 = set1Tokens->at(i);
		// 		double currMin = std::numeric_limits<double>::infinity();
		// 		double currMax = 0.0;
		// 		for (int j = 0; j < set2Tokens->size(); j++) {
		// 			int word2 = set2Tokens->at(j);
		// 			double sim = dl->calculate_similarity(word1, word2);
		// 			temp.push_back(0.0 - sim);
		// 			// handle the min update:
		// 			if (currMin > sim) {
		// 				currMin = sim;
		// 				int start_index = max(0, j - k + 1);
		// 				for (int s = start_index; s <= j; s++) {
		// 					rowStats[i][s].first = currMin;
		// 				}
		// 			}

		// 			// handle the max update:
		// 			if (sim > currMax) {
		// 				currMax = sim;
		// 				int start_index = max(0, j - k + 1);
		// 				for (int s = start_index; s <= j; s++) {
		// 					rowStats[i][s].second = currMax;
		// 				}
		// 			}
		// 		}
		// 		costMatrix.push_back(temp);
		// 	}
        //     width = set1Tokens->size() - k + 1;
		// 	height = set2Tokens->size() - k + 1;
		// 	alignmentMatrix = new double[width * height];
		// }

		void computeAlignment(int k) {
			int rows = costMatrix.size();
			int cols = costMatrix[0].size();
            HungarianAlgorithm HungAlgo;
			for (int i = 0; i <= rows - k; i++) {
				for (int j = 0; j <= cols - k; j++) {
					vector<vector<double>> currWindow(k, vector<double>(k));
					for (int m = 0; m < k; m++) {
						for (int n = 0; n < k; n++) {
							currWindow[m][n] = costMatrix[i + m][j + n];
						}
					}
					// HungarianKevinStern* HungAlgoStern = new HungarianKevinStern(k);
					vector<int> assignment;
					double sim;
					if (currWindow.size() == 0) {
						sim = 0.0;
					} else {
						double cost = HungAlgo.Solve(currWindow, assignment);
						// double cost = HungAlgoStern->solve(currWindow, theta);
						sim = -cost/k;
					}
					if (sim >= theta) {
						alignmentMatrix[i + j * width] = sim;
					} else {
						alignmentMatrix[i + j * width] = 0.0;
						zero_entries += 1;
					}
					
				}
			}
		}
		/**
		 * @param pruning_method : {0 : column_sum, 1 : matrix_min, 2 : row sum}
		*/
		void computeAlignmentWithPruning_precomputed(int k, int pruning_method) {
			int rows = costMatrix.size();
			int cols = costMatrix[0].size();
			for (int i = 0; i <= rows - k; i++) {
				for (int j = 0; j <= cols - k; j++) {
					vector<vector<double>> currWindow(k, vector<double>(k));
					double row_min_sum = 0.0;
					double row_max_sum = 0.0;
					for (int m = 0; m < k; m++) {
						int n;
						for (n = 0; n < k; n++) {
							currWindow[m][n] = costMatrix[i + m][j + n];
						}
						row_min_sum += rowStats[i + m][j + n - 1].first;
						row_max_sum += rowStats[i + m][j + n - 1].second;
					}
					HungarianAlgorithm HungAlgo;
					HungarianKevinStern* HungAlgoStern = new HungarianKevinStern(k);
					vector<int> assignment;
					double sim;
					double lb_cost;
					if (pruning_method == 0) {

					} else if (pruning_method == 1) {

					} else {
						lb_cost = row_min_sum;
					}

					double est_normalized_sim = - (lb_cost / static_cast<double>(k));
					if (currWindow.size() == 0) {
						sim = 0.0;
					} else if (est_normalized_sim > theta) {
						// sim = est_normalized_sim;
						num_cells_geq_threshold_estimate += 1;
						double cost = HungAlgoStern->solve(currWindow, theta);
						sim = -cost/k;
						if (sim >= theta) {
							alignmentMatrix[i + j * width] = sim;
							num_cells_geq_threshold += 1;
						} else {
							alignmentMatrix[i + j * width] = 0.0;
							zero_entries += 1;
						}
					} else {
						num_cells_pruned += 1;
					}
					
				}
			}
		}


		void computeAlignmentWithPruning(int k, int pruning_method) {
			int rows = costMatrix.size();
			int cols = costMatrix[0].size();
            HungarianAlgorithm HungAlgo;
			for (int i = 0; i <= rows - k; i++) {
				for (int j = 0; j <= cols - k; j++) {
					vector<vector<double>> currWindow(k, vector<double>(k));
					for (int m = 0; m < k; m++) {
						int n;
						for (n = 0; n < k; n++) {
							currWindow[m][n] = costMatrix[i + m][j + n];
						}
					}
					// HungarianAlgorithm HungAlgo;
					// HungarianKevinStern* HungAlgoStern = new HungarianKevinStern(k);
					vector<int> assignment;
					double sim;
					double lb_cost;
					if (pruning_method == 0) {
						lb_cost = get_column_sum(currWindow);
					} else if (pruning_method == 1) {
						lb_cost = k * get_matrix_min(currWindow);
					} else {
						lb_cost = get_column_row_sum(currWindow);
					}

					double est_normalized_sim = - (lb_cost / static_cast<double>(k));
					if (currWindow.size() == 0) {
						sim = 0.0;
					} else if (est_normalized_sim > theta) {
						// sim = est_normalized_sim;
						num_cells_geq_threshold_estimate += 1;
						// double cost = HungAlgoStern->solve(currWindow, theta);
                        double cost = HungAlgo.Solve(currWindow, assignment);
						sim = -cost/k;
						if (sim >= theta) {
							alignmentMatrix[i + j * width] = sim;
							num_cells_geq_threshold += 1;
						} else {
							alignmentMatrix[i + j * width] = 0.0;
							zero_entries += 1;
						}
					} else {
						num_cells_pruned += 1;
						zero_entries += 1;
					}
					
				}
			}
		}

		double get_matrix_min(vector<vector<double>>& matrix) {
			double min = matrix[0][0];
			for (size_t i = 0; i < matrix.size(); i++) {
				for (size_t j = 0; j < matrix[i].size(); j++) {
					if (matrix[i][j] < min) {
						min = matrix[i][j];
					}
				}
			}
			return min;
		}

		double get_column_sum(vector<vector<double>>& matrix) {
			double col_sum = 0.0;
			for (size_t j = 0; j < matrix.size(); ++j) {
				double col_min = matrix[0][j];
				for (size_t i = 1; i < matrix[0].size(); ++i) {
					double cost = matrix[i][j];
					if (cost < col_min) {
						col_min = cost;
					}
				}
				col_sum += col_min;
			}
			return col_sum;
		}

		double sum(const std::vector<double>& vec) {
			double result = 0.0;
			for (double val : vec) {
				result += val;
			}
			return result;
		}

		double get_column_row_sum(vector<vector<double>>& matrix) {
			int k = matrix.size(); 
			std::vector<double> k_buffer(k, std::numeric_limits<double>::max());
			double row_sum = 0;

			for (int i = 0; i < k; ++i) {
				const std::vector<double>& line = matrix[i];
				double row_min = std::numeric_limits<double>::max();

				for (int j = 0; j < k; ++j) {
					const double val = line[j];
					if (val < row_min) {
						row_min = val;
					}

					if (val < k_buffer[j]) {
						k_buffer[j] = val;
					}
				}

				row_sum += row_min;
			}

			double col_sum = sum(k_buffer);
			double min_cost = std::max(row_sum, col_sum);

			return min_cost;
		}

		int get_matrix_size() {
			return width * height;
		}

		int zeroCells() {
			return zero_entries;
		}

		int prunedCount() {
			return num_cells_pruned;
		}

		pair<int, int> getEstimateCounts() {
			return make_pair(num_cells_geq_threshold_estimate, num_cells_geq_threshold);
		}

		void printAMatrix() {
			for (int i = 0; i < width; i++) {
				for (int j = 0; j < height; j++) {
					cout << alignmentMatrix[i + j * width] << "," << " \n"[j == height - 1];
				}
			}
		}

		vector<vector<double>> getCostMatrix() {
			return costMatrix;
		}

};


class AMatrix {
	private:
		int width;
		int height;
		double* data; // double* matrix 1-D representation of a matrix: matrix[i + j*width] = matrix[i][j].. matrix[width*height]... delete [] matrix;
		vector<vector<int>> *set1Windows;
		vector<vector<int>> *set2Windows;
		// unordered_map<size_t, double> validEdges;
		double theta;
		int zero_entries = 0;
		DataLoader *dl;
	
	public:
		AMatrix(vector<vector<int>>* Ws, vector<vector<int>>* Wt, double threshold, DataLoader *_dl) {
			width = Ws->size();
			height = Wt->size();
			set1Windows = Ws;
			set2Windows = Wt;
			// validEdges = validedges;
			theta = threshold;
			data = new double[width * height];
			dl = _dl;
		}

		void computeAlignment() {
			for (int i = 0; i < set1Windows->size(); ++i) {
				int nsize = set2Windows->size();
				ValidMatrix *m;
				double sim;
				int j;
				// #pragma omp parallel default(none) num_threads(4) shared(set1Windows, i, nsize, data) private(j, m, sim)
        		// #pragma omp for schedule(static)
				for (j = 0; j < nsize; ++j) {
					vector<int> set1_tokens = set1Windows->at(i);
					vector<int> set2_tokens = set2Windows->at(j);
                    m = new ValidMatrix(set1_tokens, set2_tokens, dl);
					sim = m->solveQ(set1_tokens.size());
					// sim = 1.0;
					if (sim >= theta) {
						data[i + j*width] = sim;
					} else {
						data[i + j*width] = 0.0;
						zero_entries += 1;
					}
				}
			}
		}

		double getMatrixValue(int row, int col) {
			return data[row + col * width];
		}

		void setMatrixValue(int row, int col, double value) {
			data[row + col * width] = value;
		}

		int get_matrix_size() {
			return width * height;
		}

		int zeroCells() {
			return zero_entries;
		}
};

struct cmp_decreasing {
	bool operator() (pair<double, int> lhs, pair<double, int> rhs) const {
		return std::get<0>(lhs) >= std::get<0>(rhs);
	}
};

struct sliding_window_ordering {
	bool operator() (pair<int, vector<int>> lhs, pair<int, vector<int>> rhs) const {
		return std::lexicographical_compare(std::get<1>(lhs).begin(), std::get<1>(lhs).end(), std::get<1>(rhs).begin(), std::get<1>(rhs).end());
	}
};

// Custom hash function for vector<int>
struct VectorHash {
	size_t operator()(const vector<int>& vec) const {
		size_t hash = 0;
		for (int num : vec) {
			hash ^= hash << 1 ^ num;
		}
		return hash;
	}
};

class IndexStructures{
	private:
		vector<int>* set1Tokens; // query tokens
		vector<int>* set2Tokens; // candidate tokens
		DataLoader *dl;
		int k;
		double theta;
		std::vector<std::vector<double>> costMatrix;
		// std::unordered_map<int, std::vector<std::vector<int>>> slidingWindowInvertedIndex;
		std::unordered_map<int, std::set<pair<int, vector<int>>, sliding_window_ordering>> slidingWindowInvertedIndex_v3;
		std::unordered_map<int, std::vector<pair<int, std::vector<int>>>> slidingWindowInvertedIndex_v2;
		std::unordered_map<int, std::set<pair<double, int>, cmp_decreasing>> tokensSimIndex;
		double tokenSimIndexTime = 0.0;

	public:
		IndexStructures(vector<int>* _set1Tokens, vector<int>* _set2Tokens, DataLoader* _dl, int _k, double _theta) : 
			set1Tokens(_set1Tokens), set2Tokens(_set2Tokens), dl(_dl), k(_k), theta(_theta) 
		{
			// compute the tokenSimIndex
			std::chrono::time_point<std::chrono::high_resolution_clock> simstart, simend;
			std::chrono::duration<double> sim_elapsed;
			simstart = std::chrono::high_resolution_clock::now();
			for (vector<int>::iterator it1 = set1Tokens->begin(); it1 != set1Tokens->end(); ++it1) {
				int word1 = *it1;
				for (vector<int>::iterator it2 = set2Tokens->begin(); it2 != set2Tokens->end(); ++it2) {
					int word2 = *it2;
					double sim = dl->calculate_similarity(word1, word2);
					if (sim >= theta) {
						tokensSimIndex[word1].insert(make_pair(sim, word2));
					}
					// tokensSimIndex[word1].insert(make_pair(sim, word2));
				}
			}
			simend = std::chrono::high_resolution_clock::now();
			sim_elapsed = simend - simstart;
			tokenSimIndexTime = sim_elapsed.count();

			// compute slidingWindowInvertedIndex
			vector<vector<int>> set1SlidingWindows = slidingWindows(*set1Tokens, k);
			vector<vector<int>> set2SlidingWindows = slidingWindows(*set2Tokens, k);

			// for (auto it1 = set1SlidingWindows.begin(); it1 != set1SlidingWindows.end(); ++it1) {
			// 	vector<int> currWindow = *it1;
			// 	for (int t : currWindow) {
			// 		slidingWindowInvertedIndex[t].push_back(currWindow);
			// 	}
			// }

			// for (auto it2 = set2SlidingWindows.begin(); it2 != set2SlidingWindows.end(); ++it2) {
			// 	vector<int> currWindow = *it2;
			// 	for (int t : currWindow) {
			// 		slidingWindowInvertedIndex[t].push_back(currWindow);
			// 	}
			// }

			for (int j = 0; j < set2SlidingWindows.size(); j++) {
				vector<int> currWindow = set2SlidingWindows[j];
				for (int t : currWindow) {
					// slidingWindowInvertedIndex_v2[t].push_back(make_pair(j, currWindow));
					slidingWindowInvertedIndex_v3[t].insert(make_pair(j, currWindow));
				}
			}
		}

		// Function to compute the intersection of two vector<vector<int>>
		// vector<vector<int>> intersect(const vector<vector<int>>& v1, const vector<vector<int>>& v2) {
		// 	unordered_set<vector<int>, VectorHash> set1(v1.begin(), v1.end());
		// 	vector<vector<int>> result;

		// 	for (const auto& v : v2) {
		// 		if (set1.count(v) > 0) {
		// 			result.push_back(v);
		// 		}
		// 	}

		// 	return result;
		// }
		vector<vector<int>> intersect(const vector<vector<int>>& v1, const vector<pair<int, vector<int>>>& v2) {
			unordered_set<vector<int>, VectorHash> set1(v1.begin(), v1.end());
			vector<vector<int>> result;

			for (const auto& v : v2) {
				if (set1.count(v.second) > 0) {
					result.push_back(v.second);
				}
			}

			return result;
		}


		// sort functions
		// static bool compareVectors(const std::vector<int>& v1, const std::vector<int>& v2) {
		// 	return std::lexicographical_compare(v1.begin(), v1.end(), v2.begin(), v2.end());
		// }

		// static bool pairComparator(const std::pair<int, std::vector<int>>& p1, const std::pair<int, std::vector<int>>& p2) {
		// 	return compareVectors(p1.second, p2.second);
		// }
		// Function to find candidate windows based on query window and data structures
		vector<vector<int>> findCandidateWindows(const vector<int>& queryWindow) {
			vector<vector<int>> candidates;

			for (int token : queryWindow) {
				// Get the list of similar tokens
				if (tokensSimIndex.find(token) != tokensSimIndex.end()) {
					const set<pair<double, int>, cmp_decreasing>& simTokens = tokensSimIndex.at(token);

					for (const auto& simToken : simTokens) {
						int similarToken = simToken.second;

						// Get the sliding windows for the similar token
						if (slidingWindowInvertedIndex_v2.find(similarToken) != slidingWindowInvertedIndex_v2.end()) {
							// const vector<vector<int>>& slidingWindows = slidingWindowInvertedIndex.at(similarToken);
							const vector<pair<int, vector<int>>>& slidingWindows = slidingWindowInvertedIndex_v2.at(similarToken);
							// std::sort(slidingWindows.begin(), slidingWindows.end(), [](const auto& p1, const auto& p2) {
							// 	const auto& v1 = p1.second;
							// 	const auto& v2 = p2.second;
							// 	return std::lexicographical_compare(v1.begin(), v1.end(), v2.begin(), v2.end());
							// });
							// Compute the intersection of sliding windows & candidates
							if (!candidates.empty()) {
								candidates = intersect(candidates, slidingWindows);
							} else {
								for (auto t : slidingWindows) {
									candidates.push_back(t.second);
								}
							}
						}
					}
				}
			}

			return candidates;
		}

		std::set<std::pair<int, std::vector<int>>, sliding_window_ordering> computeIntersection(const std::set<std::pair<int, std::vector<int>>, sliding_window_ordering>& set1, const std::set<std::pair<int, std::vector<int>>, sliding_window_ordering>& set2) {

			std::set<std::pair<int, std::vector<int>>, sliding_window_ordering> intersection;

			// Ensure that both sets are sorted based on the lexicographical order of the vector<int>
			// If your sets are not already sorted, you can sort them before computing the intersection.

			// Set intersection requires both sets to be sorted.
			std::set_intersection(
				set1.begin(), set1.end(),
				set2.begin(), set2.end(),
				std::inserter(intersection, intersection.begin()),
				[](const auto& p1, const auto& p2) {
					// Comparator to compare pairs based on the lexicographical order of the vector<int>
					return std::lexicographical_compare(p1.second.begin(), p1.second.end(),
														p2.second.begin(), p2.second.end());
				});

			return intersection;
		}
		// Function to find candidate windows based on query window and data structures
		std::set<pair<int, vector<int>>, sliding_window_ordering> getCandidates(const vector<int>& queryWindow) {
			std::set<pair<int, vector<int>>, sliding_window_ordering> candidates;
			for (int token : queryWindow) {
				if (tokensSimIndex.find(token) != tokensSimIndex.end()) {
					const set<pair<double, int>, cmp_decreasing>& simTokens = tokensSimIndex.at(token);

					for (const auto& simToken : simTokens) {
						int similarToken = simToken.second;

						if (slidingWindowInvertedIndex_v3.find(similarToken) != slidingWindowInvertedIndex_v3.end()) {
							set<pair<int, vector<int>>, sliding_window_ordering>& slidingWindows = slidingWindowInvertedIndex_v3.at(similarToken);

							if (!candidates.empty()) {
								candidates = computeIntersection(candidates, slidingWindows);
							} else {
								candidates = slidingWindows;
							}

						}
					}
				}
			}

			return candidates;
		}

		void reportStats() {
			std::cout << "Index Structure Statistics" << std::endl;
			std::cout << "Sliding Window Inverted Index" << std::endl;
			int max_len = 0;
			int min_len = std::numeric_limits<int>::max();
			int sum = 0;
			for (auto it = slidingWindowInvertedIndex_v3.begin(); it != slidingWindowInvertedIndex_v3.end(); ++it) {
				int curr_len = it->second.size();
				max_len = max(max_len, curr_len);
				min_len = min(min_len, curr_len);
				sum += curr_len;
			}
			int average_len = static_cast<int>(sum / slidingWindowInvertedIndex_v3.size());
			std::cout << "Maximum Length: " << max_len << std::endl;
			std::cout << "Minimum Length: " << min_len << std::endl;
			std::cout << "Average Length: " << average_len << std::endl;
			std::cout << "Sim Time: "  << tokenSimIndexTime << std::endl;

		}

};

void baseline(Environment *env, DataLoader *dl, int k, double theta) {

	std::unordered_map<int, vector<int>> sets = env->getSets(); // all sets stored as key: set integer id, value: set data (int token integer ids)
	std::unordered_set<int> wordSet = env->getWordSet(); // all unique tokens in copora
	vector<set<int>> invertedIndex = env->getInvertedIndex(); // inverted index that returns all sets containing given token
	vector<int> dictionary = dl->getDictionary(); // vectors database instance
	set<int> text1Sets = env->getText1SetIds();
	set<int> text2Sets = env->getText2SetIds();
	std::mutex gmtx_internal;

	int numberOfGraphMatchingComputed = 0;
	int numberOfZeroEntries = 0;
	int pruned_cells = 0;
	int numCellsAboveThresoldEstimate = 0;
	int numCellsAboveThresold = 0;
	
	// std::unordered_map<size_t, AMatrix*> results; // key(text1SetId, text2SetId) --> AlignmentMatrix
	std::chrono::time_point<std::chrono::high_resolution_clock> sliding_window_start, sliding_window_end;
	std::chrono::duration<double> sliding_window_elapsed;
	double slidingWindowTime = 0.0;
	sliding_window_start = std::chrono::high_resolution_clock::now();
	std::unordered_map<int, vector<vector<int>>> kWidthWindows = env->computeSlidingWindows(k); // setID --> sliding windows
	sliding_window_end = std::chrono::high_resolution_clock::now();
	sliding_window_elapsed = sliding_window_end - sliding_window_start;
	slidingWindowTime = sliding_window_elapsed.count();
	cout << "Sliding Window Computation time: " << slidingWindowTime << endl;
 	std::unordered_map<size_t, double> validedges;
	/**
	 * @todo: Check the correctness in terms of getting the IDS from the dictionary
	*/
	// for all tokens between the two texts, do a faiss similarity search and cache the edges
	// int nq = 0;
	// int i = 0;
	// vector<float> vxq;
	// for (auto it = wordSet.begin(); it != wordSet.end(); it++) {
	// 	int tq = *it;
	// 	// handle out of dictionary words
	// 	if (std::find(dictionary.begin(), dictionary.end(), tq) == dictionary.end()) {
	// 		i += 1;
	// 		validedges[key(tq, tq)] = 1.0;
	// 	} else {
	// 		// vector<float> vec = db->get_normalized_vector(tq);
	// 		vector<float> vec = dl->get_vector(tq);
	// 		if (vec.size() == 0) {
	// 			cerr << "Vector should not be empty" << endl;
	// 		}
	// 		vxq.insert(vxq.end(), vec.begin(), vec.end());
	// 		nq += 1;
	// 	}
	// }

	// cout << "Out of dictionary words: " << i << endl;
	// // get the k nearest neighbours for each word, here set k = nq. 
	// // @todo: change to range search with radius = theta
	// std::chrono::time_point<std::chrono::high_resolution_clock> faiss_search_start, faiss_search_end;
	// std::chrono::duration<double> faiss_search_elapsed;
	// double faiss_search_time = 0.0;
	// faiss_search_start = std::chrono::high_resolution_clock::now();
	// tuple<vector<idx_t>, vector<float>> rt = faissIndex->kNNSearch(nq, vxq, faissIndex->index->ntotal, 300);
	// faiss_search_end = std::chrono::high_resolution_clock::now();
	// faiss_search_elapsed = faiss_search_end - faiss_search_start;
	// faiss_search_time = faiss_search_elapsed.count();
	// cout << "Faiss Search Time: " << faiss_search_time << endl;
	// vector<idx_t> I = std::get<0>(rt);
	// vector<float> D = std::get<1>(rt);
	// cout << "size of I: " << I.size() << endl;
	// int cur = 0;
	// for (vector<idx_t>::iterator it = I.begin(); it != I.end(); it++) {
	// 	int tq_cur = I[cur - (cur % nq)];
	// 	int tq = dictionary[tq_cur];
	// 	int word = dictionary[*it];
	// 	float fsim = D[cur];
	// 	double sim = static_cast<double>(fsim);
	// 	if (sim > 0.0) {
	// 		validedges[key(tq, word)] = sim;
	// 	}
	// 	cur += 1;
	// }

	// cout << "Size of valid edges: " << validedges.size() << endl;

	// for each set in text1Sets, we compute the k-width window and compute the alignment matrix
	std::chrono::time_point<std::chrono::high_resolution_clock> bs_search_start, bs_search_end;
	std::chrono::duration<double> bs_search_elapsed;
	double bs_search_time = 0.0;
	bs_search_start = std::chrono::high_resolution_clock::now();
	int counter = 1;

	// Local Cost Matrix
	// for (int set1Id : text1Sets) {
	// 	vector<vector<int>> &set1Windows = kWidthWindows[set1Id];
	// 	for (int set2Id : text2Sets) {
	// 		vector<vector<int>> &set2Windows = kWidthWindows[set2Id];
	// 		// compute the alignment matrix if not computed previously
	// 		// if (results.find(key(set1Id, set2Id)) == results.end()) {
	// 			AMatrix *A = new AMatrix(&set1Windows, &set2Windows, theta, dl);
	// 			A->computeAlignment();
	// 			// results[key(set1Id, set2Id)] = A;
	// 			numberOfGraphMatchingComputed += A->get_matrix_size();
	// 			numberOfZeroEntries += A->zeroCells();
	// 		// }
	// 	}
	// 	// cout << "Done with: " << counter << " / " << text1Sets.size() << endl;
	// 	counter += 1;
	// }

	// Permutation Index Search
	// for (int set1Id : text1Sets) {
	// 	vector<vector<int>> set1Windows = kWidthWindows[set1Id];
	// 	for (int set2Id : text2Sets) {
	// 		vector<vector<int>> set2Windows = kWidthWindows[set2Id];
	// 		// compute the alignment matrix if not computed previously
	// 		if (results.find(key(set1Id, set2Id)) == results.end()) {
	// 			PermutationOptimizedSearch* A = new PermutationOptimizedSearch(&set1Windows, &set2Windows, theta, dl, k);
	// 			A->computeAlignment();
	// 			A->printAMatrix();
	// 			numberOfGraphMatchingComputed += A->get_matrix_size();
	// 			numberOfZeroEntries += A->zeroCells();
	// 		}
	// 	}
	// 	// cout << "Done with: " << counter << " / " << text1Sets.size() << endl;
	// 	counter += 1;
	// }

	// // Global Cost Matrix 
	// for (int set1Id : text1Sets) {
	// 	vector<int> &set1Tokens = sets[set1Id];
	// 	for (int set2Id : text2Sets) {
	// 		vector<int> &set2Tokens = sets[set2Id];
	// 		// compute the alignment matrix if not computed previously
	// 		// if (results.find(key(set1Id, set2Id)) == results.end()) {
	// 			GlobalAMatrix* A = new GlobalAMatrix(&set1Tokens, &set2Tokens, theta, dl, k);
	// 			A->computeAlignment(k);
	// 			// A->printAMatrix();
	// 			// results[key(set1Id, set2Id)] = A;
	// 			numberOfGraphMatchingComputed += A->get_matrix_size();
	// 			numberOfZeroEntries += A->zeroCells();
	// 		// }
	// 	}
	// 	// cout << "Done with: " << counter << " / " << text1Sets.size() << endl;
	// 	counter += 1;
	// }

	// // Global Cost Matrix with Pruning
	// for (int set1Id : text1Sets) {
	// 	vector<int> set1Tokens = sets[set1Id];
	// 	for (int set2Id : text2Sets) {
	// 		vector<int> set2Tokens = sets[set2Id];
	// 		// compute the alignment matrix if not computed previously
	// 		// if (results.find(key(set1Id, set2Id)) == results.end()) {
	// 			// GlobalAMatrix* A = new GlobalAMatrix(&set1Tokens, &set2Tokens, theta, dl);
	// 			GlobalAMatrix* A = new GlobalAMatrix(&set1Tokens, &set2Tokens, theta, dl, k);
	// 			// A->computeAlignmentWithPruning_precomputed(k, 2);
    //             A->computeAlignmentWithPruning(k, 1);
	// 			// A->printAMatrix();
	// 			// results[key(set1Id, set2Id)] = A;
	// 			numberOfGraphMatchingComputed += A->get_matrix_size();
	// 			numberOfZeroEntries += A->zeroCells();
	// 			pruned_cells += A->prunedCount();
	// 			pair<int, int> count_est = A->getEstimateCounts();
	// 			numCellsAboveThresoldEstimate += count_est.first;
	// 			numCellsAboveThresold += count_est.second;
	// 		// }
	// 	}
	// 	// cout << "Done with: " << counter << " / " << text1Sets.size() << endl;
	// 	counter += 1;
	// }

	// // Candidate Generation 
	cout << "KWidthWindows: " << kWidthWindows.size() << endl;
	int set1ID = *text1Sets.begin();
	int set2ID = *text2Sets.begin();
	vector<int>& set1Tokens = sets[set1ID];
	vector<int>& set2Tokens = sets[set2ID];
	IndexStructures* indexStruct = new IndexStructures(&set1Tokens, &set2Tokens, dl, k, theta);
	indexStruct->reportStats();
	vector<vector<int>> &set1Windows = kWidthWindows[set1ID];
	for (vector<int> window : set1Windows) {
		// vector<vector<int>> candidates = indexStruct->findCandidateWindows(window);
		set<pair<int, vector<int>>, sliding_window_ordering> candidates = indexStruct->getCandidates(window);
		cout << candidates.size() << endl;
	}	



	bs_search_end = std::chrono::high_resolution_clock::now();
	bs_search_elapsed = bs_search_end - bs_search_start;
	bs_search_time = bs_search_elapsed.count();
	cout << "Main Loop Time: " << bs_search_time << endl;
	cout << "Number of Graph Matching Computed: " << numberOfGraphMatchingComputed << endl;
	cout << "Number of Zero Entries Cells: " << numberOfZeroEntries << endl;
	cout << "Number of Pruned Cells: " << pruned_cells << endl;
	cout << "Number of Estimated Cells Above Threshold: " << numCellsAboveThresoldEstimate << endl;
	cout << "Number of Cells Above Threshold: " << numCellsAboveThresold << endl;
}

/**
 * Main Function : Entry Point
*/
int main(int argc, char const *argv[]) {

  	// arguments: text1_location, text2_location, window_width, threshold, result_file, data_file location
	string text1_location = argv[1];
	string text2_location = argv[2];
	int k = stoi(argv[3]);
	double theta = stod(argv[4]);
	string result_file = argv[5];
	string data_file = argv[6];	

	double envtime = 0.0;
	double invertedIndexSize = 0;
	double dataloader_time = 0.0;
	double algo_time = 0.0;

	std::chrono::time_point<std::chrono::high_resolution_clock> envstart, envend, dl_start, dl_end, algo_start, algo_end;
	std::chrono::duration<double> envlapsed, dl_elapsed, algo_elapsed;
	envstart = std::chrono::high_resolution_clock::now();
	Environment *env = new Environment(text1_location, text2_location);
	envend = std::chrono::high_resolution_clock::now();
	envlapsed = envend - envstart;
	envtime = envlapsed.count();

	

	vector<set<int>> invertedIndex = env->getInvertedIndex();
	for (set<int> f : invertedIndex) {
		invertedIndexSize += f.size();
	}
	invertedIndexSize += invertedIndex.size();
	cout << "Inverted Index Size: " << invertedIndexSize << endl;
	cout << "Words: " << env->getWordSet().size() << endl;

	dl_start = std::chrono::high_resolution_clock::now();
	DataLoader *loader = new DataLoader(data_file, env);
	dl_end = std::chrono::high_resolution_clock::now();
	dl_elapsed = dl_end - dl_start;
	dataloader_time = dl_elapsed.count();

	algo_start = std::chrono::high_resolution_clock::now();
	baseline(env, loader, k, theta);
	algo_end = std::chrono::high_resolution_clock::now();
	algo_elapsed = algo_end - algo_start;
	algo_time = algo_elapsed.count();

	cout << "Env Time: " << envtime << endl;
	cout << "Dataloader Time: " << dataloader_time << endl;
	cout << "Algorithm Time: " << algo_time << endl;

	return 0;
}