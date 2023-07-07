/**

    Implement the baseline:
    - read the data from the txt file 
    - read the vectors for each token
    - solve the sliding window bipartite graph matching 
      to compute the alignment matrix 

	@todo:
		- implement the Faiss Index for efficient search [done]
		- update the valid edges [done]
		- implement the AMatrix->computeAlignment()
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
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/IndexFlat.h>
#include <faiss/utils/distances.h>
#include <thread>
#include <future>
#include <omp.h>
#include <regex>
#include "thread_pool.hpp"
#include "../modules/timing.h"
#include "./hungarian-algorithm-cpp-master/Hungarian.h"

std::mutex gmtx;
using namespace std;
namespace fs = std::filesystem;
using idx_t = faiss::Index::idx_t;
inline size_t key(int i,int j) {return (size_t) i << 32 | (unsigned int) j;} // concat unsigned int with two integer set id as an edge's integer id

vector<set<int>> slidingWindows(vector<int>& tokens, int k) {
	vector<set<int>> windows;
	if (tokens.empty() || k <= 0 || k > tokens.size()) {
		return windows;
	}

	auto it = tokens.begin();
	set<int> currentWindow(it, next(it, k));
	windows.push_back(currentWindow);

	while (next(it, k) != tokens.end()) {
		currentWindow.erase(*it);
		currentWindow.insert(*next(it, k));
		windows.push_back(currentWindow);
		++it;
	}
	return windows;
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

		std::unordered_map<int, vector<set<int>>> computeSlidingWindows(int windowWidth) {
			std::unordered_map<int, vector<set<int>>> windows;
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
						float* r = new float[300];
						r = embedding.data();
						faiss::fvec_renorm_L2(300, 1, r);
						vector<float> vr(r, r + 300);
						vectors[token_id] = vr;
						dictionary.push_back(token_id);
					}
				}
			}
			file.close();
		}

		vector<float> get_vector(int id){
			return vectors[id];
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
			double sim = double(dot / (sqrt(denom_a) * sqrt(denom_b)));
			cache[key(token1_ID, token2_ID)] = sim;
			return sim;
		}
};

/**
 * CPU Faiss Index Wrapper
*/
class FaissIndexCPU {
	private:
		faiss::IndexFlatIP *index;
		std::unordered_map<int, vector<float>> normalized;
		vector<int> dictionary;
	
	public:
		FaissIndexCPU(string path, DataLoader *dl, std::unordered_set<int> validSet) {
			string indexPath = path + "faiss.index";
			int d = 300;

			// vector<vector<float>> vectors = db->get_valid_vectors(validSet);
			vector<vector<float>> vectors = dl->getAllVectors();
			cout << "Vector Size: " << vectors[0].size() << endl;
			// dictionary = db->get_dictionary();

			index = new faiss::IndexFlatIP(d);

			int nb = vectors.size();
			float *xb = new float[d * nb];
			for (int i = 0; i < nb; i++) {
				for (int j = 0; j < d; j++) {
					xb[d * i + j] = vectors[i][j];
				}
				xb[d * i] += i / 1000.;
			}

			faiss::fvec_renorm_L2(d, nb, xb);

			cout << "Normalized vector storage check" << endl;
			index->add(nb, xb);
			cout << "Index Trained: " << index->is_trained << endl;
			cout << "Ntotal: " << index->ntotal << endl;
		}

		std::tuple<vector<idx_t>, vector<float>> kNNSearch(int nq, vector<float> vxq, int k) {
			int d = 300;
			float* xq = vxq.data();
			for (int i = 0; i < nq; i++) {
				xq[d * i] += i / 1000.;
			}

			// search xq
			idx_t *I = new idx_t[k * nq];
			float *D = new float[k * nq];

			index->search(nq, xq, k, D, I);

			vector<idx_t> vI(I, I + k * nq);
			vector<float> vD(D, D + k * nq);

			delete [] I;
			delete [] D;
			return {vI, vD};
		}

		std::tuple<vector<vector<idx_t>>, vector<vector<float>>> rangeSearch(int nq, vector<float> vxq, float radius) {
			int d = 300;
			float* xq = vxq.data();

			faiss::RangeSearchResult* result = new faiss::RangeSearchResult(nq);
			
			index->range_search(nq, xq, radius, result);

			vector<vector<idx_t>> vI(nq);
			vector<vector<float>> vD(nq);

			for (int i = 0; i < nq; i++) {
				size_t start = result->lims[i];
				size_t end = result->lims[i+1];
				size_t count = end - start;

				for (size_t j = 0; j < count; j++) {
					vI[i].push_back(result->labels[start + j]);
					vD[i].push_back(result->distances[start + j]);
				}
			}

			return {vI, vD};
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
		ValidMatrix(set<int> query_set, set<int> target_set, std::unordered_map<size_t, double> validedge) {
			int i = 0;
			set<int> query_set_pruned;
			set<int> target_set_pruned;
			matching = min(query_set.size(), target_set.size());
			for(set<int>::iterator itq = query_set.begin(); itq != query_set.end(); ++itq) {
				vector<double> temp;
				int qword = *itq;
				for(set<int>::iterator itt = target_set.begin(); itt != target_set.end(); ++itt) {
					int tword = *itt;
					if (validedge.find(key(qword, tword)) != validedge.end()) {
						query_set_pruned.insert(qword);
						target_set_pruned.insert(tword);
					} else {
						if (qword == tword) {
							query_set_pruned.insert(qword);
							target_set_pruned.insert(tword);
						}
					}// bucket upperbound not considering this
				}
			}
			realsize = query_set_pruned.size() * target_set_pruned.size();
			// vector<int> query_tokens;
			for(set<int>::iterator itq = query_set_pruned.begin(); itq != query_set_pruned.end(); ++itq) {
				vector<double> temp;
				vector<pair<int, int>> temp_tokens;
				int qword = *itq;
				for(set<int>::iterator itt = target_set_pruned.begin(); itt != target_set_pruned.end(); ++itt) {
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

		ValidMatrix(set<int> query_set, set<int> target_set, DataLoader *dl) {
			int i = 0;
			matching = min(query_set.size(), target_set.size());
			for(set<int>::iterator itq = query_set.begin(); itq != query_set.end(); ++itq) {
				vector<double> temp;
				vector<pair<int, int>> temp_tokens;
				int qword = *itq;
				for(set<int>::iterator itt = target_set.begin(); itt != target_set.end(); ++itt) {
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
		double* alignmentMatrix;
		vector<int> *set1Tokens;
		vector<int> *set2Tokens;
		double theta;
		DataLoader *dl;
		int Ws;
		int Wt;
		int zero_entries = 0;

	public:
		GlobalAMatrix(vector<int>* _set1Tokens, vector<int>* _set2Tokens, double _theta, DataLoader* _dl) {
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
		}

		void computeAlignment(int k) {
			int rows = costMatrix.size();
			int cols = costMatrix[0].size();
			Ws = set1Tokens->size() - k + 1;
			Wt = set2Tokens->size() - k + 1;
			alignmentMatrix = new double[Ws * Wt];
			for (int i = 0; i <= rows - k; i++) {
				for (int j = 0; j <= cols - k; j++) {
					vector<vector<double>> currWindow(k, vector<double>(k));
					for (int m = 0; m < k; m++) {
						for (int n = 0; n < k; n++) {
							currWindow[m][n] = costMatrix[i + m][j + n];
						}
					}
					HungarianAlgorithm HungAlgo;
					vector<int> assignment;
					double sim;
					if (currWindow.size() == 0) {
						sim = 0.0;
					} else {
						double cost = HungAlgo.Solve(currWindow, assignment);
						sim = -cost/k;
					}
					if (sim >= theta) {
						alignmentMatrix[i + j * Ws] = sim;
					} else {
						alignmentMatrix[i + j * Ws] = 0.0;
						zero_entries += 1;
					}
					
				}
			}
		}

		int get_matrix_size() {
			return Ws * Wt;
		}

		int zeroCells() {
			return zero_entries;
		}

};

class AMatrix {
	private:
		int width;
		int height;
		double* data; // double* matrix 1-D representation of a matrix: matrix[i + j*width] = matrix[i][j].. matrix[width*height]... delete [] matrix;
		vector<set<int>> *set1Windows;
		vector<set<int>> *set2Windows;
		unordered_map<size_t, double> validEdges;
		double theta;
		int zero_entries = 0;
		DataLoader *dl;
	
	public:
		AMatrix(vector<set<int>>* Ws, vector<set<int>>* Wt, unordered_map<size_t, double> validedges, double threshold, DataLoader *_dl) {
			width = Ws->size();
			height = Wt->size();
			set1Windows = Ws;
			set2Windows = Wt;
			validEdges = validedges;
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
					set<int> set1_tokens = set1Windows->at(i);
					set<int> set2_tokens = set2Windows->at(j);
					if (set1_tokens.size() <= 10) {
						m = new ValidMatrix(set1_tokens, set2_tokens, dl);
					} else {
						m = new ValidMatrix(set1_tokens, set2_tokens, validEdges);
					}
					// cout << "here" << endl;
					sim = m->solveQ(set1_tokens.size());
					// sim = 1.0;
					// cout << sim << endl;
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


void baseline(Environment *env, DataLoader *dl, FaissIndexCPU *faissIndex, int k, double theta) {

	std::unordered_map<int, vector<int>> sets = env->getSets(); // all sets stored as key: set integer id, value: set data (int token integer ids)
	std::unordered_set<int> wordSet = env->getWordSet(); // all unique tokens in copora
	vector<set<int>> invertedIndex = env->getInvertedIndex(); // inverted index that returns all sets containing given token
	vector<int> dictionary = dl->getDictionary(); // vectors database instance
	set<int> text1Sets = env->getText1SetIds();
	set<int> text2Sets = env->getText2SetIds();
	std::mutex gmtx_internal;

	int numberOfGraphMatchingComputed = 0;
	int numberOfZeroEntries = 0;
	
	std::unordered_map<size_t, AMatrix*> results; // key(text1SetId, text2SetId) --> AlignmentMatrix
	std::chrono::time_point<std::chrono::high_resolution_clock> sliding_window_start, sliding_window_end;
	std::chrono::duration<double> sliding_window_elapsed;
	double slidingWindowTime = 0.0;
	sliding_window_start = std::chrono::high_resolution_clock::now();
	std::unordered_map<int, vector<set<int>>> kWidthWindows = env->computeSlidingWindows(k); // setID --> sliding windows
	sliding_window_end = std::chrono::high_resolution_clock::now();
	sliding_window_elapsed = sliding_window_end - sliding_window_start;
	slidingWindowTime = sliding_window_elapsed.count();
	cout << "Sliding Window Computation time: " << slidingWindowTime << endl;
 	std::unordered_map<size_t, double> validedges;
	// for all tokens between the two texts, do a faiss similarity search and cache the edges
	int nq = 0;
	int i = 0;
	vector<float> vxq;
	for (auto it = wordSet.begin(); it != wordSet.end(); it++) {
		int tq = *it;
		// handle out of dictionary words
		if (std::find(dictionary.begin(), dictionary.end(), tq) == dictionary.end()) {
			i += 1;
			validedges[key(tq, tq)] = 1.0;
		} else {
			// vector<float> vec = db->get_normalized_vector(tq);
			vector<float> vec = dl->get_vector(tq);
			if (vec.size() == 0) {
				cerr << "Vector should not be empty" << endl;
			}
			vxq.insert(vxq.end(), vec.begin(), vec.end());
			nq += 1;
		}
	}

	cout << "Out of dictionary words: " << i << endl;
	// get the k nearest neighbours for each word, here set k = nq. 
	// @todo: change to range search with radius = theta
	std::chrono::time_point<std::chrono::high_resolution_clock> faiss_search_start, faiss_search_end;
	std::chrono::duration<double> faiss_search_elapsed;
	double faiss_search_time = 0.0;
	faiss_search_start = std::chrono::high_resolution_clock::now();
	tuple<vector<idx_t>, vector<float>> rt = faissIndex->kNNSearch(nq, vxq, nq);
	faiss_search_end = std::chrono::high_resolution_clock::now();
	faiss_search_elapsed = faiss_search_end - faiss_search_start;
	faiss_search_time = faiss_search_elapsed.count();
	cout << "Faiss Search Time: " << faiss_search_time << endl;
	vector<idx_t> I = std::get<0>(rt);
	vector<float> D = std::get<1>(rt);
	cout << "size of I: " << I.size() << endl;
	int cur = 0;
	for (vector<idx_t>::iterator it = I.begin(); it != I.end(); it++) {
		int tq_cur = I[cur - (cur % nq)];
		int tq = dictionary[tq_cur];
		int word = dictionary[*it];
		float fsim = D[cur];
		// cout << "fsim: " << fsim << endl;
		double sim = static_cast<double>(fsim);
		// cout << "sim: " << sim << endl;
		if (sim > 0.0) {
			// cout << "here" << endl;
			validedges[key(tq, word)] = sim;
		}
		cur += 1;
	}

	cout << "Size of valid edges: " << validedges.size() << endl;

	// for each set in text1Sets, we compute the k-width window and compute the alignment matrix
	std::chrono::time_point<std::chrono::high_resolution_clock> bs_search_start, bs_search_end;
	std::chrono::duration<double> bs_search_elapsed;
	double bs_search_time = 0.0;
	bs_search_start = std::chrono::high_resolution_clock::now();
	int counter = 1;
	// for (int set1Id : text1Sets) {
	// 	vector<set<int>> set1Windows = kWidthWindows[set1Id];
	// 	for (int set2Id : text2Sets) {
	// 		vector<set<int>> set2Windows = kWidthWindows[set2Id];
	// 		// compute the alignment matrix if not computed previously
	// 		if (results.find(key(set1Id, set2Id)) == results.end()) {
	// 			AMatrix *A = new AMatrix(&set1Windows, &set2Windows, validedges, theta, dl);
	// 			A->computeAlignment();
	// 			results[key(set1Id, set2Id)] = A;
	// 			numberOfGraphMatchingComputed += A->get_matrix_size();
	// 			numberOfZeroEntries += A->zeroCells();
	// 		}
	// 	}
	// 	// cout << "Done with: " << counter << " / " << text1Sets.size() << endl;
	// 	counter += 1;
	// }
	for (int set1Id : text1Sets) {
		vector<int> set1Tokens = sets[set1Id];
		for (int set2Id : text2Sets) {
			vector<int> set2Tokens = sets[set2Id];
			// compute the alignment matrix if not computed previously
			if (results.find(key(set1Id, set2Id)) == results.end()) {
				GlobalAMatrix* A = new GlobalAMatrix(&set1Tokens, &set2Tokens, theta, dl);
				A->computeAlignment(k);
				// results[key(set1Id, set2Id)] = A;
				numberOfGraphMatchingComputed += A->get_matrix_size();
				numberOfZeroEntries += A->zeroCells();
			}
		}
		// cout << "Done with: " << counter << " / " << text1Sets.size() << endl;
		counter += 1;
	}
	bs_search_end = std::chrono::high_resolution_clock::now();
	bs_search_elapsed = bs_search_end - bs_search_start;
	bs_search_time = bs_search_elapsed.count();
	cout << "Main Loop Time: " << bs_search_time << endl;
	cout << "Number of Graph Matching Computed: " << numberOfGraphMatchingComputed << endl;
	cout << "Number of Zero Entries Cells: " << numberOfZeroEntries << endl;
}

/**
 * Main Function : Entry Point
*/
int main(int argc, char const *argv[]) {

  	// arguments: text1_location, text2_location, window_width, threshold, result_file, data_file location
	string text1_location = argv[1];
	string text2_location = argv[2];
	int k = stoi(argv[3]);
	// cout << "Window Width: " << k << endl;
	double theta = stod(argv[4]);
	string result_file = argv[5];
	// string database_path = argv[6];
	string data_file = argv[6];	

	double envtime = 0.0;
	double invertedIndexSize = 0;
	double faiss_time = 0.0;
	double dataloader_time = 0.0;
	double algo_time = 0.0;

	std::chrono::time_point<std::chrono::high_resolution_clock> envstart, envend, faiss_start, faiss_end, dl_start, dl_end, algo_start, algo_end;
	std::chrono::duration<double> envlapsed, faiss_elapsed, dl_elapsed, algo_elapsed;
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
	// Database *db = new Database(database_path, env);
	cout << "Words: " << env->getWordSet().size() << endl;

	dl_start = std::chrono::high_resolution_clock::now();
	DataLoader *loader = new DataLoader(data_file, env);
	dl_end = std::chrono::high_resolution_clock::now();
	dl_elapsed = dl_end - dl_start;
	dataloader_time = dl_elapsed.count();
	// vector<float> test_vector = loader->get_vector(20);
	// cout << "Words: " << loader->getWords().size() << endl;
	// cout << "Test Vector Size: " << test_vector.size() << endl;

	faiss_start = std::chrono::high_resolution_clock::now();
	FaissIndexCPU *faissIndex = new FaissIndexCPU("./", loader, env->getWordSet());
	faiss_end = std::chrono::high_resolution_clock::now();
	faiss_elapsed = faiss_end - faiss_start;
	faiss_time = faiss_elapsed.count();

	algo_start = std::chrono::high_resolution_clock::now();
	baseline(env, loader, faissIndex, k, theta);
	algo_end = std::chrono::high_resolution_clock::now();
	algo_elapsed = algo_end - algo_start;
	algo_time = algo_elapsed.count();

	cout << "Env Time: " << envtime << endl;
	cout << "Dataloader Time: " << dataloader_time << endl;
	cout << "FaissIndex Time: " << faiss_time << endl;
	cout << "Algorithm Time: " << algo_time << endl;

	return 0;
}