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

vector<set<int>> slidingWindows(set<int> tokens, int k) {
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
		std::unordered_map<int, set<int>> sets;
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
		// int k = 0;
		
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
								sets[sid].insert(id);
								std::set<int> nset = {sid};
								invertedIndex.push_back(nset);
								id += 1;
							} else {
								sets[sid].insert(word2int[line]);
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
				string f = text1_files[i];
				set2int[f] = sid;
				int2set[sid] = f;
				text2sets.insert(sid);
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
								sets[sid].insert(id);
								std::set<int> nset = {sid};
								invertedIndex.push_back(nset);
								id += 1;
							} else {
								sets[sid].insert(word2int[line]);
								invertedIndex[word2int[line]].insert(sid);
							}
						}
					}
				}
				sid += 1;
			}

			text2_average_cardinality = (tokens - text1_tokens) / (sets.size() - text1_sets);

		}

		std::unordered_map<int, set<int>> getSets() {
			return sets;
		}

		std::unordered_map<int, vector<set<int>>> computeSlidingWindows(int windowWidth) {
			std::unordered_map<int, vector<set<int>>> windows;
			for (auto i = sets.begin(); i != sets.end(); i++) {
				windows[i->first] = slidingWindows(i->second, windowWidth);
			}
			return windows;
		}

		std::unordered_set<int> getWordSet() {
			return wordSet;
		}

		vector<set<int>> getInvertedIndex() {
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

		set<int> getText1SetIds() {
			return text1sets;
		}

		set<int> getText2SetIds() {
			return text2sets;
		}
};


/**
 * Database
*/
class Database {
  private:
    sqlite3 *db; // database instance that stores the vectors
    sqlite3 *dest; // databse location
    std::unordered_map<size_t, double> cache;
    vector<int> dictionary; // array of unique tokens, index is it's integer id
    Environment *env;

    // convert mysql entry from bytes to word vector
		vector<float> bytes_to_wv(const unsigned char *data, size_t dimension) {
			vector<float> result;
			for (size_t i = 0; i < dimension * 4; i = i+4) {
				float f;
				unsigned char b[] = {data[i], data[i+1], data[i+2], data[i+3]};
				memcpy(&f, b, sizeof(f));
				// cout << f << endl;
				result.push_back(f);
			}
			return result;
		}

    // tool for investigating bytes
		void print_bytes(ostream& out, const unsigned char *data, size_t dataLen, bool format = true) {
			out << setfill('0');
			for(size_t i = 0; i < dataLen; ++i) {
				out << hex << setw(2) << (int)data[i];
				if (format) {
					out << (((i + 1) % 16 == 0) ? "\n" : " ");
				}
			}
			out << endl;
		}

		// tool for investigating vectors
		void print_vector(vector<float> v) {
			for (int i = 0; i < v.size(); ++i){
				cout << v[i] << ", ";
			}
			cout << "\n" << endl;
		}

		// cache for cosine similarity, not in use *
		bool cache_lookup(int qtoken, int ttoken) {
			return cache.find(key(qtoken, ttoken)) != cache.end();
		}
  
  public:
    Database(string path, Environment *e) {
      env = e;
      int rc = sqlite3_open(path.c_str(), &dest);
      if (rc) {
        cout << "Cannot open database: " << sqlite3_errmsg(dest) << endl;
      } else {
        cout << "Successfully opened sqlite3 database" << endl;
      }

      // sqlite3 in-memory for performance
      rc = sqlite3_open(":memory:", &db);
      if (rc) {
				cout << "Cannot open in-memory database:  " << sqlite3_errmsg(db) << endl; 
				exit(0);
			} else {
				cout << "Successfully opened in-memory database" << endl;
			}

      sqlite3_backup *pBackup;
			pBackup = sqlite3_backup_init(db, "main", dest, "main");
			if( pBackup ){
				sqlite3_backup_step(pBackup, -1);
				sqlite3_backup_finish(pBackup);
			}
			rc = sqlite3_errcode(db);
			if (rc) {
				cout << "Cannot copy database:  " << sqlite3_errmsg(db) << endl; 
				exit(0);
			} else {
				cout << "Successfully copied to memory" << endl;
			}
			sqlite3_close(dest);
    }

    void terminate() {
			sqlite3_close(db);
			cout << "Successfully terminated sqlite3 database" << endl;
		}

		void clear_cache() {
			cache.clear();
		}

    bool isnum(string s) {
			size_t pos;
			string t = ".";
			while ((pos = s.find(t)) != string::npos) {
				s.erase(pos, t.length());
			}
			for (string::iterator it = s.begin(); it != s.end(); ++it) {
				if (!isdigit(*it)) {
					return false;
				}
			}
			return true;
		}

		// cosine similarity helper function
		double cosine_similarity(vector<float> A, vector<float> B, int vlength){
			float dot = 0.0, denom_a = 0.0, denom_b = 0.0 ;
			 for(int i = 0; i < vlength; ++i) {
				dot += A[i] * B[i] ;
				denom_a += A[i] * A[i] ;
				denom_b += B[i] * B[i] ;
			}
			return double(dot / (sqrt(denom_a) * sqrt(denom_b))) ;
		}

    // calculate cosine similarity between a pair of words
		float calculate_similarity(int src, int tgt) {
			int rc;

			string srcstring = env->toWord(src);
			string tgtstring = env->toWord(tgt);

			if (src == tgt) {
				return 1.0;
			}
			if (isnum(srcstring) || isnum(tgtstring)) {
				return 0.0;
			}
			if (cache_lookup(src, tgt)) {
				return cache[key(src, tgt)];
			}

			string querybase = "SELECT vec FROM en_vectors WHERE word=?;";
			

			vector<float> vector1;
			vector<float> vector2;

			sqlite3_stmt *stmt = NULL;

			bool s = false;
			bool t = false;

			rc = sqlite3_prepare_v2(db, querybase.c_str(), -1, &stmt, NULL);
			rc = sqlite3_bind_text(stmt, 1, srcstring.c_str(), -1, SQLITE_TRANSIENT);
			if (rc != SQLITE_OK) {
				cout << "SELECT failed: " << sqlite3_errmsg(db) << endl;
			}
			while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
				unsigned char *bytes = (unsigned char*)sqlite3_column_blob(stmt, 0);
				vector1 = bytes_to_wv(bytes, 300);
				if (vector1.size() == 300) {
					s = true;
				}
			}
			sqlite3_finalize(stmt);

			rc = sqlite3_prepare_v2(db, querybase.c_str(), -1, &stmt, NULL);
			rc = sqlite3_bind_text(stmt, 1, tgtstring.c_str(), -1, SQLITE_TRANSIENT);
			if (rc != SQLITE_OK) {
				cout << "SELECT failed: " << sqlite3_errmsg(db) << endl;
			}
			while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
				unsigned char *bytes = (unsigned char*)sqlite3_column_blob(stmt, 0);
				vector2 = bytes_to_wv(bytes, 300);
				if (vector2.size() == 300) {
					t = true;
				}
			}
			sqlite3_finalize(stmt);

			if (!(s && t)) {
				return 0.0;
			} 
			double sim = cosine_similarity(vector1, vector2, 300);
			cache[key(src, tgt)] = sim;
			cache[key(tgt, src)] = sim;
			return sim;
		}
    
    // get normalized vector with token's integer id
		vector<float> get_normalized_vector(int tokenid){
			vector<float> result;
			string token = env->toWord(tokenid);
			// cout << token << endl;
			int rc;
			stringstream ss;
			ss << "SELECT vec FROM en_vectors WHERE word=\"" << token << "\";";
			string query = ss.str();
			string querybase = "SELECT vec FROM en_vectors WHERE word=?;";
			sqlite3_stmt *stmt = NULL;
			rc = sqlite3_prepare(db, querybase.c_str(), -1, &stmt, NULL);
			rc = sqlite3_bind_text(stmt, 1, token.c_str(), -1, SQLITE_TRANSIENT);
			if (rc != SQLITE_OK) {
				cout << query << endl;
				cout << "SELECT failed: " << sqlite3_errmsg(db) << endl;
			} else {
				while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
					unsigned char *bytes = (unsigned char*)sqlite3_column_blob(stmt, 0);
					int size = sqlite3_column_bytes(stmt, 0);
					result = bytes_to_wv(bytes, size / sizeof(float));
				}
			}
			
			float* r = new float[300];
			r = result.data();
			// faiss::fvec_renorm_L2(300, 1, r);
			vector<float> vr(r, r + 300);
			return vr;
			return result;
		}

		// get vectors of words in valid set
		vector<vector<float>> get_valid_vectors(std::unordered_set<int> validset){
			vector<vector<float>> result;
			int rc;
			stringstream ss;
			ss << "SELECT word, vec FROM en_vectors;";
			string query = ss.str();
			sqlite3_stmt *stmt = NULL;
			rc = sqlite3_prepare(db, query.c_str(), -1, &stmt, NULL);
			if (rc != SQLITE_OK) {
				cout << "SELECT failed: " << sqlite3_errmsg(db) << endl;
			}
			dictionary.clear();
			
			while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
				const char *word = (char*)sqlite3_column_text(stmt, 0);
				unsigned char *bytes = (unsigned char*)sqlite3_column_blob(stmt, 1);
				string str_t(word);
				// cout << str_t << endl;
				int wid = env->toInt(str_t);
				// cout << wid << endl;
				bool is_in = validset.find(wid) != validset.end();
				if (is_in) {
					dictionary.push_back(wid);
					vector<float> vector_t = bytes_to_wv(bytes, 300);
					result.push_back(vector_t);
				}
			}
			return result;
		}

		vector<int> get_dictionary(){
			return dictionary;
		}
		void clear_dictionary(){
			dictionary.clear();
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
					// cout << "Token Id: " << token_id << endl;
					// cout << "Embedding Size: " << embedding.size() << endl;
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

			// for (int i = 0; i < nb; i++) {
			// 	vector<float> vec;
			// 	for (int j = 0; j < d; j++) {
			// 		vec.push_back(xb[d * i + j]);
			// 	}
			// }

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
						// temp.push_back(validedge[key(qword, tword)]);
					} else {
						if (qword == tword) {
							temp.push_back(-1.0);
							// temp.push_back(1.0);
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

		const vector<vector<double>> reveal() {
			return M;
		}
		int getrealsize() {
			return realsize;
		}

		// solve matching with |Query| cardinality as base
		// double solveQ(int q, std::set<pair<double, int>, cmp_increasing> *L_, std::mutex *mtx, double *etm, bool isEarly = true){
		// 	HungarianAlgorithm HungAlgo;
		// 	vector<int> assignment;
		// 	double cost = HungAlgo.Solve(M, assignment, non_exact_token_indices, q, L_, mtx, etm, isEarly);
		// 	double overlap = 0.0;
		// 	assignment_internal = assignment;
		// 	if (matching == 0) {
		// 		return 0.0;
		// 	}
		// 	return -cost/q;
		// }

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
	
	public:
		AMatrix(vector<set<int>>* Ws, vector<set<int>>* Wt, unordered_map<size_t, double> validedges, double threshold) {
			width = Ws->size();
			height = Wt->size();
			set1Windows = Ws;
			set2Windows = Wt;
			validEdges = validedges;
			theta = threshold;
			data = new double[width * height];
		}

		void computeAlignment() {
			for (int i = 0; i < set1Windows->size(); ++i) {
				for (int j = 0; j < set2Windows->size(); ++j) {
					set<int> set1_tokens = set1Windows->at(i);
					set<int> set2_tokens = set2Windows->at(j);

					ValidMatrix *m = new ValidMatrix(set1_tokens, set2_tokens, validEdges);
					// cout << "here" << endl;
					// double sim = m->solveQ(set1_tokens.size());
					double sim = 1.0;
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

	std::unordered_map<int, set<int>> sets = env->getSets(); // all sets stored as key: set integer id, value: set data (int token integer ids)
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
	int counter = 1;
	for (int set1Id : text1Sets) {
		vector<set<int>> set1Windows = kWidthWindows[set1Id];
		for (int set2Id : text2Sets) {
			vector<set<int>> set2Windows = kWidthWindows[set2Id];
			// compute the alignment matrix if not computed previously
			if (results.find(key(set1Id, set2Id)) == results.end()) {
				AMatrix *A = new AMatrix(&set1Windows, &set2Windows, validedges, theta);
				A->computeAlignment();
				results[key(set1Id, set2Id)] = A;
				numberOfGraphMatchingComputed += A->get_matrix_size();
				numberOfZeroEntries += A->zeroCells();
			}
		}
		cout << "Done with: " << counter << " / " << text1Sets.size() << endl;
		counter += 1;
	}

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