/**

    Implement the baseline:
    - read the data from the txt file 
    - read the vectors for each token
    - solve the sliding window bipartite graph matching 
      to compute the alignment matrix 

	@todo:
		- implement the Faiss Index for efficient search [done]
		- update the valid edges
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
	if (tokens.empty() || k<= 0 || k > tokens.size()) {
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
		int k;
		
	public:
		Environment(string text1location, string text2location, int windowWidth) {
			cout << "creating environment with lakes: " << text1location << ", " << text2location << endl;
			int k = windowWidth;
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

		std::unordered_map<int, vector<set<int>>> computeSlidingWindows() {
			std::unordered_map<int, vector<set<int>>> windows;
			for (auto i = sets.begin(); i != sets.end(); i++) {
				windows[i->first] = slidingWindows(i->second, k);
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
				memcpy(&f, &b, sizeof(f));
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
					result = bytes_to_wv(bytes, 300);
				}
			}
			
			float* r = new float[300];
			r = result.data();
			faiss::fvec_renorm_L2(300, 1, r);
			vector<float> vr(r, r + 300);
			return vr;
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
 * CPU Faiss Index Wrapper
*/
class FaissIndexCPU {
	private:
		faiss::IndexFlatL2 *index;
		std::unordered_map<int, vector<float>> normalized;
		vector<int> dictionary;
	
	public:
		FaissIndexCPU(string path, Database *db, std::unordered_set<int> validSet) {
			string indexPath = path + "faiss.index";
			int d = 300;

			vector<vector<float>> vectors = db->get_valid_vectors(validSet);
			dictionary = db->get_dictionary();

			index = new faiss::IndexFlatL2(d);

			int nb = vectors.size();
			float *xb = new float[d * nb];
			for (int i = 0; i < nb; i++) {
				for (int j = 0; j < d; j++) {
					xb[d * i + j] = vectors[i][j];
				}
			}

			faiss::fvec_renorm_L2(d, nb, xb);

			for (int i = 0; i < nb; i++) {
				vector<float> vec;
				for (int j = 0; j < d; j++) {
					vec.push_back(xb[d * i + j]);
				}
			}

			cout << "Normalized vector storage check" << endl;
			index->add(nb, xb);
		}

		std::tuple<vector<idx_t>, vector<float>> kNNSearch(int nq, vector<float> vxq, int k) {
			int d = 300;
			float* xq = vxq.data();

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

class AMatrix {
	private:
		int width;
		int height;
		double* data; // double* matrix 1-D representation of a matrix: matrix[i + j*width] = matrix[i][j].. matrix[width*height]... delete [] matrix;
		vector<set<int>> set1Windows;
		vector<set<int>> set2Windows;
		unordered_map<size_t, double> validEdges;
		double theta;
	
	public:
		AMatrix(vector<set<int>> Ws, vector<set<int>> Wt, unordered_map<size_t, double> validedges, double threshold) {
			width = Ws.size();
			height = Wt.size();
			set1Windows = Ws;
			set2Windows = Wt;
			validEdges = validedges;
			theta = threshold;
			data = new double[width * height];
		}

		void computeAlignment() {

		}

		double getMatrixValue(int row, int col) {
			return data[row + col * width];
		}

		void setMatrixValue(int row, int col, double value) {
			data[row + col * width] = value;
		}

};


void baseline(Environment *env, Database *db, int k, double theta) {

	std::unordered_map<int, set<int>> sets = env->getSets(); // all sets stored as key: set integer id, value: set data (int token integer ids)
	std::unordered_set<int> wordSet = env->getWordSet(); // all unique tokens in copora
	vector<set<int>> invertedIndex = env->getInvertedIndex(); // inverted index that returns all sets containing given token
	vector<int> dictionary = db->get_dictionary(); // vectors database instance
	set<int> text1Sets = env->getText1SetIds();
	set<int> text2Sets = env->getText2SetIds();
	std::mutex gmtx_internal;

	double numberOfGraphMatchingComputed = 0;
	
	std::unordered_map<size_t, AMatrix*> results; // key(text1SetId, text2SetId) --> AlignmentMatrix
	std::unordered_map<int, vector<set<int>>> kWidthWindows = env->computeSlidingWindows(); // setID --> sliding windows

	// @todo: populate valid edges, by computing the pairwise cosine similarity between all tokens of the environment
	std::unordered_map<size_t, double> validedges;

	// for each set in text1Sets, we compute the k-width window and compute the alignment matrix
	for (int set1Id : text1Sets) {
		vector<set<int>> set1Windows = kWidthWindows[set1Id];
		for (int set2Id : text2Sets) {
			vector<set<int>> set2Windows = kWidthWindows[set2Id];
			// compute the alignment matrix if not computed previously
			if (results.find(key(set1Id, set2Id)) == results.end()) {
				AMatrix *A = new AMatrix(set1Windows, set2Windows, validedges, theta);
				A->computeAlignment();
				results[key(set1Id, set2Id)] = A;
			}
		}
		
	}
}

/**
 * Main Function : Entry Point
*/
int main(int argc, char const *argv[]) {

  	// arguments: text1_location, text2_location, window_width, threshold, result_folder, database location
	string text1_location = argv[1];
	string text2_location = argv[2];
	int k = stoi(argv[3]);
	double theta = stod(argv[4]);
	string result_folder = argv[5];
	string database_path = argv[6];	

	double envtime = 0.0;
	double invertedIndexSize = 0;
	std::chrono::time_point<std::chrono::high_resolution_clock> envstart, envend;
	std::chrono::duration<double> envlapsed;
	envstart = std::chrono::high_resolution_clock::now();
	Environment *env = new Environment(text1_location, text2_location, k);
	envend = std::chrono::high_resolution_clock::now();
	envlapsed = envend - envstart;
	envtime = envlapsed.count();

	vector<set<int>> invertedIndex = env->getInvertedIndex();
	for (set<int> f : invertedIndex) {
		invertedIndexSize += f.size();
	}
	invertedIndexSize += invertedIndex.size();
	cout << "Inverted Index Size: " << invertedIndexSize << endl;
	Database *db = new Database(database_path, env);
	cout << "Words: " << env->getWordSet().size() << endl;
	return 0;
}