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
// #include "./hungarian-algorithm-cpp/Hungarian.h"

std::mutex gmtx;
using namespace std;
// using idx_t = faiss::Index::index_t;
inline size_t key(int i,int j) {return (size_t) i << 32 | (unsigned int) j;} // concat unsigned int with two integer set id as an edge's integer id

/**
 * Environment
*/

class Environment {};



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
			ss << "SELECT vec FROM wv WHERE word=\"" << token << "\";";
			string query = ss.str();
			string querybase = "SELECT vec FROM wv WHERE word=?;";
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
			ss << "SELECT word, vec FROM wv;";
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