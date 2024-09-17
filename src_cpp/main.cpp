#include <iostream>
#include <sstream>
#include <math.h>       /* sqrt */
#include <algorithm>    // std::sort
#include <chrono>

#include "Environment.h"
#include "Hungarian.h"
#include "PermutationSolver.h"
#include "Experiment.h"
#include "Solutions.h"

using namespace std;
inline size_t key(int i,int j) {return (size_t) i << 32 | (unsigned int) j;} // concat unsigned int with two integer set id as an edge's integer id

const bool LOGGING_MODE = false;

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
                    //TODO faiss::fvec_renorm_L2(300, 1, r);
                    vector<float> vr(r, r + 300);
                    vectors[token_id] = vr;
                    dictionary.push_back(token_id);
                }else{
                    cout << "Token not found in embeddings "<< values[1] << "->" << token_id << endl;
                }
            }
        }else{
            cout << "Error: Could not load " << location << endl;
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

    double sim(const int token1_ID, const int token2_ID) {
        if(token1_ID==token2_ID) {
            return 1;
        }
        // if either token doesn't have a vector then similarity is 0
        if ((vectors.find(token1_ID) == vectors.end()) || (vectors.find(token2_ID) == vectors.end())) {
            return 0;
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
        if(sim<0) {
            sim=0;
        }
        if(sim>1) {
            sim=1;
        }
        return sim;
    }

    double dist(const int token1_ID, const int token2_ID) {
        if(token1_ID==token2_ID) {
            return 0;
        }
        // if either token doesn't have a vector then similarity is 0
        if ((vectors.find(token1_ID) == vectors.end()) || (vectors.find(token2_ID) == vectors.end())) {
            return 1;
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
        double dist = 1-sim;
        if(dist<0) {
            dist=0;
        }
        if(dist>1) {
            dist=1;
        }
        return dist;
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
        if(sim<0) {
            sim=0;
        }
        if(sim>1) {
            sim=1;
        }
        cache[key(token1_ID, token2_ID)] = sim;

        return sim;
    }
};

class AMatrix {
private:
    const int width;
    const int height;
    const int k;

    double* data; // double* matrix 1-D representation of a matrix: matrix[i + j*width] = matrix[i][j].. matrix[width*height]... delete [] matrix;
    vector<vector<int>> *set1Windows;
    vector<vector<int>> *set2Windows;
    const double theta;
    int zero_entries = 0;
    DataLoader *dl;
    vector<vector<double>> cost_matrix_buffer;
    HungarianAlgorithm HungAlgo;

public:
    AMatrix(vector<vector<int>>* Ws, vector<vector<int>>* Wt, int _k, double threshold, DataLoader *_dl) :
        width(Ws->size())
        , height(Wt->size())
        , k(_k)
        , cost_matrix_buffer(k,vector<double>(k))
        , theta(threshold)
    {
        set1Windows = Ws;
        set2Windows = Wt;
        data = new double[width * height];
        dl = _dl;
    }

    void computeAlignment() {
        PermutationSolver ps(k);//for sanity checking XXX damn slow

        for (int row = 0; row < set1Windows->size(); ++row) {
            const int nsize = set2Windows->size();
            for (int col = 0; col < nsize; ++col) {
                const vector<int>& set1_tokens = set1Windows->at(row);
                const vector<int>& set2_tokens = set2Windows->at(col);

                for(int k_row=0;k_row<k;k_row++){
                    int set_1_id = set1_tokens.at(k_row);
                    const vector<float>& vec_1 = dl->get_vector(set_1_id);

                    //Compute the local distance matrix on the fly
                    for(int k_col=0;k_col<k;k_col++){
                        const int set_2_id = set2_tokens.at(k_col);
                        const vector<float>& vec_2 = dl->get_vector(set_2_id);
                        double token_similarity = 1-dl->calculate_similarity(set_1_id,set_2_id);//Compute the distances on the fly
                        set_cost_matrx(k_row,k_col,token_similarity);
                    }
                }

                double sim = solve(cost_matrix_buffer);
                double sim_by_perm = ps.solve(cost_matrix_buffer);

                if(sim != sim_by_perm){
                    cout << "sim != sim_by_perm @ row=" << row << " column="<<col <<endl;
                    for(vector<double> line : cost_matrix_buffer){
                        for(double d : line){
                            cout << d << " ";
                        }
                        cout << endl;
                    }
                    cout << endl;
                }

                // sim = 1.0;
                if (sim >= theta) {
                    setMatrixValue(row, col, sim);
                    //data[row + col*width] = sim;
                } else {
                    //data[row + col*width] = 0.0;
                    setMatrixValue(row, col, 0.0);
                    zero_entries += 1;
                }
            }
        }
    }

    double solve(vector<vector<double>>& cost_matrix) {
        vector<int> assignment;

        double cost = HungAlgo.Solve(cost_matrix, assignment);
        return -cost/k;
        return 1.0;
    }

    inline double getMatrixValue(const int row, const int col) const {
        return data[row + col * width];
    }

    inline void setMatrixValue(const int row, const int col, const double value) {
        data[row + col * width] = value;
    }

    inline void set_cost_matrx(const int row, const int col, const double value){
        cost_matrix_buffer.at(row).at(col) = value;
    }

    int get_matrix_size() {
        return width * height;
    }

    int zeroCells() {
        return zero_entries;
    }
};

void baseline(Environment& env, DataLoader *dl, int k, double theta){
    cout << "Running Baseline k=" << k << " theta=" << theta << endl;
    std::unordered_map<int, vector<int>> sets = env.getSets();  // all sets stored as key: set integer id, value: set data (int token integer ids)
    std::unordered_set<int> wordSet = env.getWordSet();         // all unique tokens in copora
    vector<set<int>> invertedIndex = env.getInvertedIndex();    // inverted index that returns all sets containing given token
    vector<int> dictionary = dl->getDictionary();               // vectors database instance
    set<int> text1Sets = env.getText1SetIds();
    set<int> text2Sets = env.getText2SetIds();

    std::unordered_map<size_t, AMatrix*> results; // key(text1SetId, text2SetId) --> AlignmentMatrix
    std::unordered_map<int, vector<vector<int>>> kWidthWindows = env.computeSlidingWindows(k); // setID --> sliding windows
    for(auto id : kWidthWindows){//<int, vector<vector<int>>
        cout << "set id=" << id.first << "->";
        for(vector<int> line : id.second){//vector<<vector<int>>
            cout << "{";
            for(int i : line){
                cout << i << " ";
            }
            cout << "} ";
        }
        cout << endl;
    }

    //Create buffers
    for (int set1Id : text1Sets) {
        vector<vector<int>>& set1Windows = kWidthWindows.at(set1Id);//access by offset?
        for (int set2Id : text2Sets) {
            vector<vector<int>>& set2Windows = kWidthWindows.at(set2Id);
            AMatrix *A = new AMatrix(&set1Windows, &set2Windows, k, theta, dl);
            results[key(set1Id, set2Id)] = A;
        }
    }

    int counter = 1;
    int numberOfGraphMatchingComputed = 0;
    int numberOfZeroEntries = 0;

    // Local Cost Matrix
    for (int set1Id : text1Sets) {
        cout << "set1Id=" << set1Id << endl;
     	for (int set2Id : text2Sets) {
     	    //Get pre-allocated, but not yet filled matrix
            AMatrix *A =  results.find(key(set1Id, set2Id))->second;
            //Next is the important line
            A->computeAlignment();
     	}
     	// cout << "Done with: " << counter << " / " << text1Sets.size() << endl;
     	counter += 1;
    }

    // Global Cost Matrix
    /*for (int set1Id : text1Sets) {
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
    }*/
}

vector<int> get_raw_book(Environment& env, set<int>& text_sets){
    vector<int> text_1_paragraph_ids;
    for(int id : text_sets){
        text_1_paragraph_ids.push_back(id);
    }
    sort (text_1_paragraph_ids.begin(), text_1_paragraph_ids.end());
    /*for(int id : text_1_paragraph_ids){
        cout << id << " ";
    }
    cout << endl;*/

    vector<vector<int>> raw_paragraphs;

    auto all_sets = env.getSets();

    vector<int> raw_book;

    for(int p_id : text_1_paragraph_ids){
        vector<int> paragraph = all_sets.at(p_id);
        for(int word_id : paragraph){
            //cout << env.toWord(word_id) << " ";
            raw_book.push_back(word_id);
        }
        //cout << endl;
    }
    return raw_book;
}

vector<vector<double>> get_sim_matrix(vector<int>& raw_book_1, vector<int>& raw_book_2, DataLoader& loader, Environment& env){
    //get size of matrix
    int max_id = 0;
    for(int id : raw_book_1){
        if(id>max_id){
            max_id = id;
        }
    }
    for(int id : raw_book_2){
        if(id>max_id){
            max_id = id;
        }
    }
    vector<vector<double>> global_dist_matrix(max_id+1, vector<double>(max_id+1));

    cout << "Computing global sim matrix [BEGIN]" << endl;

    chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
    for(int line =0;line<global_dist_matrix.size();line++){
        global_dist_matrix.at(line).at(line) = 1;
        for(int column=line+1;column<global_dist_matrix.at(0).size();column++){
            double dist = loader.sim(line,column);
            /*if(dist>0){
                cout << line <<" "<< column <<"->"<< dist << endl;
            }*/
            //exploit symmetry
            global_dist_matrix.at(line).at(column) = dist;
            global_dist_matrix.at(column).at(line) = dist;
        }
    }
    chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;
    //cout << "Sliding Window Computation time: " << time_elapsed.count() << endl;

    double sum = 0;

    for(auto arr : global_dist_matrix){
        for(double d : arr){
            sum+=d;
        }
    }

    cout << "Computing global similarity matrix Check sum="<< sum << " size= "<< global_dist_matrix.size()*global_dist_matrix.at(0).size() <<" [DONE] "<<time_elapsed.count()<< endl;

    return global_dist_matrix;//by value
}

vector<vector<double>> get_dist_matrix(vector<int>& raw_book_1, vector<int>& raw_book_2, DataLoader& loader, Environment& env){
    int max_id = 0;
    for(int id : raw_book_1){
        if(id>max_id){
            max_id = id;
        }
    }
    for(int id : raw_book_2){
        if(id>max_id){
            max_id = id;
        }
    }
    vector<vector<double>> global_dist_matrix(max_id+1, vector<double>(max_id+1));

    cout << "Computing global dist matrix [BEGIN]" << endl;

    chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
    for(int line =0;line<global_dist_matrix.size();line++){
        for(int column=line+1;column<global_dist_matrix.at(0).size();column++){
            double dist = loader.dist(line,column);
            //exploit symmetry
            global_dist_matrix.at(line).at(column) = dist;
            global_dist_matrix.at(column).at(line) = dist;
        }
    }
    chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;
    //cout << "Sliding Window Computation time: " << time_elapsed.count() << endl;

    double sum = 0;

    for(auto arr : global_dist_matrix){
        for(double d : arr){
            sum+=d;
        }
    }

    cout << "Computing global dist matrix Check sum="<< sum << " size= "<< global_dist_matrix.size()*global_dist_matrix.at(0).size() <<" [DONE] "<<time_elapsed.count()<< endl;

    if(LOGGING_MODE){
        vector<double> raw_data;
        for(auto arr : global_dist_matrix) {
            for(double d : arr) {
                raw_data.push_back(d);
            }
        }
        Histogram hist(raw_data);
        cout << "size\tsum\tmin\tmax" << endl;
        cout << hist.size() << "\t" << hist.sum() << "\t" << hist.get_min() << "\t"<<hist.get_max() << endl;
        cout << hist.getStatistics() << endl;
        cout << hist.toString() << endl;

        set<int> text1Sets = env.getText1SetIds();
        set<int> text2Sets = env.getText2SetIds();

        vector<int> book_1 = get_raw_book(env, text1Sets);
        vector<int> book_2 = get_raw_book(env, text2Sets);

        vector<double> raw_data_2;
        vector<vector<double>> matrix(book_1.size(), vector<double>(book_2.size()));
        for(int line = 0; line < book_1.size(); line++) {
            int token_id_1 = book_1.at(line);

            for(int column = 0; column < book_2.size(); column++){
                int token_id_2 = book_2.at(column);
                double dist = global_dist_matrix.at(token_id_1).at(token_id_2);
                raw_data_2.push_back(dist);
            }
        }
        Histogram hist_2(raw_data_2);
        cout << "size\tsum\tmin\tmax" << endl;
        cout << hist_2.size() << "\t" << hist_2.sum() << "\t" << hist_2.get_min() << "\t"<<hist_2.get_max() << endl;
        cout << hist_2.getStatistics() << endl;
        cout << hist_2.toString() << endl;
    }

    return global_dist_matrix;//by value
}

/**
 * Primarily maps Pramay's data strucutres to equivalent one of the Java implementation.
 * @param env
 * @param loader
 * @param theta
 */
void run_experiments(Environment& env, DataLoader& loader, const double theta){
    int num_paragraphs = 123;
    int max_id = 123;

    set<int> text1Sets = env.getText1SetIds();
    set<int> text2Sets = env.getText2SetIds();

    auto raw_book_1 = get_raw_book(env, text1Sets);
    auto raw_book_2 = get_raw_book(env, text2Sets);

    //vector<vector<double>> dist_matrix = get_dist_matrix(raw_book_1, raw_book_2, loader, env);
    vector<vector<double>> sim_matrix = get_sim_matrix(raw_book_1, raw_book_2, loader, env);

    /*vector<vector<double>> sim_copy(dist_matrix.size(), vector<double>(dist_matrix.at(0).size()));
    for(int line =0; line < dist_matrix.size();line++){
        for(int column=0;column<dist_matrix.at(0).size();column++){
            double dist = dist_matrix.at(line).at(line);
            sim_copy.at(line).at(column) = 1-dist;
        }
    }
    double sum = 0.0;
    for(auto arr : sim_copy){
        for(double d : arr){
            sum+=d;
        }
    }
    cout << "sim opcy check sum =" << sum << endl;*/

    vector<int> k_s = {3,4,5,6,7,8,9,10,11,12,13,14,15};
    //vector<int> k_s = {3};
    vector<double> run_times;



    for(int k : k_s){
        //Experiment exp(num_paragraphs, k, theta, max_id, raw_book_1, raw_book_2, dist_matrix);
        Solutions s(num_paragraphs, k, theta, max_id, raw_book_1, raw_book_2, sim_matrix);
        //exp.out_config();
        //double run_time = s.run_naive();
        //double run_time = s.run_baseline();
        //double run_time = s.run_baseline_deep();
        //double run_time = s.run_incremental_cell_pruning();
        double run_time = s.run_solution();
        //double run_time = exp.run_baseline();
        //double run_time = exp.run_baseline_safe();
        //double run_time = exp.run_pruning();
        //double run_time = exp.run_zick_zack();
        //double run_time = exp.run_pruning_max_matrix_value();
        //double run_time = exp.run_candidates();

        run_times.push_back(run_time);
    }
    for(int k : k_s){
        cout << "k="<<k<<"\t";
    }
    cout << endl;
    for(double t : run_times){
        cout << t <<"\t";
    }
    cout << endl;
}

int main() {
    //load_texts();
    int k = 3;
    double theta = 0.7;
    string text1location = "..//data/en/esv.txt";
    string text2location = "..//data/en/king_james_bible.txt";
    Environment env(text1location, text2location);
    //Environment env;
    env.out();

    bool ignore_stopwords = false;
    string data_file;
    if(ignore_stopwords) {
        data_file = "..//data/en/matches.en.min.tsv";
    }else{
        data_file = "..//data/en/matches_stopwords.en.min.tsv";
    }
    DataLoader loader(data_file, &env);

    run_experiments(env, loader, theta);

    return 0;
}
