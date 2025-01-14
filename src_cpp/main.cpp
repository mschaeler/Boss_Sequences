#include <iostream>
#include <sstream>
#include <cmath>       /* sqrt */
#include <chrono>

#include "Environment.h"
#include "Solutions.h"

using namespace std;

const int test_experiment  = 0;
const int bible_experiment = 1;
const int pan_experiment   = 2;
const int wiki_experiment  = 3;

const int run_naive = 0;
const int run_basem = 1;
const int run_seda  = 2;

const int run_naive_rb = 3;
const int run_basem_rb = 4;
const int run_seda_rb  = 5;
const int run_fast_text = 6;


/**
* Data loader to read vectors from a tsv file
*/
class DataLoader {
private:
    Environment *env;
    vector<int> dictionary;

    static std::vector<string> splitString(const string& line, char del) {
        std::vector<string> result;
        stringstream ss(line);
        string item;

        while (getline(ss, item, del)) {
            result.push_back(item);
        }
        return result;
    }
    std::unordered_map<int, vector<float>> vectors;
public:

    DataLoader(const string& location, Environment *e){
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
                    float* r;
                    r = embedding.data();
                    vector<float> vr(r, r + 300);
                    vectors[token_id] = vr;
                    dictionary.push_back(token_id);
                }else{
                    //cout << "Token not found in embeddings "<< values[1] << "->" << token_id << endl;
                }
            }
        }else{
            cout << "Error: Could not load " << location << endl;
        }
        file.close();
    }

    vector<vector<double>> get_word_vectors(){
        int max_id = 0;
        int vec_length;
        for (auto& it: vectors) {
            // Do stuff
            int my_id =  it.first;
            if(my_id>max_id){
                max_id = my_id;
            }
            vec_length = it.second.size();
        }
        vector<vector<double>> ret(max_id+1,vector<double>(vec_length));//Should all be inited with zero vectors
        for (auto& it: vectors) {
            // Do stuff
            vector<float>& my_vec =  it.second;
            for(int dim=0;dim<my_vec.size();dim++){
                ret.at(it.first).at(dim) = my_vec.at(dim);
            }
        }
        return ret;
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
        vector<float>& A = vectors[token1_ID];
        vector<float>& B = vectors[token2_ID];
        for(int i = 0; i < A.size(); ++i) {
            dot += A[i] * B[i];
            denom_a += A[i] * A[i];
            denom_b += B[i] * B[i];
        }

        auto sim = double(dot / (sqrt(denom_a) * sqrt(denom_b)));
        if(sim<0) {
            sim=0;
        }
        if(sim>1) {
            sim=1;
        }
        return sim;
    }
};

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

    cout << "Materializing sim() [BEGIN]" << endl;

    chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
    for(int line =0;line<global_dist_matrix.size();line++){
        global_dist_matrix.at(line).at(line) = 1;
        for(int column=line+1;column<global_dist_matrix.at(0).size();column++){
            double dist = loader.sim(line,column);
            //exploit symmetry
            global_dist_matrix.at(line).at(column) = dist;
            global_dist_matrix.at(column).at(line) = dist;
        }
    }
    chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;

    double sum = 0;

    for(const auto& arr : global_dist_matrix){
        for(double d : arr){
            sum+=d;
        }
    }

    cout << "Materializing sim() Check sum="<< sum << " size= "<< global_dist_matrix.size()*global_dist_matrix.at(0).size() <<" [DONE] "<<time_elapsed.count()<< endl;

    return global_dist_matrix;//by value
}

/**
 * Primarily maps Pranay's data structures to equivalent one of the Java implementation.
 * @param env
 * @param loader
 * @param theta
 */
vector<double> run_experiments(Environment& env, DataLoader& loader, const double theta, const vector<int>& k_s, const int approach, const int num_repitition){
    set<int> text1Sets = env.getText1SetIds();
    set<int> text2Sets = env.getText2SetIds();

    unordered_map<int, vector<int>> texts = env.getSets();
    vector<int> raw_book_1 = texts.at(0);
    vector<int> raw_book_2 = texts.at(1);

    vector<vector<double>> sim_matrix = get_sim_matrix(raw_book_1, raw_book_2, loader, env);
    vector<double> run_times;

    for(int k : k_s){//TODO repetitions
        Solutions s(k, theta, raw_book_1, raw_book_2, sim_matrix, loader.get_word_vectors());
        double run_time = 0;
        for(int i=0; i < num_repitition; i++) {
            if (approach == run_naive) {
                run_time += s.run_naive();
            } else if (approach == run_basem) {
                run_time += s.run_baseline();
            } else if (approach == run_seda) {
                run_time += s.run_solution();
            }else if (approach == run_naive_rb) {
                run_time += s.run_naive_rb();
            }else if (approach == run_basem_rb) {
                run_time += s.run_baseline_rb();
            }else if (approach == run_seda_rb) {
                run_time += s.run_solution_rb();
            }else if (approach == run_fast_text) {
                run_time += s.run_fast_text();
            } else {
                cout << "run_experiments() Now such approach " << approach << endl;
            }
        }
        run_times.push_back(run_time/double(num_repitition));
    }
    for(int k : k_s){
        cout << "k="<<k<<"\t";
    }
    cout << endl;
    for(double t : run_times){
        cout << t <<"\t";
    }
    cout << endl;

    return run_times;
}

int main(int argc, char* argv[]) {
    int experiment, approach_to_run;

    cout << "***SeDA experiment framework" << endl;
    double theta = 0.7;

    printf("You have entered %d arguments:\n", argc);

    for (int i = 0; i < argc; i++) {
        printf("%s\n", argv[i]);
    }

    if(argc != 3){
        cout << "Count of run time args must be equal to three, but got " << argc << " arguments listed above." << endl;
        cout << "Usage [program] [test,bible,pan,wiki] [0=naive,1=BaSEM,2=SeDA]" << endl;
        cout << "Running test experiment with SeDA instead" << endl;
        experiment = test_experiment;
        approach_to_run = run_seda;
    }else{
        //Determine experiment
        if(strcmp(argv[1],"test")==0){
            experiment = test_experiment;
        }else if(strcmp(argv[1],"bible")==0){
            experiment = bible_experiment;
        }else if(strcmp(argv[1],"pan")==0){
            experiment = pan_experiment;
        }else if(strcmp(argv[1],"wiki")==0){
            experiment = wiki_experiment;
        }else{
            printf("Unknown experiment %s. Running test experiment instead.\n", argv[1]);
            experiment = test_experiment;
        }

        //Determine approach
        if(strcmp(argv[2],"0")==0){//naive
            approach_to_run = run_naive;
        }else if(strcmp(argv[2],"1")==0){//BaSEM
            approach_to_run = run_basem;
        }else if(strcmp(argv[2],"2")==0){//SeDA
            approach_to_run = run_seda;
        }else if(strcmp(argv[2],"3")==0){//naive with ring buffer
            approach_to_run = run_naive_rb;
        }else if(strcmp(argv[2],"4")==0){//BaSEM with ring buffer
            approach_to_run = run_basem_rb;
        }else if(strcmp(argv[2],"5")==0){//SeDA with ring buffer
            approach_to_run = run_seda_rb;
        }else if(strcmp(argv[2],"6")==0){//FastText
            theta = 0.1;
            approach_to_run = run_fast_text;
        }else{
            printf("Unknown approach %s. Running SeDA approach instead.\n", argv[2]);
            approach_to_run = run_seda;
        }
    }
    cout << "***SeDA experiment framework running experiment=" << experiment << " with approach = " << approach_to_run << endl;

    if(experiment == test_experiment){
        vector<int> k_s = {3,4,5,6,7,8,9,10,11,12,13,14,15};
        string text1location = "..//data/en/esv.txt";
        string text2location = "..//data/en/king_james_bible.txt";
        Environment env(text1location, text2location);
        env.out();

        string data_file;
        data_file = "..//data/en/matches_stopwords.en.min.tsv";

        DataLoader loader(data_file, &env);
        int num_repition = 3;
        run_experiments(env, loader, theta, k_s, approach_to_run, num_repition);
    }else if(experiment == bible_experiment) {
        vector<vector<double>> all_runtimes;

        //First the two English texts
        vector<int> k_s = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
        int num_repition = 10;
        string data_file;
        {
            string text1location = "..//data/en/esv.txt";
            string text2location = "..//data/en/king_james_bible.txt";
            Environment env(text1location, text2location);
            data_file = "..//data/en/matches_stopwords.en.min.tsv";
            DataLoader loader(data_file, &env);
            auto temp = run_experiments(env, loader, theta, k_s, approach_to_run, num_repition);
            all_runtimes.push_back(temp);
        }
        vector<string> german_bible_versions = {"elberfelder.txt","luther.txt","ne.txt","schlachter.txt","volxbibel.txt",};
        data_file = "..//data/de/matches.de.min.tsv";
        //For each pair of biblical books
        for(auto b_1=0;b_1<german_bible_versions.size();b_1++){
            for(auto b_2=b_1+1;b_2<german_bible_versions.size();b_2++){
                cout << "**New pair " << german_bible_versions.at(b_1) <<" vs. "<< german_bible_versions.at(b_2) << endl;
                string text1location = "..//data/de/"+german_bible_versions.at(b_1);
                string text2location = "..//data/de/"+german_bible_versions.at(b_2);
                Environment env(text1location, text2location, Environment::DE);
                env.out();
                DataLoader loader(data_file, &env);
                auto temp = run_experiments(env, loader, theta, k_s, approach_to_run, num_repition);
                all_runtimes.push_back(temp);
            }
        }
        cout << "Results Bible experiment " << endl;
        for(int k : k_s){
            cout << "k="<<k<<"\t";
        }
        cout << endl;
        vector<double> avg_runtimes(k_s.size());
        for(auto run_times : all_runtimes) {
            for (auto i=0;i<run_times.size();i++) {
                double t = run_times[i];
                cout << t << "\t";
                avg_runtimes[i] +=t;
            }
            cout << endl;
        }
        cout << "Avg. run times Bible experiment " << endl;
        for(int k : k_s){
            cout << "k="<<k<<"\t";
        }
        cout << endl;
        for (auto i=0;i<avg_runtimes.size();i++) {
            double t = avg_runtimes[i]/(double) all_runtimes.size();
            cout << t << "\t";
        }
        cout << endl;
    }else if(experiment == pan_experiment){
        //TODO
        cout << "Pan experiment not implemented yet." << endl;
    }else if(experiment == wiki_experiment){
        vector<int> k_s = {10};
        //Parse dump into vector
        string wiki_dump_file = "..//data/en/wiki-1024000.txt";
        string line;
        ifstream infile(wiki_dump_file);
        vector<double> run_times;

        if (infile.is_open()) {
            getline(infile, line);
            vector<string> tokens = Environment::tokenize(line, Environment::EN);
            cout << "Parsed wiki dump into " << tokens.size() << " tokens" << endl;
            for(int i=0;i<10;i++){
                cout << tokens.at(i) << "\t";
            }
            cout << endl;
            int THOUSAND = 1000;
            vector<int> intput_sequence_length = {2*THOUSAND, 4*THOUSAND, 8*THOUSAND, 16*THOUSAND, 2*16*THOUSAND, 3*16*THOUSAND, 4*16*THOUSAND};//, 5*16*THOUSAND};//, 5*16*THOUSAND, 6*16*THOUSAND, 7*16*THOUSAND, 8*16*THOUSAND, 9*16*THOUSAND};
            for(int length : intput_sequence_length){
                cout << "Length = " << length << endl;
                Environment env(tokens, length);
                string embedding_file = "..//data/en/all_words_wiki.tsv";
                DataLoader loader(embedding_file, &env);

                int num_repition = 2;
                run_times.push_back(run_experiments(env, loader, theta, k_s, approach_to_run, num_repition).at(0));
            }
            cout << "Wikipedia run times for k=" << k_s.at(0) << endl;
            for(int i=0;i<intput_sequence_length.size();i++){
                cout << intput_sequence_length.at(i) << "\t" << run_times.at(i) << endl;
            }
        }else{
            cout << "Could not open " << wiki_dump_file << endl;
        }
    }else{
        cout << "Unknown experiment " << experiment << endl;
    }

    return 0;
}
