//
// Created by Martin on 28.07.2023.
// Edited by Pranay on 04.08.2023.
//

#ifndef PRANAY_TEST_EXPERIMENT_H
#define PRANAY_TEST_EXPERIMENT_H

#include <string>
#include <vector>
#include "HungarianKevinStern.h"
#include "PermutationSolver.h"
#include "Environment.h"
#include <algorithm>    // std::sort
#include <chrono>

using namespace std;

class Histogram{
private:
    vector<double> raw_data;
    vector<double> bins_counts;
    vector<double> bins_min;
    double bin_width;

    int get_bin(double value) {
        for(int i=bins_min.size()-1;i>=0;i--) {
            if(bins_min.at(i)<=value) {
                return i;
            }
        }
        return 0;
    }

public:
    Histogram(vector<double> _raw_data) : raw_data(_raw_data), bins_counts(vector<double>(20)), bins_min(vector<double>(20)){

        sort(raw_data.begin(), raw_data.end());
        int num_bins = 20;

        bin_width = 0.05;
        double min_value = 0.05;
        for(int i=1;i<bins_min.size();i++) {
            bins_min.at(i) = min_value;
            min_value += bin_width;
        }
        for(double value : raw_data){
            int bin = get_bin(value);
            bins_counts.at(bin)++;
        }
    }

    int size() {
        return raw_data.size();
    }

    double sum() {
        double sum = 0;
        for(double i : raw_data) {
            sum+=i;
        }
        return sum;
    }

    double get_min() {
        return raw_data.at(0);
    }
    double get_max() {
        return raw_data.at(size()-1);
    }
    string getStatistics() {
        string header = "#Pairs\t#Distance\tavg(computations)";
        string data = to_string(size())+"\t"+to_string(sum())+"\t"+to_string((double)sum()/(double)size());
        return header+"\n"+data;
    }

    string toString() {
        string header = "";
        string data   = "";
        for(int bin=0;bin<bins_counts.size();bin++) {
            header+="<="+to_string(bins_min.at(bin))+"\t";
            data  +=to_string(bins_counts.at(bin))+"\t";
        }

        return header+"\n"+data;
    }
};


/**
 * At Book granularity
 */
class Experiment{
private:
    const double DOUBLE_PRECISION_BOUND = 0.0001;
    const double MAX_DOUBLE = 10000;

    const int k;
    const double threshold;

    const vector<int> book_1;
    const vector<int> book_2;

    vector<double> col_min;

    vector<vector<int>> k_with_windows_b1;
    vector<vector<int>> k_with_windows_b2;

    const vector<vector<double>> global_cost_matrix;
    vector<vector<double>> alignment_matrix;
    double col_sum = 0;

    /**
	 *
	 * @param raw_paragraphs all the paragraphs
	 * @param k - window size
	 * @return
	 */
    vector<vector<int>> create_windows(vector<int> book, int k) {
        vector<vector<int>> windows;
        for(int i=0;i<book.size()-k+1;i++){
            //create one window
            vector<int> window(k);
            for(int j=0;j<k;j++) {
                window.at(j) = book.at(i+j);
            }
            windows.push_back(window);
        }
        return windows;
    }

    double sum(const vector<vector<double>>& matrix) const {
        double sum = 0;
        for(auto arr : matrix){
            for(double d : arr){
                sum+=d;
            }
        }
        return sum;
    }

    double get_row_sum(const vector<vector<double>>& cost_matrix) const {
        double row_sum = 0;
        for(int i=0;i<k;i++) {
            const vector<double>& line = cost_matrix[i];
            double row_min = line[0];
            for(int j=1;j<k;j++) {
                const double val = line[j];
                if(val<row_min) {
                    row_min = val;
                }
            }
            row_sum += row_min;
        }

        return row_sum;
    }

    void min_col(const vector<vector<double>>& cost_matrix, const int replace_position) {
        col_sum-=col_min[replace_position];

        int row = 0;
        double min = cost_matrix[row][replace_position];
        row++;

        for(;row<k;row++) {
            double val = cost_matrix[row][replace_position];
            if(val < min) {
                min = val;
            }
        }

        col_min[replace_position] = min;
        col_sum+=min;
    }

public:
    Experiment(int _k, double _threshold, vector<int> _book_1, vector<int> _book_2, vector<vector<double>> _cost_matrix) :
        k(_k)
        , threshold(_threshold)
        , book_1(_book_1)
        , book_2(_book_2)
        , global_cost_matrix(_cost_matrix)
        , col_min(vector<double>(k))
    {
        k_with_windows_b1 = create_windows(book_1, k);
        k_with_windows_b2 = create_windows(book_2, k);
        vector<vector<double>> temp(k_with_windows_b1.size(), vector<double>(k_with_windows_b2.size()));
        alignment_matrix = temp;
        for(vector<double> arr : alignment_matrix){
            std::fill(arr.begin(),arr.end(),0);
        }
    }

    double run_baseline_safe(){
        cout << "run_baseline_safe() " << endl;
        out_config();

        PermutationSolver ps(k);
        HungarianKevinStern HKS(k);

        vector<vector<double>> cost_matrix(k, vector<double>(k));

        vector<int> assignment;

        cout << "Alignment matrix size " << alignment_matrix.size() << " x " << alignment_matrix.at(0).size() << endl;
        cout << "Cost matrix size " << global_cost_matrix.size() << " x " << global_cost_matrix.at(0).size() << endl;
        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

        for(int line=0;line<alignment_matrix.size();line++){
            vector<double> alignment_matrix_line = alignment_matrix.at(line);

            for(int column=0;column<alignment_matrix.at(0).size();column++){
                //Create local cost matrix for the Hungarian
                fill_local_cost_matrix(line,column,cost_matrix);

                double cost_safe = ps.solve(cost_matrix);
                double cost_hks = HKS.solve_cached(cost_matrix, threshold);

                if(cost_hks!=cost_safe){
                    cout << "sim != cost_hks @ row=" << line << " column="<<column << endl;
                    cout << cost_hks << " vs. "<<cost_safe << endl;
                    for(vector<double> line : cost_matrix){
                        for(double d : line){
                            cout << d << " ";
                        }
                        cout << endl;
                    }
                    cout << endl;
                    ps.solve(cost_matrix);
                }

                double normalized_similarity = 1.0 - (cost_hks / (double)k);
                if(normalized_similarity>=threshold) {
                    alignment_matrix_line.at(column) = normalized_similarity;
                }//else keep it zero
            }
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;
        cout << "run_baseline_sfae() time: " << time_elapsed.count() << endl;

        cout << " run_baseline_safe() [DONE]" << endl;
        return time_elapsed.count();
    }

    double sum(const vector<double> vec) const {
        double sum = 0;
        for(double d : vec){
            sum+=d;
        }
        return sum;
    }

    inline bool has_value_exceeding_threshold(const vector<vector<double>>& cost_matrix) {
        double dist_threshold = 1 - threshold;
        for(auto& arr : cost_matrix){
            for(double val : arr){
                if(val<=dist_threshold){
                    return true;
                }
            }
        }
        return false;
    }

    inline double get_column_row_sum(const vector<vector<double>>& cost_matrix) {
        double row_sum = 0;
        std::fill(col_min.begin(), col_min.end(), MAX_DOUBLE);
        for(int i=0;i<k;i++) {
            vector<double> line = cost_matrix[i];
            double row_min = MAX_DOUBLE;
            for(int j=0;j<k;j++) {
                const double val = line[j];
                if(val<row_min) {
                    row_min = val;
                }
                if(val<col_min.at(j)) {
                    col_min[j] = val;
                }
            }
            row_sum += row_min;
        }
        col_sum = sum(col_min);
        double min_cost = max(row_sum, col_sum);

        return min_cost;
    }

    inline void fill_local_cost_matrix(const int line, const int column, vector<vector<double>>& cost_matrix){
        for(int i=0;i<k;i++) {
            const int token_id_window_b1 = k_with_windows_b1[line][i];
            for(int j=0;j<k;j++) {
                const int token_id_window_b2 = k_with_windows_b2[column][j];
                cost_matrix[i][j] = global_cost_matrix[token_id_window_b1][token_id_window_b2];
            }
        }
    }

    double run_zick_zack(){
        cout << "run_zick_zack() " << endl;
        out_config();

        HungarianKevinStern HKS(k);
        vector<vector<double>> cost_matrix(k, vector<double>(k));
        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        double lb_cost;

        long count_survived_pruning = 0;
        long count_computed_cells   = 0;

        for(int line=0;line<alignment_matrix.size();line++){
            vector<double>& alignment_matrix_line = alignment_matrix.at(line);
            int column = 0;
            {	//Initially we really fill the entire cost matrix
                fill_local_cost_matrix(line,column,cost_matrix);

                lb_cost = get_column_row_sum(cost_matrix);

                double up_normalized_similarity = 1.0 - (lb_cost / (double)k);
                if(up_normalized_similarity+DOUBLE_PRECISION_BOUND>threshold) {
                    count_survived_pruning++;
                    //That's the important line
                    double cost = HKS.solve_cached(cost_matrix, threshold);//TODO , k_buffer);
                    //normalize costs: Before it was distance. Now it is similarity.
                    double normalized_similarity = 1.0 - (cost / (double)k);
                    if(normalized_similarity>=threshold) {
                        count_computed_cells++;
                        alignment_matrix_line[column] = normalized_similarity;
                    }//else keep it zero
                }
                column++;
            }
            //the idea is that we now change only one k-vector, but not re-fill the entire matrix again
            for(;column<alignment_matrix.at(0).size();column++) {
                //Update the cost matrix exploiting the rolling window. I.e., the cost matrix is ring buffer.
                const int replace_position = (column-1)%k;

                for(int i=0;i<k;i++) {
                    const int set_id_window_p1 = k_with_windows_b1[line][i];
                    const int set_id_window_p2 = k_with_windows_b2[column][k-1];//Always the new one
                    const double cost_l_c = global_cost_matrix[set_id_window_p1][set_id_window_p2];
                    cost_matrix[i][replace_position] = cost_l_c;
                }

                lb_cost = get_row_sum(cost_matrix);
                min_col(cost_matrix, replace_position);
                lb_cost = max(lb_cost, col_sum);//XXX should'nt that be the max?

                //lb_cost = get_column_row_sum(cost_matrix);
                const double up_normalized_similarity = 1.0 - (lb_cost / (double)k);
                if(up_normalized_similarity+DOUBLE_PRECISION_BOUND>threshold) {
                    count_survived_pruning++;
                    //That's the important line
                    double cost = HKS.solve_cached(cost_matrix, threshold);//;, k_buffer);
                    //normalize costs: Before it was distance. Now it is similarity.
                    double normalized_similarity = 1.0 - (cost / (double)k);
                    if(normalized_similarity>=threshold) {
                        count_computed_cells++;
                        alignment_matrix_line[column] = normalized_similarity;
                    }//else keep it zero
                }
            }

        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;
        int size = alignment_matrix.size() * alignment_matrix.at(0).size();
        double check_sum = sum(alignment_matrix);
        cout << "run_pruning() [DONE]" << endl;
        cout << "0\t" << time_elapsed.count() << "\t" << size << "\t" << check_sum << "\t" << count_survived_pruning << "\t" << count_computed_cells << endl;
        return time_elapsed.count();
    }

    /**
    * inverted_index.get(i) -> index for paragraph token_id
    * inverted_index.get(i)[token_id] -> some other_token_id with sim(token_id, other_token_id) > threshold
    */
    vector<vector<int>> create_neihborhood_index(const vector<vector<double>>& matrix) {
        cout << "create_neihborhood_index() BEGIN" << endl;
        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

        vector<vector<int>> indexes;
        for(const vector<double>& line : matrix) {
            vector<int> index;//TODO remove double effort?
            for(int id=0;id<line.size();id++) {
                const double dist = line[id];
                const double sim = 1 - dist;
                if(sim>=threshold){
                    index.push_back(id);
                }
            }
            indexes.push_back(index);
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;
        cout << "create_neihborhood_index() END in\t"+to_string(time_elapsed.count()) << endl;
        return indexes;
    }

    //TODO exploit running window property: Maybe order ids by frequency
    inline bool is_in(const vector<int>& neihborhood_index, const vector<int>& curr_window) const {
        for(int neighbor : neihborhood_index) {
            for(int t : curr_window) {
                if(t==neighbor) {
                    return true;
                }
            }
        }
        return false;
    }

    /**
    * indexes.get(token_id)[paragraph]-> int[] of position with token having sim(token_id, some token) > threshold
    * @param k_with_windows_b12
    * @return
    */
    vector<vector<int>> create_inverted_window_index(const vector<vector<int>>& k_with_windows, const vector<vector<int>>& neihborhood_indexes) {
        cout << "ArrayList<ArrayList<int[]>> create_inverted_window_index() BEGIN" << endl;
        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

        vector<vector<int>> indexes;
        //For each token
        for(int token_id = 0;token_id<neihborhood_indexes.size();token_id++) {
            //Create the list of occurrences for token: token_id
            const vector<int>& neihborhood_index = neihborhood_indexes.at(token_id);
            vector<vector<int>> occurences_per_paragraph;

            //For each windowed paragraph: Inspect whether one of the tokens in neihborhood_indexes is in the windows
            vector<int> index_this_paragraph;
            for(int pos=0;pos<k_with_windows.size();pos++) {
                vector<int> curr_window = k_with_windows[pos];
                if(is_in(neihborhood_index, curr_window)) {
                    index_this_paragraph.push_back(pos);
                }
            }

            indexes.push_back(index_this_paragraph);
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;
        cout << "create_inverted_window_index() END in\t"+to_string(time_elapsed.count()) << endl;

        return indexes;
    }

    double run_pruning_max_matrix_value(){
        cout << "run_pruning_max_matrix_value() " << endl;
        out_config();

        HungarianKevinStern HKS(k);
        vector<vector<double>> cost_matrix(k, vector<double>(k));
        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

        long count_survived_pruning = 0;
        long count_computed_cells   = 0;

        for(int line=0;line<alignment_matrix.size();line++){
            vector<double>& alignment_matrix_line = alignment_matrix.at(line);

            for(int column=0;column<alignment_matrix.at(0).size();column++){
                //Create local cost matrix for the Hungarian
                fill_local_cost_matrix(line,column,cost_matrix);

                const bool candidate = has_value_exceeding_threshold(cost_matrix);

                if(candidate) {
                    count_survived_pruning++;
                    double cost = HKS.solve_cached(cost_matrix, threshold);
                    //double cost = k;//For testing only
                    double normalized_similarity = 1.0 - (cost / (double) k);
                    if (normalized_similarity >= threshold) {
                        count_computed_cells++;
                        alignment_matrix_line.at(column) = normalized_similarity;
                    }//else keep it zero
                }
            }
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;
        int size = alignment_matrix.size() * alignment_matrix.at(0).size();
        double check_sum = sum(alignment_matrix);
        cout << "run_pruning_max_matrix_value() [DONE]" << endl;
        cout << "0\t" << time_elapsed.count() << "\t" << size << "\t" << check_sum << "\t" << count_survived_pruning << "\t" << count_computed_cells << endl;
        return time_elapsed.count();
    }

    double run_candidates(){
        cout << "run_candidates() " << endl;
        out_config();

        HungarianKevinStern HKS(k);
        vector<vector<double>> cost_matrix(k, vector<double>(k));
        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

        long count_survived_pruning = 0;
        long count_computed_cells   = 0;

        /**
         * inverted_index.get(i) -> index for paragraph token_id
         * inverted_index.get(i)[token_id] -> some other_token_id with sim(token_id, other_token_id) > threshold
         */
        const vector<vector<int>> neighborhood_index = create_neihborhood_index(global_cost_matrix);
        const vector<vector<int>> inverted_window_index = create_inverted_window_index(k_with_windows_b2, neighborhood_index);

        for(int line=0;line<alignment_matrix.size();line++){
            vector<bool> candidates(alignment_matrix[0].size(), false);
            vector<double>& alignment_matrix_line = alignment_matrix.at(line);
            //Create candidates: the window of p2 is fixed
            const vector<int>& window_p1 = k_with_windows_b1[line];
            for(int id : window_p1) {
                const vector<int>& index = inverted_window_index.at(id);
                for(int pos : index) {
                    candidates[pos] = true;
                }
            }

            //Validate candidates
            for(int column=0;column<candidates.size();column++) {
                bool is_candidate = candidates[column];
                if(is_candidate){
                    count_survived_pruning++;
                    //get local cost matrix
                    fill_local_cost_matrix(line,column,cost_matrix);

                    // (4) compute the bound
                    const double lb_cost = get_column_row_sum(cost_matrix);
                    const double up_normalized_similarity = 1.0 - (lb_cost / (double)k);
                    if(up_normalized_similarity+DOUBLE_PRECISION_BOUND>threshold) {
                        //That's the important line
                        double cost = HKS.solve_cached(cost_matrix, threshold);
                        //normalize costs: Before it was distance. Now it is similarity.
                        double normalized_similarity = 1.0 - (cost / (double)k);
                        if(normalized_similarity>=threshold) {
                            count_computed_cells++;
                            alignment_matrix_line[column] = normalized_similarity;
                        }//else keep it zero
                    }
                }//else safe mode
                /*if(SAFE_MODE) {//TODO
                    safe_mode_run_candidates(k_windows_p1, k_windows_p2, line, column, cost_matrix ,is_candidate);
                }*/
            }
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;
        int size = alignment_matrix.size() * alignment_matrix.at(0).size();
        double check_sum = sum(alignment_matrix);
        cout << "run_candidates() [DONE]" << endl;
        cout << "0\t" << time_elapsed.count() << "\t" << size << "\t" << check_sum << "\t" << count_survived_pruning << "\t" << count_computed_cells << endl;
        return time_elapsed.count();
    }

    double run_pruning_column_row_sum(){
        cout << "run_pruning_column_row_sum() " << endl;
        out_config();

        HungarianKevinStern HKS(k);
        vector<vector<double>> cost_matrix(k, vector<double>(k));
        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

        long count_survived_pruning = 0;
        long count_computed_cells   = 0;

        for(int line=0;line<alignment_matrix.size();line++){
            vector<double>& alignment_matrix_line = alignment_matrix.at(line);

            for(int column=0;column<alignment_matrix.at(0).size();column++){
                //Create local cost matrix for the Hungarian
                fill_local_cost_matrix(line,column,cost_matrix);

                const double lb_cost = get_column_row_sum(cost_matrix);
                //const double lb_cost = k;
                const double up_normalized_similarity = 1.0 - (lb_cost / (double)k);
                if(up_normalized_similarity+DOUBLE_PRECISION_BOUND>threshold) {
                    count_survived_pruning++;
                    double cost = HKS.solve_cached(cost_matrix, threshold);
                    //double cost = k;//For testing only
                    double normalized_similarity = 1.0 - (cost / (double) k);
                    if (normalized_similarity >= threshold) {
                        count_computed_cells++;
                        alignment_matrix_line.at(column) = normalized_similarity;
                    }//else keep it zero
                }
            }
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;
        int size = alignment_matrix.size() * alignment_matrix.at(0).size();
        double check_sum = sum(alignment_matrix);
        cout << "run_pruning_column_row_sum() [DONE]" << endl;
        cout << "0\t" << time_elapsed.count() << "\t" << size << "\t" << check_sum << "\t" << count_survived_pruning << "\t" << count_computed_cells << endl;
        return time_elapsed.count();
    }

    double run_baseline(){
        cout << "run_baseline() " << endl;
        out_config();

        HungarianKevinStern HKS(k);

        vector<vector<double>> cost_matrix(k, vector<double>(k));

        vector<int> assignment;

        cout << "Alignment matrix size " << alignment_matrix.size() << " x " << alignment_matrix.at(0).size() << endl;
        cout << "Cost matrix size " << global_cost_matrix.size() << " x " << global_cost_matrix.at(0).size() << endl;
        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

        for(int line=0;line<alignment_matrix.size();line++){
            vector<double> alignment_matrix_line = alignment_matrix.at(line);

            for(int column=0;column<alignment_matrix.at(0).size();column++){
                //Create local cost matrix for the Hungarian
                fill_local_cost_matrix(line,column,cost_matrix);

                //double cost = HungAlgo.Solve(cost_matrix, assignment);
                //HungarianKevinStern HKS(k);
                double cost = HKS.solve_cached(cost_matrix, threshold);
                double normalized_similarity = 1.0 - (cost / (double) k);
                if (normalized_similarity >= threshold) {
                    alignment_matrix_line.at(column) = normalized_similarity;
                }//else keep it zero
            }
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;
        cout << "run_baseline() time: " << time_elapsed.count() << endl;

        cout << "run_baseline() [DONE]" << endl;
        return time_elapsed.count();
    }

    void out_config(){
        cout << "Experiment k=" << k << " threshold=" << threshold << endl;
    }

    void out_data(){
        cout << "Book 1" << endl;
        for(int token_id : book_1){
            cout << token_id << " ";
        }
        cout << endl;

        cout << "Book 1 first 10 windows" << endl;
        for(vector<int> window : k_with_windows_b1){
            cout << "[";
            for(int word_id : window){
                cout << word_id << " ";
            }
            cout << "] ";
        }
        cout << endl;


        cout << "Book 2" << endl;
        for(int token_id : book_2){
            cout << token_id << " ";
        }
        cout << endl;
    }

    void out_data(Environment& env){
        cout << "Book 1" << endl;
        for(int token_id : book_1){
            cout << env.toWord(token_id) << " ";
        }
        cout << endl;

        cout << "Book 2" << endl;
        for(int token_id : book_2){
            cout << env.toWord(token_id) << " ";
        }
        cout << endl;
    }
};

#endif //PRANAY_TEST_EXPERIMENT_H
