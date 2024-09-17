//
// Created by Martin on 23.08.2023.
//

#ifndef PRANAY_TEST_SOLUTIONS_H
#define PRANAY_TEST_SOLUTIONS_H

#include <bitset>

/**
 * At Book granularity
 */
class Solutions{
    const double DOUBLE_PRECISION_BOUND = 0.0001;
    const double MAX_DOUBLE = 10000;

    const int num_paragraphs;
    const int k;
    const double k_double;
    const double threshold;
    const double threshold_times_k;
    const int max_id;

    const vector<int> book_1;
    const vector<int> book_2;

    vector<double> col_maxima;

    vector<vector<int>> k_with_windows_b1;
    vector<vector<int>> k_with_windows_b2;

    vector<int> tokens_b1;
    vector<int> tokens_b2;

    const vector<vector<double>> global_similarity_matrix;
    vector<vector<double>> alignment_matrix;
    double sum_cols = 0;

    const double MAX_SIM_ADDITION_NEW_NODE;

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
        for(vector<double> arr : matrix){
            for(double d : arr){
                sum+=d;
            }
        }
        return sum;
    }
    double sum(const vector<double>& arr) const {
        double sum = 0;
        for(double d : arr){
            sum+=d;
        }
        return sum;
    }
    void out_config(string name){
        cout << "Solutions "<<name<<" k=" << k << " threshold=" << threshold << " " << threshold_times_k << endl;
    }
    vector<vector<double>> fill_similarity_matrix() {
        vector<vector<double>> book_matrix(book_1.size(), vector<double>(book_2.size()));
        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

        for(int line=0;line<book_1.size();line++) {
            const int set_id_window_p1 = book_1.at(line);
            const vector<double>& sim_matrix_line = global_similarity_matrix.at(set_id_window_p1);
            for(int column=0;column<book_2.size();column++) {
                const int set_id_window_p2 = book_2.at(column);
                const double sim = sim_matrix_line.at(set_id_window_p2);
                book_matrix.at(line).at(column) = sim;
            }
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;
        cout << "fill_similarity_matrix END in " << time_elapsed.count() << endl;

        return book_matrix;//by value
    }

    /**
     *
     * @return -sim[][]
     */
    vector<vector<double>> fill_similarity_matrix_deep() {
        vector<vector<double>> book_matrix(book_1.size(), vector<double>(book_2.size()));

        for(int line=0;line<book_1.size();line++) {
            const int set_id_window_p1 = book_1.at(line);
            const vector<double>& sim_matrix_line = global_similarity_matrix.at(set_id_window_p1);
            for(int column=0;column<book_2.size();column++) {
                const int set_id_window_p2 = book_2.at(column);
                const double sim = sim_matrix_line.at(set_id_window_p2);
                book_matrix.at(line).at(column) = -sim;// XXX this is the difference to the method above
            }
        }

        return book_matrix;//by value
    }

    void fill_local_similarity_matrix(vector<vector<double>>& local_cost_matrix, const vector<vector<double>>& global_cost_matrix_book, const int line, const int column) const {
        for(int i=0;i<k;i++) {
            for(int j=0;j<k;j++) {
                local_cost_matrix.at(i).at(j) = -global_cost_matrix_book.at(line+i).at(column+j);//XXX - Note the minus for the Hungarian
            }
        }
    }

    double o_k_square_bound(const vector<vector<double>>& similarity_matrix) {
        double row_sum = 0;
        std::fill(col_maxima.begin(), col_maxima.end(), MAX_DOUBLE);
        for(int i=0;i<k;i++) {
            const vector<double>& line = similarity_matrix.at(i);
            double row_min = MAX_DOUBLE;
            for(int j=0;j<k;j++) {
                const double val = line.at(j);
                if(val<row_min) {
                    row_min = val;
                }
                if(val<col_maxima.at(j)) {
                    col_maxima.at(j) = val;
                }
            }
            row_sum += row_min;
        }
        sum_cols = sum(col_maxima);//FIXME
        double max_similarity = -max(row_sum, sum_cols);

        return max_similarity;
    }

    double o_k_square_bound(const vector<const double*>& similarity_matrix) {
        double row_sum = 0;
        std::fill(col_maxima.begin(), col_maxima.end(), MAX_DOUBLE);
        for(int i=0;i<k;i++) {
            const double* line = similarity_matrix.at(i);
            double row_min = MAX_DOUBLE;
            for(int j=0;j<k;j++) {
                const double val = line[j];
                if(val<row_min) {
                    row_min = val;
                }
                if(val<col_maxima.at(j)) {
                    col_maxima.at(j) = val;
                }
            }
            row_sum += row_min;
        }
        sum_cols = sum(col_maxima);//FIXME
        double max_similarity = -max(row_sum, sum_cols);

        return max_similarity;
    }

    double get_sum_of_column_row_minima_deep(const vector<vector<double>>& matrix_deep, const int line, const int column) {
        double row_sum = 0;
        std::fill(col_maxima.begin(), col_maxima.end(), MAX_DOUBLE);
        for(int i=0;i<k;i++) {
            const vector<double>& my_line = matrix_deep.at(line+i);
            double row_min = MAX_DOUBLE;
            for(int j=0;j<k;j++) {
                const double val = my_line.at(column+j);
                if(val<row_min) {
                    row_min = val;
                }
                if(val<col_maxima.at(j)) {
                    col_maxima.at(j) = val;
                }
            }
            row_sum += row_min;
        }
        double col_sum = sum(col_maxima);
        double max_similarity = -max(row_sum, col_sum);

        return max_similarity;
    }

    double get_sum_of_column_row_minima(const vector<vector<double>>& similarity_matrix) {
        double row_sum = 0;
        std::fill(col_maxima.begin(), col_maxima.end(), MAX_DOUBLE);
        for(int i=0;i<k;i++) {
            const vector<double>& line = similarity_matrix.at(i);
            double row_min = MAX_DOUBLE;
            for(int j=0;j<k;j++) {
                const double val = line.at(j);
                if(val<row_min) {
                    row_min = val;
                }
                if(val<col_maxima.at(j)) {
                    col_maxima.at(j) = val;
                }
            }
            row_sum += row_min;
        }
        double col_sum = sum(col_maxima);
        double max_similarity = -max(row_sum, col_sum);

        return max_similarity;
    }

    double sum_bound_similarity(const vector<vector<double>>& similarity_matrix) {
        double row_sum = 0;
        std::fill(col_maxima.begin(), col_maxima.end(), MAX_DOUBLE);
        for(int i=0;i<k;i++) {
            const vector<double>& line = similarity_matrix.at(i);
            double row_min = MAX_DOUBLE;
            for(int j=0;j<k;j++) {
                const double val = line.at(j);
                if(val<row_min) {
                    row_min = val;
                }
                if(val<col_maxima.at(j)) {
                    col_maxima.at(j) = val;
                }
            }
            row_sum += row_min;
        }
        sum_cols = sum(col_maxima);
        double min_cost = max(row_sum, sum_cols);

        return -min_cost;
    }

    double min_vector(const vector<vector<double>>& similarity_matrix) const {
        double min = similarity_matrix.at(0).at(k-1);
        for(int line=1;line<similarity_matrix.size();line++) {
            if(min>similarity_matrix.at(line).at(k-1)) {
                min=similarity_matrix.at(line).at(k-1);
            }
        }
        return -min;
    }

    double max_val(const vector<vector<double>>& similarity_matrix) const {
        double max = -2000;//TODO remove this line?
        for(auto line : similarity_matrix) {
            if(max<line.at(0)) {//similarity of the deleted token
                max=line.at(0);
            }
        }
        return -max;
    }

    void fill_local_similarity_matrix(const vector<int>& k_window_p1, const vector<int>& k_window_p2, vector<vector<double>>& local_similarity_matrix){
        for(int i=0;i<k;i++) {
            const int token_id_1 = k_window_p1.at(i);
            const vector<double>& matrix_line = global_similarity_matrix.at(token_id_1);
            for(int j=0;j<k;j++) {
                const int token_id_2 = k_window_p2.at(j);
                double sim = (token_id_1==token_id_2) ? 1 : matrix_line.at(token_id_2);
                local_similarity_matrix.at(i).at(j) = -sim;//Note the minus-trick for the Hungarian
            }
        }
    }

    void out(const vector<int>& vector) {
        cout << "size=" << vector.size() <<" ";
        for(int v : vector){
            cout << v << " ";
        }
        cout << endl;
    }

    void create_indexes_bit_vectors(vector<vector<bool>>& inverted_window_index){
        cout << "create neighborhood index BEGIN" << endl;
        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        vector<vector<int>> indexes;
        //find for each set all other sets such that sim(set,other_set)>=threshold
        for(int token_id : tokens_b1){
            const vector<double>& line = global_similarity_matrix.at(token_id);
            vector<int> index;
            for(int id : tokens_b2){
                const double sim = line.at(id);
                if(sim>=threshold){
                    index.push_back(id);
                }
            }
            indexes.push_back(index);
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;
        cout << "create neighborhood index END in " << time_elapsed.count() << endl;

        cout << "Bit vector create BEGIN" << endl;
        start = std::chrono::high_resolution_clock::now();
        //For each token
        for(int token_id : tokens_b1) {
            /**
             * The list of all tokens with sim > threshold
             */
            const vector<int>& neihborhood_index = indexes.at(token_id);
            vector<bool>& bit_vector = inverted_window_index.at(token_id);

            for(int pos=0;pos<book_2.size();pos++) {
                const int token_id_in_b2 = book_2.at(pos);

                if(isIn(neihborhood_index,token_id_in_b2)) {
                    int start = max(0, pos-k+1);
                    int stop =  (k_with_windows_b2.size()-1 < pos) ? k_with_windows_b2.size()-1 : pos;
                    for(int pos=start;pos<=stop;pos++) {
                        bit_vector.at(pos) = 1;
                    }
                }
            }
        }
        time_elapsed = std::chrono::high_resolution_clock::now() - start;
        cout << "Bit vector create END in " << time_elapsed.count() << endl;
    }

    /**
     * O(n)
     * @param value
     * @return
     */
    bool isIn(const vector<int>& neihborhood_index, const int value) const {
        const int size = neihborhood_index.size();
        for(int i=0;i<size;i++) {
            if(neihborhood_index[i]==value) {
                return true;
            }
        }
        return false;
    }

    bool is_in(const vector<int>& neihborhood_index, const vector<int>& curr_window) const {
        for(int i=0;i<neihborhood_index.size();i++) {
            const int neighbor = neihborhood_index.at(i);
            for(int t : curr_window) {
                if(t==neighbor) {
                    return true;
                }
            }
        }
        return false;
    }

    vector<int> get_tokens(const vector<int>& book) {
        unordered_set<int> temp;
        for(int id : book){
            temp.insert(id);
        }
        vector<int> ret;
        for(auto v : temp){
            ret.push_back(v);
        }
        sort(ret.begin(), ret.end());

        return ret;
    }

public:

    Solutions(int _num_paragraphs, int _k, double _threshold, int _max_id, vector<int> _book_1, vector<int> _book_2, vector<vector<double>> _cost_matrix) :
            num_paragraphs(_num_paragraphs)
            , k(_k)
            , k_double((double)_k)
            , threshold_times_k(_threshold*_k)
            , threshold(_threshold)
            , max_id(_max_id)
            , book_1(_book_1)
            , book_2(_book_2)
            , global_similarity_matrix(_cost_matrix)
            , col_maxima(vector<double>(k))
            , MAX_SIM_ADDITION_NEW_NODE(1.0 / k_double)
    {
        k_with_windows_b1 = create_windows(book_1, k);
        k_with_windows_b2 = create_windows(book_2, k);

        tokens_b1 = get_tokens(book_1);
        tokens_b2 = get_tokens(book_2);

        vector<vector<double>> temp(k_with_windows_b1.size(), vector<double>(k_with_windows_b2.size()));
        alignment_matrix = temp;
        for(vector<double> arr : alignment_matrix){
            std::fill(arr.begin(),arr.end(),0);
        }

        //cout << sum(global_similarity_matrix) << endl;
    }

    void out_matrix(vector<vector<double>>& matrix){
        for(auto arr : matrix){
            for(auto d : arr){
                cout << d << "\t";
            }
            cout << endl;
        }
    }

    //XXX this one does not compute the distances on the fly. Add time?
    double run_naive(){
        out_config("run_naive()");
        long count_computed_cells = 0;
        HungarianKevinStern solver(k);

        vector<vector<double>> local_similarity_matrix(k, vector<double>(k));
        //USE_GLOBAL_MATRIX = false;

        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

        const vector<vector<double>> matrix_book = fill_similarity_matrix();
        //For each pair of windows
        for(int line=0;line<alignment_matrix.size();line++) {
            for(int column=0;column<alignment_matrix.at(0).size();column++) {
                //Fill local matrix of the current window combination from global matrix
                fill_local_similarity_matrix(local_similarity_matrix, matrix_book, line, column);
                //That's the important line
                const double similarity = -solver.solve_cached(local_similarity_matrix, threshold);
                //normalize costs: Before it was distance. Now it is similarity.
                if(similarity>=threshold_times_k) {
                    alignment_matrix.at(line).at(column) = similarity/(double)k;//normalize
                    count_computed_cells++;
                }//else keep it zero
            }
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;

        double check_sum = sum(alignment_matrix);
        long size = alignment_matrix.size()*alignment_matrix.at(0).size();
        cout << "run_naive() time: " << time_elapsed.count() << "\t" << check_sum << "\t" <<  size << "\t" << count_computed_cells << endl;

        return time_elapsed.count();
    }

    double run_baseline_deep() {
        out_config("run_baseline_deep()");
        long count_computed_cells = 0;
        long count_survived_pruning = 0;
        vector<const double*> local_similarity_matrix(k);//Can't use a vector to point into an existing buffer.
        HungarianDeep solver_deep(k);

        //This is the main difference, we can re-use it for any k. Thus, not part of run time.
        const vector<vector<double>> matrix_book = fill_similarity_matrix_deep();

        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        //For each pair of windows
        for (int line = 0; line < alignment_matrix.size(); line++) {
            for (int column = 0; column < alignment_matrix.at(0).size(); column++) {
                for(int i = 0;i<k;i++){
                    const double* temp = &matrix_book.at(line+i)[column];
                    local_similarity_matrix.at(i) = temp;
                }

                const double upper_bound_sim = o_k_square_bound(local_similarity_matrix);

                if (upper_bound_sim + DOUBLE_PRECISION_BOUND >= threshold_times_k) {
                    count_survived_pruning++;
                    double similarity_deep = -solver_deep.solve(col_maxima, local_similarity_matrix);
                    if (similarity_deep >= threshold_times_k) {
                        alignment_matrix.at(line).at(column) = similarity_deep / (double) k;//normalize
                        count_computed_cells++;
                    }//else keep it zero
                }
            }
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;

        double check_sum = sum(alignment_matrix);
        long size = alignment_matrix.size() * alignment_matrix.at(0).size();
        cout << "run_baseline_deep() time: " << time_elapsed.count() << "\t" << check_sum << "\t" << size << "\t"
             << count_survived_pruning << "\t" << count_computed_cells << endl;

        return time_elapsed.count();
    }

    /*double run_baseline_deep() {
        out_config("run_baseline_deep()");
        long count_computed_cells = 0;
        long count_survived_pruning = 0;
        vector<const double*> local_similarity_matrix(k);//Can't use a vector to point into an existing buffer.
        //HungarianKevinStern solver(k);
        HungarianDeep solver_deep(k);

        //vector<vector<double>> local_similarity_matrix_baseline(k, vector<double>(k));

        //This is the main difference, we can re-use it for any k. Thus, not part of run time.
        const vector<vector<double>> matrix_book = fill_similarity_matrix_deep();
        //const vector<vector<double>> matrix_book_baseline = fill_similarity_matrix();
        //TODO check equivalence minused
        /*for(int line=0;line<matrix_book.size();line++){
            for(int column=0;column<matrix_book.at(0).size();column++){
                if(matrix_book.at(line).at(column)!= -matrix_book_baseline.at(line).at(column)){
                    cout << "Matrices not equal at line=" << line << " column=" << column <<" "<<  matrix_book.at(line).at(column) <<" "<<  matrix_book_baseline.at(line).at(column) << endl;
                }
            }
        }

        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        //For each pair of windows
        for (int line = 0; line < alignment_matrix.size(); line++) {
            for(int i = 0;i<k;i++){
                const double* temp = &matrix_book[line+i][0];
                local_similarity_matrix.at(i) = temp;
            }
            /*fill_local_similarity_matrix(local_similarity_matrix_baseline, matrix_book_baseline, line, 0);
            for(int i=0;i<k;i++){
                for(int c=0;c<k;c++){
                    if(local_similarity_matrix.at(i)[c]!= local_similarity_matrix_baseline.at(i).at(c)){
                        cout << "Local matrices not equal at line=" << i << " column=" << c <<" "<<  matrix_book.at(i).at(c) <<" "<<  matrix_book_baseline.at(i).at(c) << endl;
                    }
                }
            }
            for (int column = 0; column < alignment_matrix.at(0).size(); column++) {
                for(int i = 0;i<k;i++){
                    const double* temp = &matrix_book.at(line+i)[column];
                    local_similarity_matrix.at(i) = temp;
                }
                //Fill local matrix of the current window combination from global matrix
                //fill_local_similarity_matrix(local_similarity_matrix_baseline, matrix_book_baseline, line, column);

                //TODO check equivalence of local matrices
                /*for(int i=0;i<k;i++){
                    for(int c=0;c<k;c++){
                        if(local_similarity_matrix.at(i)[c]!= local_similarity_matrix_baseline.at(i).at(c)){
                            cout << "Local matrices not equal at line=" << i << " column=" << c <<" "<<  matrix_book.at(i).at(c) <<" "<<  matrix_book_baseline.at(i).at(c) << endl;
                        }
                    }
                }

                const double upper_bound_sim = o_k_square_bound(local_similarity_matrix);

                if (upper_bound_sim + DOUBLE_PRECISION_BOUND >= threshold_times_k) {
                    count_survived_pruning++;
                    //That's the important line
                    //double similarity = -solver.solve_cached(local_similarity_matrix_baseline, threshold);
                    double similarity_deep = -solver_deep.solve(col_maxima, local_similarity_matrix);

                    /*if(similarity!=similarity_deep){
                        cout << similarity << "\t" << similarity_deep << endl;
                    }

                    //normalize costs: Before it was distance. Now it is similarity.
                    if (similarity_deep >= threshold_times_k) {
                        alignment_matrix.at(line).at(column) = similarity_deep / (double) k;//normalize
                        count_computed_cells++;
                    }//else keep it zero
                }
            }
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;

        double check_sum = sum(alignment_matrix);
        long size = alignment_matrix.size() * alignment_matrix.at(0).size();
        cout << "run_baseline_deep() time: " << time_elapsed.count() << "\t" << check_sum << "\t" << size << "\t"
             << count_survived_pruning << "\t" << count_computed_cells << endl;

        return time_elapsed.count();
    }*/

    double run_baseline() {
        out_config("run_baseline()");
        long count_computed_cells = 0;
        long count_survived_pruning = 0;
        HungarianKevinStern solver(k);

        vector<vector<double>> local_similarity_matrix(k, vector<double>(k));

        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

        const vector<vector<double>> matrix_book = fill_similarity_matrix();
        //For each pair of windows
        for (int line = 0; line < alignment_matrix.size(); line++) {
            for (int column = 0; column < alignment_matrix.at(0).size(); column++) {
                //Fill local matrix of the current window combination from global matrix
                fill_local_similarity_matrix(local_similarity_matrix, matrix_book, line,
                                             column);//das hier ist der einzige Unterschied
                const double upper_bound_sim = get_sum_of_column_row_minima(local_similarity_matrix);

                if (upper_bound_sim + DOUBLE_PRECISION_BOUND >= threshold_times_k) {
                    count_survived_pruning++;
                    //That's the important line
                    double similarity = -solver.solve_cached(local_similarity_matrix, threshold);
                    //normalize costs: Before it was distance. Now it is similarity.
                    if (similarity >= threshold_times_k) {
                        alignment_matrix.at(line).at(column) = similarity / (double) k;//normalize
                        count_computed_cells++;
                    }//else keep it zero
                }
            }
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;

        double check_sum = sum(alignment_matrix);
        long size = alignment_matrix.size() * alignment_matrix.at(0).size();
        cout << "run_baseline() time: " << time_elapsed.count() << "\t" << check_sum << "\t" << size << "\t"
             << count_survived_pruning << "\t" << count_computed_cells << endl;

        return time_elapsed.count();
    }

    void check_bit_vector_index(vector<vector<bool>>& inverted_window_index){
        for(const vector<int>& window_1: k_with_windows_b1){
            for(int w=0;w<k_with_windows_b2.size();w++){
                const vector<int>& window_2 = k_with_windows_b2.at(w);
                for(int id_1 : window_1){
                    double max_sim = 0;
                    for(int id_2 : window_2){
                        double sim = global_similarity_matrix.at(id_1).at(id_2);
                        if(sim>max_sim){
                            max_sim = sim;
                        }
                    }
                    if(max_sim>=threshold){
                        if(!inverted_window_index.at(id_1).at(w)){//must be true
                            cout << "inverted_window_index wrong must be true at " << id_1 << " " << w << endl;
                            out(window_1);
                            out(window_2);
                        }
                    }else{
                        if(inverted_window_index.at(id_1).at(w)){//must not be true
                            cout << "inverted_window_index wrong must be false at " << id_1 << " " << w << endl;
                            out(window_1);
                            out(window_2);
                        }
                    }
                }
            }
        }
    }

    double run_solution(){
        out_config("run_solution()");
        HungarianDeep solver(k);
        /**
         * Indicates for token i whether the corresponding windows of the other sequence is a candidate.
         */
        vector<vector<bool>> inverted_window_index(global_similarity_matrix.size(), vector<bool>(k_with_windows_b2.size()));
        vector<vector<int>> all_candidates(k_with_windows_b1.size());
        //Not needed later
        vector<const double*> window(k);//Can't use a vector to point into an existing buffer.
        const vector<vector<double>> matrix_book = fill_similarity_matrix_deep();

        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        create_indexes_bit_vectors(inverted_window_index);

        long count_candidates = 0;
        long count_survived_o_1 = 0;
        long count_survived_o_k = 0;
        long count_survived_o_k_squar = 0;
        long count_cells_exceeding_threshold = 0;

        //check_bit_vector_index(inverted_window_index);//XXX for debug only

        const int vector_size = inverted_window_index.at(0).size();
        for(int line=0;line<alignment_matrix.size();line++) {
            const vector<int>& window_b1 = k_with_windows_b1.at(line);
            vector<bool> candidates(vector_size);
            bool found_run = false;
            for(int pos=0;pos<vector_size;pos++){
                for(int i=0;i<k;i++){//There is a candidate at *pos if at least one vector at *pos* is true
                    const int token_id = window_b1.at(i);
                    if(inverted_window_index.at(token_id).at(pos)){
                        candidates.at(pos) = true;
                        break;
                    }
                }
            }
            {
                //Manually inlined condense
                vector<int> candidates_condensed(20);
                int q = 0;
                bool found_run = false;
                while(q<candidates.size()) {
                    if(candidates[q]) {//start of a run
                        candidates_condensed.push_back(q);
                        q++;
                        found_run = true;
                        while(q<candidates.size()) {
                            if(!candidates[q]){//end of run
                                candidates_condensed.push_back(q-1);
                                found_run = false;
                                break;
                            }else{
                                q++;
                            }
                        }
                    }
                    q++;
                }
                if(found_run) {
                    candidates_condensed.push_back(candidates.size()-1);
                }
                all_candidates.at(line) = candidates_condensed;//XXX does it copy the vector?
            }
        }
        chrono::duration<double> index_creation = std::chrono::high_resolution_clock::now() - start;

        //Check candidate runs
        for(int line=0;line<alignment_matrix.size();line++) {//TODO deep integration
            vector<double>& alignment_matrix_line = alignment_matrix.at(line);
            vector<int>& candidates_condensed = all_candidates.at(line);

            const int size = candidates_condensed.size();
            for(int c=0;c<size;c+=2) {//Contains start and stop index. Thus, c+=2.
                const int run_start = candidates_condensed.at(c);
                const int run_stop = candidates_condensed.at(c+1);

                double ub_sum, sim, prior_cell_similarity, prev_min_value;
                bool prior_cell_updated_matrix, column_sum_correct;
                //cout << run_start << " " << run_stop << endl;

                int column=run_start;
                {//Here we have no O(1) bound
                    count_candidates++;
                    count_survived_o_1++;
                    count_survived_o_k++;
                    for(int i = 0;i<k;i++){//Init sliding window
                        const double* temp = &matrix_book.at(line+i)[column];
                        window.at(i) = temp;
                    }
                    ub_sum = o_k_square_bound(window) / k_double;

                    if(ub_sum+DOUBLE_PRECISION_BOUND>=threshold) {
                        count_survived_o_k_squar++;
                        sim = -solver.solve(col_maxima, window);//Note the minus-trick for the Hungarian
                        sim /= k_double;
                        if(sim>=threshold) {
                            count_cells_exceeding_threshold++;
                            //if(LOGGING_MODE) count_cells_exceeding_threshold++;
                            alignment_matrix_line.at(column) = sim;
                        }//else keep it zero
                        prior_cell_similarity = sim;
                    }else{
                        prior_cell_similarity = ub_sum;
                        /*sim = -solver.solve_cached(window, threshold) / k_double;//Note the minus-trick for the Hungarian
                        if(sim>ub_sum+DOUBLE_PRECISION_BOUND){//TODO remove me
                            cout << "sim>ub_sum+DOUBLE_PRECISION_BOUND" << endl;
                        }*/
                    }
                    prev_min_value = max_column(window);
                    prior_cell_updated_matrix = true;
                    column_sum_correct = true;
                }

                //For all other columns: Here we have a O(1) and O(k) bound
                for(column=run_start+1;column<=run_stop;column++) {
                    count_candidates++;
                    for(int i = 0;i<k;i++){//Init sliding window
                        const double* temp = &matrix_book.at(line+i)[column];
                        window.at(i) = temp;
                    }
                    /*if(line == 0 && column == 4){//TODO remove me
                        cout << line << " " << column << endl;
                    }*/
                    double upper_bound_sim = prior_cell_similarity + MAX_SIM_ADDITION_NEW_NODE;// O(1) bound
                    if(prior_cell_updated_matrix) {
                        upper_bound_sim-= (prev_min_value / k_double);// (1) O(k) bound : part of the O(k) bound in case the prior cell updated the matrix, i.e., we know the minimum similarity of the leaving node
                    }

                    if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
                        count_survived_o_1++;

                        double max_sim_new_node = min(window);//(2) O(k) bound
                        upper_bound_sim-=MAX_SIM_ADDITION_NEW_NODE;
                        upper_bound_sim+=(max_sim_new_node/k_double);

                        if(column_sum_correct) {
                            sum_cols -= col_maxima.at(0);
                            sum_cols -= max_sim_new_node;//is not negated
                            double temp = -sum_cols / k_double;

                            if(temp<upper_bound_sim) {
                                upper_bound_sim = temp;
                            }
                        }

                        if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
                            count_survived_o_k++;
                            ub_sum = o_k_square_bound(window) / k_double;
                            upper_bound_sim = (ub_sum<upper_bound_sim) ? ub_sum : upper_bound_sim;//The sum bound is not necessarily tighter

                            if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
                                count_survived_o_k_squar++;
                                sim = -solver.solve(col_maxima, window);//Note the minus-trick for the Hungarian
                                //normalize
                                sim /= k_double;

                                if(sim>=threshold) {
                                    count_cells_exceeding_threshold++;
                                    alignment_matrix_line.at(column) = sim;
                                }//else keep it zero
                                prior_cell_similarity = sim;

                            }else{
                                prior_cell_similarity = upper_bound_sim;
                            }
                            column_sum_correct = true;
                        }else{
                            prior_cell_similarity = upper_bound_sim;
                            column_sum_correct = false;
                        }
                        prev_min_value = max_column(window);
                        prior_cell_updated_matrix = true;
                    }else{
                        prior_cell_similarity = upper_bound_sim;
                        prior_cell_updated_matrix = false;
                        column_sum_correct = false;
                    }
                }
            }
        }

        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;

        double check_sum = sum(alignment_matrix);
        long size = alignment_matrix.size()*alignment_matrix.at(0).size();
        cout << "run_solution(k="<<k<<") time: " << "idx_time= "<< index_creation.count() << " time= " << time_elapsed.count() << "\tsum=" << check_sum << "\tsize=" <<  size << "\t |C|=" << count_candidates << "\t |O(1)|" << count_survived_o_1 << "\t |O(k)|" << count_survived_o_k << "\tO(k*k)" << count_survived_o_k_squar <<"\t"<< count_cells_exceeding_threshold << endl;
        cout << "sum(GCM)=" << sum(matrix_book) << endl;
        return time_elapsed.count();
    }

    void check_run_correctness(const int run_start, const int run_stop, const int line, HungarianKevinStern& solver, vector<vector<double>>& local_similarity_matrix, const vector<vector<double>>& matrix_book){
        //TODO use for correctness check
        for(int column=run_start;column<=run_stop;column++) {
            //Fill local matrix of the current window combination from global matrix
            fill_local_similarity_matrix(local_similarity_matrix, matrix_book, line, column);//das hier ist der einzige Unterschied
            const double upper_bound_sim = get_sum_of_column_row_minima(local_similarity_matrix);

            //if (upper_bound_sim + DOUBLE_PRECISION_BOUND >= threshold_times_k) {
            //That's the important line
            double similarity = -solver.solve_cached(local_similarity_matrix, threshold);
            similarity /= k_double;

            /*cout << "sim\t" << "bound\t" << "line\t" << "column" << endl;
            cout << similarity << "\t" << upper_bound_sim << "\t" << line << "\t" << column << endl;
            out_matrix(local_similarity_matrix);*/
            //normalize costs: Before it was distance. Now it is similarity.

            if (similarity >= threshold) {
                double sim_new = alignment_matrix.at(line).at(column);
                if(sim_new !=similarity){
                    cout << "sim_new !=similarity / k_double at line=" << line << " and column=" << column << " : "<< sim_new << " vs. " << similarity << endl;
                }
            }else{
                double sim_new = alignment_matrix.at(line).at(column);
                if(sim_new != 0){
                    cout << "sim_new != 0 at line=" << line << " and column=" << column << endl;
                }
            }
            //}
        }
    }

    double min(const vector<vector<double>>& current_lines) const {
        double min = current_lines.at(0).at(k-1);
        for(int line=1;line<k;line++) {
            if(min>current_lines.at(line).at(k-1)) {
                min=current_lines.at(line).at(k-1);
            }
        }
        return -min;
    }

    double min(const vector<const double*>& current_lines) const {
        double min = current_lines.at(0)[k-1];
        for(int line=1;line<k;line++) {
            if(min>current_lines.at(line)[k-1]) {
                min=current_lines.at(line)[k-1];
            }
        }
        return -min;
    }

    double max_column(vector<const double*>& current_lines) const {
        double max = -2.0;//TODO remove this line?
        for(auto& line : current_lines) {
            if(max<line[0]) {//similarity of the deleted token
                max=line[0];
            }
        }
        return -max;
    }

    double max_column(const vector<vector<double>>& current_lines) const {
        double max = -2.0;//TODO remove this line?
        for(auto& line : current_lines) {
            if(max<line.at(0)) {//similarity of the deleted token
                max=line.at(0);
            }
        }
        return -max;
    }

    //XXX does not use the injection of the col maxima
    double run_incremental_cell_pruning(){
        out_config("run_incremental_cell_pruning()");
        HungarianKevinStern solver(k);

        vector<vector<double>> local_similarity_matrix(k, vector<double>(k));
        double prior_cell_similarity;
        double prev_min_value;

        int count_survived_pruning = 0;
        int count_survived_second_pruning = 0;
        int count_survived_third_pruning = 0;
        int count_cells_exceeding_threshold = 0;
        bool prior_cell_updated_matrix;

        double ub_sum;
        double sim;

        const double MAX_SIM_ADDITION_NEW_NODE = 1.0/k_double;
        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

        //For each pair of windows
        for(int line=0;line<alignment_matrix.size();line++) {
            count_survived_pruning++;
            count_survived_second_pruning++;
            //get the line to get rid of 2D array resolution
            vector<double>& alignment_matrix_line = alignment_matrix.at(line);

            int column=0;
            {//Here we have no bound
                fill_local_similarity_matrix(k_with_windows_b1.at(line), k_with_windows_b2.at(column), local_similarity_matrix);
                ub_sum = sum_bound_similarity(local_similarity_matrix)/k_double;

                if(ub_sum+DOUBLE_PRECISION_BOUND>=threshold) {
                    sim = -solver.solve_cached(local_similarity_matrix, threshold);//Note the minus-trick for the Hungarian
                    sim /= k_double;
                    if(sim>=threshold) {
                        count_cells_exceeding_threshold++;
                        alignment_matrix_line.at(column) = sim;
                    }//else keep it zero
                    prior_cell_similarity = sim;
                }else{
                    prior_cell_similarity = ub_sum;
                }

                prev_min_value = max_val(local_similarity_matrix);
                prior_cell_updated_matrix = true;
            }

            //For all other columns: Here we have a bound
            for(column=1;column<alignment_matrix.at(0).size();column++) {
                double upper_bound_sim = prior_cell_similarity + MAX_SIM_ADDITION_NEW_NODE;
                if(prior_cell_updated_matrix) {
                    upper_bound_sim-= (prev_min_value / k_double);
                }

                if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
                    count_survived_pruning++;

                    fill_local_similarity_matrix(k_with_windows_b1.at(line), k_with_windows_b2.at(column), local_similarity_matrix);
                    double max_sim_new_node = min_vector(local_similarity_matrix);
                    upper_bound_sim-=MAX_SIM_ADDITION_NEW_NODE;
                    upper_bound_sim+=(max_sim_new_node/k);

                    if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
                        count_survived_second_pruning++;

                        ub_sum = sum_bound_similarity(local_similarity_matrix)/k_double;
                        upper_bound_sim = (ub_sum<upper_bound_sim) ? ub_sum : upper_bound_sim;//The some bound is not necessarily tighter

                        if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
                            count_survived_third_pruning++;
                            //That's the important line
                            sim = -solver.solve_cached(local_similarity_matrix, threshold);//Note the minus-trick for the Hungarian
                            //normalize
                            sim /= k;

                            if(sim>=threshold) {
                                count_cells_exceeding_threshold++;
                                alignment_matrix_line.at(column) = sim;
                            }//else keep it zero
                            prior_cell_similarity = sim;

                        }else{
                            prior_cell_similarity = upper_bound_sim;
                        }
                    }else{
                        prior_cell_similarity = upper_bound_sim;
                    }
                    prev_min_value = max_val(local_similarity_matrix);
                    prior_cell_updated_matrix = true;
                }else{
                    prior_cell_similarity = upper_bound_sim;
                    prior_cell_updated_matrix = false;
                }
            }
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;

        double check_sum = sum(alignment_matrix);
        long size = alignment_matrix.size()*alignment_matrix.at(0).size();
        cout << "run_incremental_cell_pruning() time: " << time_elapsed.count() << "\t" << check_sum << "\t" <<  size << "\t" << count_survived_pruning<< "\t" << count_survived_second_pruning<< "\t" << count_survived_third_pruning << "\t" << count_cells_exceeding_threshold << endl;

        return time_elapsed.count();
    }
};

#endif //PRANAY_TEST_SOLUTIONS_H
