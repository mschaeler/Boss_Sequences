//
// Created by Martin on 23.08.2023.
//

#ifndef PRANAY_TEST_SOLUTIONS_H
#define PRANAY_TEST_SOLUTIONS_H

#include <bitset>
#include <utility>
#include "HungarianKevinStern.h"

class MatrixRingBuffer {
public:
    vector<vector<double>> buffer;
    vector<double> col_maxima;
    int size;
    double col_sum;

    explicit MatrixRingBuffer(int k) : size(k), col_maxima(vector<double>(k)), buffer(k, vector<double>(k)), col_sum(0){

    }

    /**
     *
     * @param row
     * @param column
     * @param sim - materialized sim function
     * @param book_1
     * @param book_2
     */
    void fill(const int row, const int column, const vector<vector<double>>& sim, const vector<int>& book_1, const vector<int>& book_2) {
        for(int buffer_row=0;buffer_row<size;buffer_row++) {
            vector<double>& current_row = buffer[buffer_row];
            const int token_book_1 = book_1[row+buffer_row];
            for(int buffer_col=0;buffer_col<size;buffer_col++) {
                const int token_book_2 = book_2[column+buffer_col];
                current_row[(column+buffer_col)%size] = -sim[token_book_1][token_book_2];
            }
        }
    }
    void update(const int row, const int start_column, const vector<vector<double>>& sim, const vector<int>& book_1, const vector<int>& book_2) {
        const int token_offset_b2 = start_column+size-1;
        const int token_book_2 = book_2[token_offset_b2];
        const int buffer_index = (start_column-1)%size;

        for(int buffer_row=0;buffer_row<size;buffer_row++) {
            const int token_book_1 = book_1[row+buffer_row];
            buffer[buffer_row][buffer_index] = -sim[token_book_1][token_book_2];
        }
    }
    void update_with_bound(const int row, const int start_column, const vector<vector<double>>& sim, const vector<int>& book_1, const vector<int>& book_2) {
        const int token_offset_b2 = start_column+size-1;
        const int token_book_2 = book_2[token_offset_b2];
        const int buffer_index = (start_column-1)%size;

        const double old_col_max = col_maxima[buffer_index];
        double max = 20;//some big value

        for(int buffer_row=0;buffer_row<size;buffer_row++) {
            const int token_book_1 = book_1[row+buffer_row];
            double neg_similarity = -sim[token_book_1][token_book_2];
            if(neg_similarity<max) {
                max = neg_similarity;
            }
            buffer[buffer_row][buffer_index] = neg_similarity;
        }
        col_sum-=old_col_max;
        col_sum+=col_maxima[buffer_index]=max;
    }

    double get_sum_of_column_row_minima() {
        double row_sum = 0;
        std::fill(col_maxima.begin(), col_maxima.end(), 20);

        for(int i=0;i<size;i++) {
            const auto& line = buffer[i];
            double row_min = 20;
            for(int j=0;j<size;j++) {
                const double val = line[j];
                if(val<row_min) {
                    row_min = val;
                }
                if(val<col_maxima[j]){
                    col_maxima[j] = val;
                }
            }
            row_sum += row_min;
        }
        col_sum = sum(col_maxima);
        double max_similarity = -std::max(row_sum, col_sum);

        return max_similarity;
    }

    double o_k_square_bound() {
        double row_sum = 0;
        for(int i=0;i<size;i++) {
            const auto& line = buffer[i];
            double row_min = 20;
            for(int j=0;j<size;j++) {
                const double val = line[j];
                if(val<row_min) {
                    row_min = val;
                }
            }
            row_sum += row_min;
        }
        double max_similarity = -std::max(row_sum, col_sum);

        return max_similarity;
    }


    static double sum(const vector<double>& array) {
        double sum = 0;
        for(double d : array) {
            sum+=d;
        }
        return sum;
    }

    int get_offset(const int column) const {
        return column%size;
    }

    double min(const int column) {
        const int buffer_offset = get_offset(column+size-1);

        double min = buffer[0][buffer_offset];
        for(int line=1;line<size;line++) {
            if(min>buffer[line][buffer_offset]) {
                min=buffer[line][buffer_offset];
            }
        }
        return -min;
    }
    void out(){
        cout << "Buffer" << endl;
        for(const auto& arr : buffer){
            for(auto v : arr){
                cout << v << "\t";
            }
            cout << endl;
        }
    }

    double max(const int column) {
        const int buffer_offset = get_offset(column);

        double max = -20;//TODO remove this line?
        for(const auto& line : buffer) {
            if(max<line[buffer_offset]) {//similarity of the deleted token
                max=line[buffer_offset];
            }
        }
        return -max;
    }


    double col_max(const int column) {
        const int buffer_index = get_offset(column);
        return col_maxima[buffer_index];
    }
    void compare(const vector<vector<double>>& local_sim_matrix, const int index){
        for(int line=0;line<size;line++) {
            for(int column=0;column<size;column++) {
                int buffer_index = (index+column)%size;
                if(local_sim_matrix.at(line).at(column)!=buffer.at(line).at(buffer_index)) {
                    cout << "LSM" << endl;
                    for(const auto& arr : local_sim_matrix) {
                        for(auto v : arr){
                            cout << v << "\t";
                        }
                        cout << endl;
                    }
                    cout << "Buffer org" << endl;
                    for(const auto& arr : buffer) {
                        for(auto v : arr){
                            cout << v << "\t";
                        }
                        cout << endl;
                    }
                    cout << "Buffer rotated" << endl;
                    for(const auto& arr : buffer) {
                        for(int i=0;i<size;i++) {
                            cout << arr.at((index+i)%size) << "\t";
                        }
                        cout << endl;
                    }
                }
            }
        }
    }
};

class BitSet{
    /*
    * BitSets are packed into arrays of "words."  Currently a word is
    * a long, which consists of 64 bits, requiring 6 address bits.
    * The choice of word size is determined purely by performance concerns.
    */
    uint64_t ADDRESS_BITS_PER_WORD = 6;
    uint64_t BITS_PER_WORD = 1 << ADDRESS_BITS_PER_WORD;

    /* Used to shift left or right for a partial word mask */
    uint64_t WORD_MASK = 0xffffffffffffffffL;

    /**
     * The number of words in the logical size of this BitSet.
     */
    uint32_t wordsInUse = 0;

    /**
    * Ensures that the BitSet can accommodate a given wordIndex,
    * temporarily violating the invariants.  The caller must
    * restore the invariants before returning to the user,
    * possibly using recalculateWordsInUse().
    * @param wordIndex the index to be accommodated.
    */
    void expandTo(uint32_t wordIndex) {
        uint32_t wordsRequired = wordIndex+1;
        if (wordsInUse < wordsRequired) {
            wordsInUse = wordsRequired;
        }
    }

public:
    /**
     * The internal field corresponding to the serialField "bits".
     */
    vector<uint64_t> words;
    /**
     * Given a bit index, return word index containing it.
     */
    uint32_t wordIndex(uint32_t bitIndex) const {
        return bitIndex >> ADDRESS_BITS_PER_WORD;
    }
    /**
     * Creates a bit set whose initial size is large enough to explicitly
     * represent bits with indices in the range {@code 0} through
     * {@code nbits-1}. All bits are initially {@code false}.
     *
     * @param  nbits the initial size of the bit set
     * @throws NegativeArraySizeException if the specified initial size
     *         is negative
     */
    explicit BitSet(int nbits) : words(vector<uint64_t>(nbits)) {

    }

    /**
     * Sets the bit at the specified index to {@code true}.
     *
     * @param  bitIndex a bit index
     * @throws IndexOutOfBoundsException if the specified index is negative
     * @since  JDK1.0
     */
    void set(uint64_t bitIndex) {
        const uint64_t one = 1;
        uint64_t word_index = wordIndex(bitIndex);
        expandTo(word_index);
        uint64_t mask = (one << bitIndex);
        words.at(word_index) |= mask; // Restores invariants
    }

    /**
     * Sets the bits from the specified {@code fromIndex} (inclusive) to the
     * specified {@code toIndex} (exclusive) to {@code true}.
     *
     * @param  fromIndex index of the first bit to be set
     * @param  toIndex index after the last bit to be set
     * @throws IndexOutOfBoundsException if {@code fromIndex} is negative,
     *         or {@code toIndex} is negative, or {@code fromIndex} is
     *         larger than {@code toIndex}
     * @since  1.4
     */
    void set(uint32_t fromIndex, uint32_t toIndex) {
        if (fromIndex >= toIndex)
            return;

        // Increase capacity if necessary
        uint32_t startWordIndex = wordIndex(fromIndex);
        uint32_t endWordIndex   = wordIndex(toIndex - 1);
        expandTo(endWordIndex);

        uint64_t firstWordMask = WORD_MASK << fromIndex;
        uint64_t lastWordMask  = WORD_MASK >> -toIndex;
        //uint64_t lastWordMask  = WORD_MASK >>> -toIndex; //Java unsigned right shift
        if (startWordIndex == endWordIndex) {
            // Case 1: One word
            words.at(startWordIndex) |= (firstWordMask & lastWordMask);
        } else {
            // Case 2: Multiple words
            // Handle first word
            words.at(startWordIndex) |= firstWordMask;

            // Handle intermediate words, if any
            for (auto i = startWordIndex+1; i < endWordIndex; i++)
                words.at(i) = WORD_MASK;

            // Handle last word (restores invariants)
            words[endWordIndex] |= lastWordMask;
        }
    }

    /**
     * Returns the value of the bit with the specified index. The value
     * is {@code true} if the bit with the index {@code bitIndex}
     * is currently set in this {@code BitSet}; otherwise, the result
     * is {@code false}.
     *
     * @param  bitIndex   the bit index
     * @return the value of the bit with the specified index
     * @throws IndexOutOfBoundsException if the specified index is negative
     */
    bool get(uint64_t bitIndex) {
        const uint64_t one = 1;
        uint64_t word_index = wordIndex(bitIndex);

        return (word_index < wordsInUse)
               && ((words.at(word_index) & (one << bitIndex)) != 0);
    }
    /**
     * Returns the index of the first bit that is set to {@code true}
     * that occurs on or after the specified starting index. If no such
     * bit exists then {@code -1} is returned.
     *
     * <p>To iterate over the {@code true} bits in a {@code BitSet},
     * use the following loop:
     *
     *  <pre> {@code
     * for (int i = bs.nextSetBit(0); i >= 0; i = bs.nextSetBit(i+1)) {
     *     // operate on index i here
     * }}</pre>
     *
     * @param  fromIndex the index to start checking from (inclusive)
     * @return the index of the next set bit, or {@code -1} if there
     *         is no such bit
     * @throws IndexOutOfBoundsException if the specified index is negative
     * @since  1.4
     */
    uint32_t nextSetBit(uint64_t fromIndex) {
        uint64_t u = wordIndex(fromIndex);
        if (u >= wordsInUse)
            return -1;

        uint64_t word = words.at(u) & (WORD_MASK << fromIndex);
        while (true) {
            if (word != 0)
                return (u * BITS_PER_WORD) + __builtin_ctzll(word);//Long.numberOfTrailingZeros(word);//TODO
            if (++u == wordsInUse)
                return -1;
            word = words[u];
        }
    }

    /**
     * Returns the index of the first bit that is set to {@code false}
     * that occurs on or after the specified starting index.
     *
     * @param  fromIndex the index to start checking from (inclusive)
     * @return the index of the next clear bit
     * @throws IndexOutOfBoundsException if the specified index is negative
     * @since  1.4
     */
    uint32_t nextClearBit(uint32_t fromIndex) {
        // Neither spec nor implementation handle bitsets of maximal length.
        // See 4816253.
        uint32_t u = wordIndex(fromIndex);
        if (u >= wordsInUse)
            return fromIndex;

        uint64_t word = ~words[u] & (WORD_MASK << fromIndex);

        while (true) {
            if (word != 0)
                return (u * BITS_PER_WORD) + __builtin_ctzll(word);
            if (++u == wordsInUse)
                return wordsInUse * BITS_PER_WORD;
            word = ~words[u];
        }
    }

    void logic_or(const vector<BitSet>& all_sets, const vector<int>& ids) {
        for(int id : ids) {
            if(wordsInUse<all_sets[id].wordsInUse) {
                wordsInUse = all_sets[id].wordsInUse;
            }
        }

        // Perform logical OR on words in common
        for (int i = 0; i < wordsInUse; i++) {
            for(int id : ids) {
                words[i] |= all_sets[id].words[i];
            }
        }
    }
};

/**
 * At Book granularity
 */
class Solutions{
    const double DOUBLE_PRECISION_BOUND = 0.0001;
    const double MAX_DOUBLE = 10000;

    const int k;
    const double k_double;
    const double threshold;
    const double threshold_times_k;

    const vector<int> book_1;
    const vector<int> book_2;

    vector<double> col_maxima;

    vector<vector<int>> k_with_windows_b1;
    vector<vector<int>> k_with_windows_b2;

    vector<int> tokens_b1;
    vector<int> tokens_b2;

    const vector<vector<double>> global_similarity_matrix;
    const vector<vector<double>> word_vectors;
    vector<vector<double>> book_matrix;
    vector<vector<double>> alignment_matrix;
    double sum_cols = 0;

    const double MAX_SIM_ADDITION_NEW_NODE;

    /**
	 *
	 * @param raw_paragraphs all the paragraphs
	 * @param k - window size
	 * @return
	 */
    static vector<vector<int>> create_windows(vector<int> book, int k) {
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
    static double sum(const vector<vector<double>>& matrix) {
        double sum = 0;
        for(const vector<double>& arr : matrix){
            for(double d : arr){
                sum+=d;
            }
        }
        return sum;
    }
    static double sum(const vector<double>& arr) {
        double sum = 0;
        for(double d : arr){
            sum+=d;
        }
        return sum;
    }
    void out_config(const string& name) const{
        cout << "Solutions "<<name<<" k=" << k << " threshold=" << threshold << " " << threshold_times_k << endl;
    }
    void fill_similarity_matrix() {
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
        cout << "GCM materialized in " << time_elapsed.count() << endl;
    }

    /**
     *
     * @return -sim[][]
     */
    void fill_similarity_matrix_deep() {
        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        for(int line=0;line<book_1.size();line++) {
            const int set_id_window_p1 = book_1[line];
            const vector<double>& sim_matrix_line = global_similarity_matrix[set_id_window_p1];
            for(int column=0;column<book_2.size();column++) {
                const int set_id_window_p2 = book_2[column];
                const double sim = sim_matrix_line[set_id_window_p2];
                book_matrix[line][column] = -sim;// XXX this is the difference to the method above
            }
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;
        cout << "-GCM materialized in " << time_elapsed.count() << endl;
    }

    void fill_local_similarity_matrix(vector<vector<double>>& local_cost_matrix, const vector<vector<double>>& global_cost_matrix_book, const int line, const int column) const {
        for(int i=0;i<k;i++) {
            for(int j=0;j<k;j++) {
                local_cost_matrix.at(i).at(j) = -global_cost_matrix_book.at(line+i).at(column+j);//XXX - Note the minus for the Hungarian
            }
        }
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
        sum_cols = sum(col_maxima);
        double max_similarity = -max(row_sum, sum_cols);

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

    void create_indexes_bit_vectors(vector<BitSet>& inverted_window_index_bit_set) const{
        vector<vector<int>> indexes;
        //find for each set all other sets such that sim(set,other_set)>=threshold
        for(int token_id : tokens_b1){
            const vector<double>& line = global_similarity_matrix[token_id];
            vector<int> index;
            for(int id : tokens_b2){
                const double sim = line[id];
                if(sim>=threshold){
                    index.push_back(id);
                }
            }
            indexes.push_back(index);
        }

        //For each token
        for(int token_id : tokens_b1) {
            /**
             * The list of all tokens with sim > threshold
             */
            const vector<int>& neighborhood_index = indexes[token_id];
            //vector<bool>& bit_vector = inverted_window_index[token_id];
            BitSet& my_set = inverted_window_index_bit_set[token_id];

            for(int pos=0;pos<book_2.size();pos++) {
                const int token_id_in_b2 = book_2[pos];

                if(isIn(neighborhood_index,token_id_in_b2)) {
                    int start = max(0, pos-k+1);
                    auto stop =  (k_with_windows_b2.size()-1 < pos) ? k_with_windows_b2.size()-1 : pos;
                    my_set.set(start,stop+1);                }
            }
        }
    }

    /**
     * O(n)
     * @param value
     * @return
     */
    static bool isIn(const vector<int>& neighborhood_index, const int value) {
        const int size = neighborhood_index.size();
        for(int i=0;i<size;i++) {
            if(neighborhood_index[i]==value) {
                return true;
            }
        }
        return false;
    }

    static vector<int> get_tokens(const vector<int>& book) {
        unordered_set<int> temp;
        for(int id : book){
            temp.insert(id);
        }
        vector<int> ret;
        ret.reserve(temp.size());
        for(auto v : temp){
            ret.push_back(v);
        }
        sort(ret.begin(), ret.end());

        return ret;
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

    static double max_column(vector<const double*>& current_lines) {
        double max = -2.0;//some very small value
        for(auto& line : current_lines) {
            if(max<line[0]) {//similarity of the deleted token
                max=line[0];
            }
        }
        return -max;
    }

    double cosine_unit_length(const vector<double>& vec_1, const vector<double>& vec_2) const {
        return 5;
        double dotProduct = 0.0;
        for (int i = 0; i < vec_1.size(); i++) {
            dotProduct += vec_1[i] * vec_2[i];
        }

        dotProduct = (dotProduct < 0) ? 0 : dotProduct;
        dotProduct = (dotProduct > 1) ? 1 : dotProduct;
        return dotProduct;
    }

public:

    Solutions(int _k, double _threshold, vector<int> _book_1, vector<int> _book_2, vector<vector<double>> _cost_matrix, vector<vector<double>> _word_vectors) :
            k(_k)
            , k_double((double)_k)
            , threshold_times_k(_threshold*_k)
            , threshold(_threshold)
            , book_1(std::move(_book_1))
            , book_2(std::move(_book_2))
            , global_similarity_matrix(std::move(_cost_matrix))
            , col_maxima(vector<double>(k))
            , MAX_SIM_ADDITION_NEW_NODE(1.0 / k_double)
            , book_matrix(book_1.size(), vector<double>(book_2.size()))
            , word_vectors(std::move(_word_vectors))
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
    }

    double run_naive_rb(){
        out_config("run_naive_rb()");
        long count_computed_cells = 0;
        HungarianKevinStern solver(k);

        //vector<vector<double>> local_similarity_matrix(k, vector<double>(k));
        MatrixRingBuffer mrb(k);
        //USE_GLOBAL_MATRIX = false;

        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

        //fill_similarity_matrix();
        //For each pair of windows
        for(int line=0;line<alignment_matrix.size();line++) {
            mrb.fill(line, 0, global_similarity_matrix, book_1, book_2);
            for(int column=0;column<alignment_matrix.at(0).size();column++) {
                //Fill local matrix of the current window combination from global matrix
                //fill_local_similarity_matrix(local_similarity_matrix, book_matrix, line, column);
                if(column!=0) {
                    mrb.update(line, column, global_similarity_matrix, book_1, book_2);
                }
                //mrb.compare(local_similarity_matrix, column);

                //That's the important line
                //const double similarity = -solver.solve_cached(local_similarity_matrix);
                const double similarity = -solver.solve_cached(mrb.buffer);
                //if(abs(similarity-similarity_rb)>DOUBLE_PRECISION_BOUND){
                //    cout << "abs(similarity-similarity_rb)>DOUBLE_PRECISION_BOUND" << endl;
                //}
                //normalize costs: Before it was distance. Now it is similarity.
                if(similarity>=threshold_times_k) {
                    alignment_matrix.at(line).at(column) = similarity/(double)k;//normalize
                    count_computed_cells++;
                }//else keep it zero
            }
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;

        double check_sum = sum(alignment_matrix);
        auto size = alignment_matrix.size()*alignment_matrix.at(0).size();
        cout << "run_naive() time: " << time_elapsed.count() << "\t" << check_sum << "\t" <<  size << "\t" << count_computed_cells << endl;

        return time_elapsed.count();
    }

    double run_baseline_rb() {
        out_config("run_baseline_rb()");
        long count_computed_cells = 0;
        long count_survived_pruning = 0;
        HungarianKevinStern solver(k);
        MatrixRingBuffer mrb(k);

        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();


        //For each pair of windows
        for (int line = 0; line < alignment_matrix.size(); line++) {
            mrb.fill(line, 0, global_similarity_matrix, book_1, book_2);
            for (int column = 0; column < alignment_matrix.at(0).size(); column++) {
                //Fill local matrix of the current window combination from global matrix
                if(column!=0) {
                    mrb.update(line, column, global_similarity_matrix, book_1, book_2);
                }
                const double upper_bound_sim = mrb.get_sum_of_column_row_minima();


                if (upper_bound_sim + DOUBLE_PRECISION_BOUND >= threshold_times_k) {
                    count_survived_pruning++;
                    //That's the important line
                    const double similarity = -solver.solve_cached(mrb.buffer);
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
        auto size = alignment_matrix.size() * alignment_matrix.at(0).size();
        cout << "run_baseline_rb() time: " << time_elapsed.count() << "\t" << check_sum << "\t" << size << "\t"
             << count_survived_pruning << "\t" << count_computed_cells << endl;

        return time_elapsed.count();
    }

    double run_fast_text(){
        out_config("run_fast_text()");
        long count_computed_cells = 0;

        int vector_size = word_vectors.at(0).size();
        vector<double> avg_vec_window_line(vector_size);
        vector<vector<double>> averaged_colum_vectors(k_with_windows_b2.size(), vector<double>(vector_size));
        //USE_GLOBAL_MATRIX = false;

        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        for(int column=0;column<k_with_windows_b2.size();column++) {
            //cout << " "<< column;
            const auto& window_b2 = k_with_windows_b2.at(column);
            vector<double>& my_avg_vec = averaged_colum_vectors.at(column);
            for(int token : window_b2){
                if(token<word_vectors.size()){
                    const vector<double>& my_vector = word_vectors.at(token);
                    for(int dim=0;dim<my_vector.size();dim++){
                        my_avg_vec.at(dim) += my_vector.at(dim);//sum up the vector
                    }
                }
            }
            //normalize to unit length: (1) get length
            double length = 0;
            for(double value : my_avg_vec){
                length+=(value*value);
            }
            length = sqrt(length);
            //normalize to unit length: (2) normalize by length
            for(int dim=0;dim<my_avg_vec.size();dim++){
                my_avg_vec.at(dim) /= length;
            }
        }
        //cout << "Done vector" << endl;
        chrono::time_point<std::chrono::high_resolution_clock> stop_avg_vectors = std::chrono::high_resolution_clock::now();
        //For each pair of windows
        for(int line=0;line<alignment_matrix.size();line++) {
            {//get the average vector of window_b1
                for(int token : k_with_windows_b1.at(line)){
                    const vector<double>& my_vector = word_vectors.at(token);
                    for(int dim=0;dim<my_vector.size();dim++){
                        avg_vec_window_line.at(dim) += my_vector.at(dim);//sum up the vector
                    }
                }
                //normalize to unit length: (1) get length
                double length = 0;
                for(double value : avg_vec_window_line){
                    length+=(value*value);
                }
                length = sqrt(length);
                //normalize to unit length: (2) normalize by length
                for(int dim=0;dim<avg_vec_window_line.size();dim++){
                    avg_vec_window_line.at(dim) /= length;
                }
            }
            for(int column=0;column<alignment_matrix.at(0).size();column++) {
                const vector<double>& avg_vec_window_column = averaged_colum_vectors.at(column);

                //That's the important line
                const double similarity = cosine_unit_length(avg_vec_window_line,avg_vec_window_column);
                //normalize costs: Before it was distance. Now it is similarity.
                if(similarity>=threshold_times_k) {
                    alignment_matrix[line][column] = similarity/(double)k;//normalize
                    count_computed_cells++;
                }//else keep it zero
            }
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;
        chrono::duration<double> time_elapsed_avg = stop_avg_vectors - start;

        double check_sum = sum(alignment_matrix);
        auto size = alignment_matrix.size()*alignment_matrix.at(0).size();
        cout << "run_fast_text() time: " << time_elapsed.count() << "\t" << check_sum << "\t" <<  size << "\t" << count_computed_cells << "\t"<< time_elapsed_avg.count()<<endl;

        return time_elapsed.count();
    }


    //XXX this one does not compute the distances on the fly. Add time?
    double run_naive(){
        out_config("run_naive()");
        long count_computed_cells = 0;
        HungarianKevinStern solver(k);

        vector<vector<double>> local_similarity_matrix(k, vector<double>(k));
        //USE_GLOBAL_MATRIX = false;

        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

        fill_similarity_matrix();
        //For each pair of windows
        for(int line=0;line<alignment_matrix.size();line++) {
            for(int column=0;column<alignment_matrix.at(0).size();column++) {
                //Fill local matrix of the current window combination from global matrix
                fill_local_similarity_matrix(local_similarity_matrix, book_matrix, line, column);
                //That's the important line
                const double similarity = -solver.solve_cached(local_similarity_matrix);
                //normalize costs: Before it was distance. Now it is similarity.
                if(similarity>=threshold_times_k) {
                    alignment_matrix.at(line).at(column) = similarity/(double)k;//normalize
                    count_computed_cells++;
                }//else keep it zero
            }
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;

        double check_sum = sum(alignment_matrix);
        auto size = alignment_matrix.size()*alignment_matrix.at(0).size();
        cout << "run_naive() time: " << time_elapsed.count() << "\t" << check_sum << "\t" <<  size << "\t" << count_computed_cells << endl;

        return time_elapsed.count();
    }

    double run_baseline() {
        out_config("run_baseline()");
        long count_computed_cells = 0;
        long count_survived_pruning = 0;
        HungarianKevinStern solver(k);

        vector<vector<double>> local_similarity_matrix(k, vector<double>(k));

        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

        fill_similarity_matrix();
        //For each pair of windows
        for (int line = 0; line < alignment_matrix.size(); line++) {
            for (int column = 0; column < alignment_matrix.at(0).size(); column++) {
                //Fill local matrix of the current window combination from global matrix
                fill_local_similarity_matrix(local_similarity_matrix, book_matrix, line, column);//This is the only difference
                const double upper_bound_sim = get_sum_of_column_row_minima(local_similarity_matrix);

                if (upper_bound_sim + DOUBLE_PRECISION_BOUND >= threshold_times_k) {
                    count_survived_pruning++;
                    //That's the important line
                    double similarity = -solver.solve_cached(local_similarity_matrix);
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
        auto size = alignment_matrix.size() * alignment_matrix.at(0).size();
        cout << "run_baseline() time: " << time_elapsed.count() << "\t" << check_sum << "\t" << size << "\t"
             << count_survived_pruning << "\t" << count_computed_cells << endl;

        return time_elapsed.count();
    }

    double run_solution(){
        out_config("run_solution()");
        HungarianDeep solver(k);
        /**
         * Indicates for token i whether the corresponding windows of the other sequence is a candidate.
         */
        //vector<vector<bool>> inverted_window_index(global_similarity_matrix.size(), vector<bool>(k_with_windows_b2.size()));
        vector<BitSet> inverted_window_index_bit_set(global_similarity_matrix.size(), BitSet(k_with_windows_b2.size()));
        //Not needed later
        vector<const double*> window(k);//Can't use a vector to point into an existing buffer.
        fill_similarity_matrix_deep();
        vector<BitSet> all_bit_candidates(k_with_windows_b1.size(), BitSet(k_with_windows_b2.size()));

        long count_candidates = 0;
        long count_survived_o_1 = 0;
        long count_survived_o_k = 0;
        long count_survived_o_k_square = 0;
        long count_cells_exceeding_threshold = 0;

        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        create_indexes_bit_vectors(inverted_window_index_bit_set);
        chrono::duration<double> index_generation = std::chrono::high_resolution_clock::now() - start;

        //Check candidate runs
        for(int line=0;line<alignment_matrix.size();line++) {
            vector<double>& alignment_matrix_line = alignment_matrix[line];

            const vector<int>& window_b1 = k_with_windows_b1[line];
            BitSet& my_candidates = all_bit_candidates.at(line);
            my_candidates.logic_or(inverted_window_index_bit_set, window_b1);

            //Manually inlined condense transforms the bit vector into runs of candidates
            vector<int> candidates_condensed_bit_set;
            uint32_t start_alt = 0, stop_alt;

            while((start_alt = my_candidates.nextSetBit(start_alt))!=-1) {
                stop_alt = my_candidates.nextClearBit(start_alt);
                candidates_condensed_bit_set.push_back(start_alt);
                candidates_condensed_bit_set.push_back(stop_alt-1);
                start_alt = stop_alt;
            }

            const vector<int>& candidates_condensed = candidates_condensed_bit_set;

            const int size = candidates_condensed.size();
            for(int c=0;c<size;c+=2) {//Contains start and stop index. Thus, c+=2.
                const int run_start = candidates_condensed[c];
                const int run_stop  = candidates_condensed[c+1];

                double ub_sum, sim, prior_cell_similarity, prev_min_value;
                bool prior_cell_updated_matrix, column_sum_correct;

                count_candidates+=run_stop-run_start+1;
                int column=run_start;
                {//First element in run: Here we have no O(1) bound
                    count_survived_o_1++;
                    count_survived_o_k++;
                    for(int i = 0;i<k;i++){//Init sliding window
                        const double* temp = &book_matrix[line+i][column];
                        window[i] = temp;
                    }
                    ub_sum = o_k_square_bound(window) / k_double;

                    if(ub_sum+DOUBLE_PRECISION_BOUND>=threshold) {
                        count_survived_o_k_square++;
                        sim = -solver.solve(col_maxima, window);//Note the minus-trick for the Hungarian
                        sim /= k_double;
                        if(sim>=threshold) {
                            count_cells_exceeding_threshold++;
                            //if(LOGGING_MODE) count_cells_exceeding_threshold++;
                            alignment_matrix_line[column] = sim;
                        }//else keep it zero
                        prior_cell_similarity = sim;

                    }else{
                        prior_cell_similarity = ub_sum;
                    }
                    prev_min_value = max_column(window);
                    prior_cell_updated_matrix = true;
                    column_sum_correct = true;
                }//END first element in run

                //For all other columns: Here we have a O(1) and O(k) bound
                for(column=run_start+1;column<=run_stop;column++) {
                    for(int i = 0;i<k;i++){//Init sliding window
                        //const double* temp = &matrix_book[line+i][column];
                        window[i]++;// = temp;
                    }

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
                            sum_cols -= col_maxima[0];
                            sum_cols -= max_sim_new_node;//is not negated
                            double temp = -sum_cols / k_double;

                            if(temp<upper_bound_sim) {
                                upper_bound_sim = temp;
                            }
                        }

                        if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
                            count_survived_o_k++;
                            ub_sum = o_k_square_bound(window) / k_double;
                            //The sum bound is not necessarily tighter, we need the tightest bound for bound cascade of the *next* window
                            upper_bound_sim = (ub_sum<upper_bound_sim) ? ub_sum : upper_bound_sim;

                            if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
                                count_survived_o_k_square++;
                                sim = -solver.solve(col_maxima, window);//Note the minus-trick for the Hungarian
                                //normalize
                                sim /= k_double;

                                if(sim>=threshold) {
                                    count_cells_exceeding_threshold++;
                                    alignment_matrix_line[column] = sim;
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
        auto size = alignment_matrix.size()*alignment_matrix.at(0).size();
        cout << "run_solution(k=" << k << ") time: " << time_elapsed.count() << " idx_gen= " << index_generation.count() << " time= " << time_elapsed.count() << "\tsum=" << check_sum << "\tsize=" << size << "\t |C|=" << count_candidates << "\t |O(1)|" << count_survived_o_1 << "\t |O(k)|" << count_survived_o_k << "\tO(k*k)" << count_survived_o_k_square << "\t" << count_cells_exceeding_threshold << endl;
        return time_elapsed.count();
    }

    double run_solution_rb(){
        out_config("run_solution_rb()");
        HungarianDeep_2 solver(k);
        MatrixRingBuffer mrb(k);
        /**
         * Indicates for token i whether the corresponding windows of the other sequence is a candidate.
         */
        //vector<vector<bool>> inverted_window_index(global_similarity_matrix.size(), vector<bool>(k_with_windows_b2.size()));
        vector<BitSet> inverted_window_index_bit_set(global_similarity_matrix.size(), BitSet(k_with_windows_b2.size()));
        //Not needed later
        //vector<const double*> window(k);//Can't use a vector to point into an existing buffer.
        //fill_similarity_matrix_deep();
        vector<BitSet> all_bit_candidates(k_with_windows_b1.size(), BitSet(k_with_windows_b2.size()));

        long count_candidates = 0;
        long count_survived_o_1 = 0;
        long count_survived_o_k = 0;
        long count_survived_o_k_square = 0;
        long count_cells_exceeding_threshold = 0;

        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        create_indexes_bit_vectors(inverted_window_index_bit_set);
        chrono::duration<double> index_generation = std::chrono::high_resolution_clock::now() - start;

        //Check candidate runs
        for(int line=0;line<alignment_matrix.size();line++) {
            vector<double>& alignment_matrix_line = alignment_matrix[line];

            const vector<int>& window_b1 = k_with_windows_b1[line];
            BitSet& my_candidates = all_bit_candidates.at(line);
            my_candidates.logic_or(inverted_window_index_bit_set, window_b1);

            //Manually inlined condense transforms the bit vector into runs of candidates
            vector<int> candidates_condensed_bit_set;
            uint32_t start_alt = 0, stop_alt;

            while((start_alt = my_candidates.nextSetBit(start_alt))!=-1) {
                stop_alt = my_candidates.nextClearBit(start_alt);
                candidates_condensed_bit_set.push_back(start_alt);
                candidates_condensed_bit_set.push_back(stop_alt-1);
                start_alt = stop_alt;
            }

            const vector<int>& candidates_condensed = candidates_condensed_bit_set;

            const int size = candidates_condensed.size();
            for(int c=0;c<size;c+=2) {//Contains start and stop index. Thus, c+=2.
                const int run_start = candidates_condensed[c];
                const int run_stop  = candidates_condensed[c+1];

                double ub_sum, sim, prior_cell_similarity, prev_min_value;

                count_candidates+=run_stop-run_start+1;
                int column=run_start;
                {//First element in run: Here we have no O(1) bound
                    count_survived_o_1++;
                    count_survived_o_k++;
                    mrb.fill(line, column, global_similarity_matrix, book_1, book_2);
                    ub_sum = mrb.get_sum_of_column_row_minima() / k_double;

                    if(ub_sum+DOUBLE_PRECISION_BOUND>=threshold) {
                        count_survived_o_k_square++;
                        sim = -solver.solve(mrb.col_maxima, mrb.buffer);//Note the minus-trick for the Hungarian
                        sim /= k_double;
                        if(sim>=threshold) {
                            count_cells_exceeding_threshold++;
                            //if(LOGGING_MODE) count_cells_exceeding_threshold++;
                            alignment_matrix_line[column] = sim;
                        }//else keep it zero
                        prior_cell_similarity = sim;

                    }else{
                        prior_cell_similarity = ub_sum;
                    }
                    prev_min_value = mrb.max(column);
                }//END first element in run

                //For all other columns: Here we have a O(1) and O(k) bound
                for(column=run_start+1;column<=run_stop;column++) {
                    mrb.update_with_bound(line, column, global_similarity_matrix, book_1, book_2);

                    double upper_bound_sim = prior_cell_similarity + MAX_SIM_ADDITION_NEW_NODE;// O(1) bound
                    upper_bound_sim-= (prev_min_value / k_double);// (1) O(k) bound : part of the O(k) bound in case the prior cell updated the matrix, i.e., we know the minimum similarity of the leaving node

                    if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
                        count_survived_o_1++;

                        double max_sim_new_node = mrb.min(column);//(2) O(k) bound
                        upper_bound_sim-=MAX_SIM_ADDITION_NEW_NODE;
                        upper_bound_sim+=(max_sim_new_node/k_double);

                        //mrb.out();//TODO remove me

                        double temp = -mrb.col_sum / k_double;//FIXME

                        if(temp<upper_bound_sim) {
                            upper_bound_sim = temp;
                        }

                        if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
                            count_survived_o_k_square++;
                            sim = -solver.solve(col_maxima, mrb.buffer);//Note the minus-trick for the Hungarian
                            //normalize
                            sim /= k_double;

                            if(sim>=threshold) {
                                count_cells_exceeding_threshold++;
                                alignment_matrix_line[column] = sim;
                            }//else keep it zero
                            upper_bound_sim = sim;//TODO
                        }
                    }
                    prev_min_value = mrb.max(column);
                    prior_cell_similarity = upper_bound_sim;
                }
            }
        }

        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;

        double check_sum = sum(alignment_matrix);
        auto size = alignment_matrix.size()*alignment_matrix.at(0).size();
        cout << "run_solution(k=" << k << ") time: " << time_elapsed.count() << " idx_gen= " << index_generation.count() << " time= " << time_elapsed.count() << "\tsum=" << check_sum << "\tsize=" << size << "\t |C|=" << count_candidates << "\t |O(1)|" << count_survived_o_1 << "\t |O(k)|" << count_survived_o_k << "\tO(k*k)" << count_survived_o_k_square << "\t" << count_cells_exceeding_threshold << endl;
        return time_elapsed.count();
    }
};

#endif //PRANAY_TEST_SOLUTIONS_H
