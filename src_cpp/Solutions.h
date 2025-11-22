//
// Created by Martin on 23.08.2023.
//

#ifndef PRANAY_TEST_SOLUTIONS_H
#define PRANAY_TEST_SOLUTIONS_H

#include <utility>
#include "HungarianKevinStern.h"

constexpr int run_naive = 0;
constexpr int run_basem = 1;
constexpr int run_seda  = 2;

constexpr int run_naive_rb = 3;
constexpr int run_basem_rb = 4;
constexpr int run_seda_rb  = 5;
constexpr int run_fast_text = 6;

constexpr int run_c_seda = 7;
constexpr int run_c_seda_2 = 8;

bool verbose = true;

int global_query_id;
int global_article_id;

class MatrixRingBuffer {
public:
    vector<vector<double>> buffer;
    vector<double> col_maxima;
    int size;
    double col_sum;

    explicit MatrixRingBuffer(int k) : buffer(k, vector<double>(k)), col_maxima(vector<double>(k)), size(k), col_sum(0){

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

    double o_k_square_bound() const {
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

    double min(const int column) const {
        const int buffer_offset = get_offset(column+size-1);

        double min = buffer[0][buffer_offset];
        for(int line=1;line<size;line++) {
            if(min>buffer[line][buffer_offset]) {
                min=buffer[line][buffer_offset];
            }
        }
        return -min;
    }
    void out() const {
        cout << "Buffer" << endl;
        for(const auto& arr : buffer){
            for(auto v : arr){
                cout << v << "\t";
            }
            cout << endl;
        }
    }

    double max(const int column) const {
        const int buffer_offset = get_offset(column);

        double max = -20;//TODO remove this line?
        for(const auto& line : buffer) {
            if(max<line[buffer_offset]) {//similarity of the deleted token
                max=line[buffer_offset];
            }
        }
        return -max;
    }


    double col_max(const int column) const {
        const int buffer_index = get_offset(column);
        return col_maxima[buffer_index];
    }
    void compare(const vector<vector<double>>& local_sim_matrix, const int index) const {
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
    * BitSets are packed into arrays of "words."  Currently, a word is
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
     * represent bits with indices in the range 0 through
     * nbits-1. All bits are initially false.
     *
     * @param  nbits the initial size of the bit set
     * @throws NegativeArraySizeException if the specified initial size
     *         is negative
     */
    explicit BitSet(int nbits) : words(vector<uint64_t>(nbits)) {

    }

    /**
     * Sets the bit at the specified index to true.
     *
     * @param  bitIndex a bit index
     */
    void set(uint64_t bitIndex) {
        constexpr uint64_t one = 1;
        uint64_t word_index = wordIndex(bitIndex);
        expandTo(word_index);
        uint64_t mask = (one << bitIndex);
        words.at(word_index) |= mask; // Restores invariants
    }

    /**
     * Sets the bits from the specified fromIndex (inclusive) to the
     * specified toIndex (exclusive) to true.
     *
     * @param  fromIndex index of the first bit to be set
     * @param  toIndex index after the last bit to be set
     * @throws IndexOutOfBoundsException if fromIndex is negative,
     *         or toIndex is negative, or fromIndex is
     *         larger than toIndex
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
     * is true if the bit with the index bitIndex
     * is currently set in this BitSet; otherwise, the result
     * is false.
     *
     * @param  bitIndex   the bit index
     * @return the value of the bit with the specified index
     * @throws IndexOutOfBoundsException if the specified index is negative
     */
    bool get(const uint64_t bitIndex) const {
        constexpr uint64_t one = 1;
        uint64_t word_index = wordIndex(bitIndex);

        return (word_index < wordsInUse)
               && ((words.at(word_index) & (one << bitIndex)) != 0);
    }
    /**
     * Returns the index of the first bit that is set to {true}
     * that occurs on or after the specified starting index. If no such
     * bit exists then {-1} is returned.
     *
     * <p>To iterate over the {true} bits in a { BitSet},
     * use the following loop:
     *
     *  <pre>
     * for (int i = bs.nextSetBit(0); i >= 0; i = bs.nextSetBit(i+1)) {
     *     // operate on index i here
     * }</pre>
     *
     * @param  fromIndex the index to start checking from (inclusive)
     * @return the index of the next set bit, or {-1} if there
     *         is no such bit
     * @throws IndexOutOfBoundsException if the specified index is negative
     * @since  1.4
     */
    uint32_t nextSetBit(const uint64_t fromIndex) const {
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
     * Returns the index of the first bit that is set to false
     * that occurs on or after the specified starting index.
     *
     * @param  fromIndex the index to start checking from (inclusive)
     * @return the index of the next clear bit
     * @throws IndexOutOfBoundsException if the specified index is negative
     * @since  1.4
     */
    uint32_t nextClearBit(const uint32_t fromIndex) const {
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

    void logic_or(const BitSet& other) {
        if (other.wordsInUse > wordsInUse) {
            wordsInUse = other.wordsInUse;
        }
        for (int i = 0; i < wordsInUse; i++) {
            words[i] |= other.words[i];
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
        if(verbose) cout << "GCM materialized in " << time_elapsed.count() << endl;
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
        if (verbose) cout << "-GCM materialized in " << time_elapsed.count() << endl;
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

        //For each token
        for(int i=0;i<tokens_b1.size();i++) {
            int token_id = tokens_b1.at(i);
            /**
             * The list of all tokens with sim > threshold
             */
            const vector<int>& neighborhood_index = indexes.at(i);//XXX the push_back above
            //vector<bool>& bit_vector = inverted_window_index[token_id];
            BitSet& my_set = inverted_window_index_bit_set.at(token_id);

            for(int pos=0;pos<book_2.size();pos++) {
                const int token_id_in_b2 = book_2.at(pos);

                if(isIn(neighborhood_index,token_id_in_b2)) {//TODO set, not vector to avoid is in
                    const uint32_t start = static_cast<uint32_t>(max(0, pos - k + 1));
                    const auto stop =   static_cast<uint32_t>((k_with_windows_b2.size()-1 < pos) ? k_with_windows_b2.size()-1 : pos);
                    //FIXME my_set.set(start,stop+1);
                    //FIXME so gehts my_set.set(start);
                    for (uint32_t bit=start;bit<stop+1;bit++) {
                        my_set.set(bit);
                    }

                    /*if (global_article_id==1 && global_query_id == 0 && token_id==62 && token_id_in_b2==47) {
                        cout << "****************** create_indexes_bit_vectors()" << endl;
                        cout << neighborhood_index.size() << endl;
                        if (!neighborhood_index.empty()) {
                            cout << neighborhood_index.at(0) << endl;
                        }
                        cout << my_set.words.at(0) << endl;
                        cout << "******************" << endl;
                        cout << start << endl;
                        cout << stop << endl;
                        //exit(0);
                    }*/
                }
            }
        }
    }

    /**
     * O(n)
     * @param neighborhood_index
     * @param value
     * @return
     */
    static bool isIn(const vector<int>& neighborhood_index, const int value) {
        for(const int i : neighborhood_index) {
            if(i==value) {
                return true;
            }
        }
        return false;
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

    static double max_column(const vector<const double*>& current_lines) {
        double max = -2.0;//some very small value
        for(auto& line : current_lines) {
            if(max<line[0]) {//similarity of the deleted token
                max=line[0];
            }
        }
        return -max;
    }

    static double cosine_unit_length(const vector<double>& vec_1, const vector<double>& vec_2) {
        double dotProduct = 0.0;
        for (int i = 0; i < vec_1.size(); i++) {
            dotProduct += vec_1[i] * vec_2[i];
        }

        dotProduct = (dotProduct < 0) ? 0 : dotProduct;
        dotProduct = (dotProduct > 1) ? 1 : dotProduct;
        return dotProduct;
    }

public:
    /**
     * Constructor for corpus level experiments
     * @param _k
     * @param _threshold
     * @param _book_1
     * @param _book_2
     * @param _cost_matrix
     * @param _k_with_windows_b1
     * @param _k_with_windows_b2
     */
    Solutions(int _k, double _threshold, vector<int> _book_1, vector<int> _book_2
              , vector<vector<double>> _cost_matrix, vector<vector<int>> _k_with_windows_b1, vector<vector<int>> _k_with_windows_b2) :
                k(_k)
                , k_double(static_cast<double>(_k))
                , threshold(_threshold)
                , threshold_times_k(_threshold*_k)
                , book_1(std::move(_book_1))
                , book_2(std::move(_book_2))
                , col_maxima(vector<double>(k))
                , k_with_windows_b1(std::move(_k_with_windows_b1))
                , k_with_windows_b2(std::move(_k_with_windows_b2))
                , global_similarity_matrix(std::move(_cost_matrix))
                , book_matrix(book_1.size(), vector<double>(book_2.size()))
                , alignment_matrix(k_with_windows_b1.size(), vector<double>(k_with_windows_b2.size()))
                , MAX_SIM_ADDITION_NEW_NODE(1.0 / k_double)
    {
        //k_with_windows_b1 = create_windows(book_1, k);
        //k_with_windows_b2 = create_windows(book_2, k);

        tokens_b1 = get_tokens(book_1);
        tokens_b2 = get_tokens(book_2);
    }

    Solutions(int _k, double _threshold, vector<int> _book_1, vector<int> _book_2, vector<vector<double>> _cost_matrix, vector<vector<double>> _word_vectors) :
            k(_k)
            , k_double(static_cast<double>(_k))
            , threshold(_threshold)
            , threshold_times_k(_threshold*_k)
            , book_1(std::move(_book_1))
            , book_2(std::move(_book_2))
            , col_maxima(vector<double>(k))
            , global_similarity_matrix(std::move(_cost_matrix))
            , word_vectors(std::move(_word_vectors))
            , book_matrix(book_1.size(), vector<double>(book_2.size()))
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
    }

    static void condense(const BitSet& bs, vector<pair<int,int>>& my_list) {
        uint32_t start_alt = 0;

        while((start_alt = bs.nextSetBit(start_alt))!=-1) {
            const uint32_t stop_alt = bs.nextClearBit(start_alt);
            my_list.emplace_back(start_alt, stop_alt-1);
            start_alt = stop_alt;
        }
    }

    static vector<int> get_tokens(const vector<int>& book) {
        unordered_set<int> temp;
        for(int id : book){
            temp.insert(id);
        }
        vector<int> ret;
        ret.reserve(temp.size());
        for(auto v : temp){
            ret.emplace_back(v);
        }
        sort(ret.begin(), ret.end());

        return ret;
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
                    alignment_matrix.at(line).at(column) = similarity/static_cast<double>(k);//normalize
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

    /**
     *
     * @param book all the paragraphs
     * @param k - window size
     * @return
     */
    static vector<vector<int>> create_windows(const vector<int> &book, int k) {
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
                        alignment_matrix.at(line).at(column) = similarity / static_cast<double>(k);//normalize
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

        const int vector_size = static_cast<int>(word_vectors.at(0).size());
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
            for(double & dim : my_avg_vec){
                dim /= length;
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
                for(double & dim : avg_vec_window_line){
                    dim /= length;
                }
            }
            for(int column=0;column<alignment_matrix.at(0).size();column++) {
                const vector<double>& avg_vec_window_column = averaged_colum_vectors.at(column);

                //That's the important line
                const double similarity = cosine_unit_length(avg_vec_window_line,avg_vec_window_column);
                //normalize costs: Before it was distance. Now it is similarity.
                if(similarity>=threshold_times_k) {
                    alignment_matrix[line][column] = similarity/static_cast<double>(k);//normalize
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
    double run_naive() {
        if(verbose) out_config("run_naive()");
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
                    alignment_matrix.at(line).at(column) = similarity/static_cast<double>(k);//normalize
                    count_computed_cells++;
                }//else keep it zero
            }
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;
        if(verbose){
            const double check_sum = sum(alignment_matrix);
            const auto size = alignment_matrix.size()*alignment_matrix.at(0).size();
            cout << "run_naive() time: " << time_elapsed.count() << "\t" << check_sum << "\t" <<  size << "\t" << count_computed_cells << endl;
        }
        return time_elapsed.count();
    }

    /**
     * Variant without zero Copy Hungarian using only O(k*k) filter. A.k.a. BaSem.
     * @return
     */
    double run_baseline() {
        if(verbose) out_config("run_baseline()");
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
                        alignment_matrix.at(line).at(column) = similarity / static_cast<double>(k);//normalize
                        count_computed_cells++;
                    }//else keep it zero
                }
            }
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;
        if(verbose){
            double check_sum = sum(alignment_matrix);
            auto size = alignment_matrix.size() * alignment_matrix.at(0).size();
            cout << "run_baseline() time: " << time_elapsed.count() << "\t" << check_sum << "\t" << size << "\t"
             << count_survived_pruning << "\t" << count_computed_cells << endl;
        }
        return time_elapsed.count();
    }

    static void out_vector_of_pairs(const vector<pair<int, int>>& my_vector) {
        for (const auto& runs : my_vector) {
            cout << "("<<runs.first<<","<<runs.second<<") ";
        }
        cout << endl;
    }

    void check_windows_for_candidates(const vector<int>& article_window, const vector<int>& query_window) {
        for(int a_token : article_window) {
            for (int q_token : query_window) {
                if (global_similarity_matrix.at(a_token).at(q_token) >= threshold) {
                    cout << "Has candidate" << endl;
                    return;
                }
            }
        }
        cout <<"Has no cani"<< endl;
        return;
    }

    void check_candidates(
        const vector<vector<pair<int, int>>>& all_candidates
        , const unordered_map<int, vector<pair<int, int>>> & my_candidates) {
        for(int article_window=0; article_window<all_candidates.size(); article_window++) {
            const auto& article_window_tokens = k_with_windows_b1.at(article_window);
            if(my_candidates.count(article_window) == 0) {//Does not exist -> should be all zero bits
                if (!all_candidates.at(article_window).empty()) {
                    cout << "SeDA candidate vector for window should be empty, but is not." << endl;
                    cout << "Window=" << article_window << endl;
                    //Check whether
                    out_vector_of_pairs(all_candidates.at(article_window));

                    for (const auto& runs : all_candidates.at(article_window)) {
                        for(int i=runs.first; i<=runs.second; i++) {
                            const auto& query_window_tokens = k_with_windows_b2.at(i);
                            check_windows_for_candidates(article_window_tokens, query_window_tokens);
                        }
                    }
                    cout << endl;
                }
            }else {//There are candidates
                const auto& window_runs_seda = all_candidates.at(article_window);
                const auto& window_runs_c_seda = my_candidates.at(article_window);
                if (window_runs_c_seda.size()!=window_runs_seda.size()) {
                    cout << "window_runs_c_seda.size()!=window_runs_seda.size()" << endl;
                    cout << "|SeDA|" << window_runs_seda.size() << endl;
                    cout << "|c-SeDA|" << window_runs_c_seda.size() << endl;
                    cout << "Article window=" << article_window << endl;
                    cout << "C-SeDA";
                    for (const auto& runs : window_runs_c_seda) {
                        cout << "("<<runs.first<<","<<runs.second<<") ";
                    }
                    cout << endl;
                    cout << "SeDA";
                    for (const auto& runs : window_runs_seda) {
                        cout << "("<<runs.first<<","<<runs.second<<") ";
                    }
                    cout << endl;
                    int a_id = 1;
                    int query_id = 0;
                    cout << "Window of article";
                    for (const int token : k_with_windows_b1.at(article_window) ) {
                        cout << token << " ";
                    }
                    cout << endl;

                    cout << "Window of query where run starts";
                    for (const int token : k_with_windows_b2.at(window_runs_c_seda.at(0).first) ) {
                        cout << token << " ";
                    }
                    cout << endl;

                    cout << "Window of query where run ends";
                    for (const int token : k_with_windows_b2.at(window_runs_c_seda.at(0).second) ) {
                        cout << token << " ";
                    }
                    cout << endl;
                    cout << "********" << endl;

                    for (const auto& runs : window_runs_seda) {
                        for(int i=runs.first; i<=runs.second; i++) {
                            const auto& query_window_tokens = k_with_windows_b2.at(i);
                            check_windows_for_candidates(article_window_tokens, query_window_tokens);
                        }
                    }
                    for (const auto& runs : window_runs_c_seda) {
                        for(int i=runs.first; i<=runs.second; i++) {
                            const auto& query_window_tokens = k_with_windows_b2.at(i);
                            check_windows_for_candidates(article_window_tokens, query_window_tokens);
                        }
                    }
                    cout << endl;
                    cout << "********" << endl;
                }else {
                    //Check until first error
                    for (int i=0;i<window_runs_c_seda.size();i++) {
                        const pair<int,int> c_seda_pair = window_runs_c_seda.at(i);
                        const pair<int,int> seda_pair = window_runs_seda.at(i);
                        if (c_seda_pair.first!=seda_pair.first) {
                            cout << "First error at i=" << i << endl;
                            break;
                        }
                        if (c_seda_pair.second!=seda_pair.second) {
                            cout << "First error at i=" << i << endl;
                            break;
                        }
                    }
                }
            }
        }
    }

    /**
     *
     * @param my_candidates <line, L<run_start,run_stop>>
     * @return
     */
    double run_solution_corpus_2(const unordered_map<int,vector<pair<int,int>>>& my_candidates){
        if (verbose) out_config("run_solution_corpus()");
        HungarianDeep solver(k);
        vector<vector<pair<int,int>>> all_candidates;
        all_candidates.reserve(k_with_windows_b1.size());
        get_candidates(all_candidates);
        //check_candidates(all_candidates, my_candidates);

        //Not needed later
        vector<const double*> window(k);//Can't use a vector to point into an existing buffer.
        fill_similarity_matrix_deep();

        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

        //Check candidate runs
        for(int line=0;line<alignment_matrix.size();line++) {
            vector<double>& alignment_matrix_line = alignment_matrix[line];

            for(const auto& current_run : all_candidates[line]) {//Contains start and stop index. Thus, c+=2.
                const int run_start = current_run.first;
                const int run_stop  = current_run.second;

                double ub_sum, sim, prior_cell_similarity, prev_min_value;
                bool prior_cell_updated_matrix, column_sum_correct;

                int column=run_start;
                {//First element in run: Here we have no O(1) bound
                    for(int i = 0;i<k;i++){//Init sliding window
                        const double* temp = &book_matrix[line+i][column];
                        window[i] = temp;
                    }
                    ub_sum = o_k_square_bound(window) / k_double;

                    if(ub_sum+DOUBLE_PRECISION_BOUND>=threshold) {
                        sim = -solver.solve(col_maxima, window);//Note the minus-trick for the Hungarian
                        sim /= k_double;
                        if(sim>=threshold) {
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
                        window[i]++;// = temp;//TODO
                    }

                    double upper_bound_sim = prior_cell_similarity + MAX_SIM_ADDITION_NEW_NODE;// O(1) bound
                    if(prior_cell_updated_matrix) {//We know what similarity we loose at least
                        upper_bound_sim-= (prev_min_value / k_double);// (1) O(k) bound : part of the O(k) bound in case the prior cell updated the matrix, i.e., we know the minimum similarity of the leaving node
                    }

                    if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
                        const double max_sim_new_node = min(window);    //(2) O(k) bound
                        upper_bound_sim-=MAX_SIM_ADDITION_NEW_NODE;     //Instead of assuming the incoming node adds max_sim=1.0...
                        upper_bound_sim+=(max_sim_new_node/k_double);   // ... we use the maximum of all sim() values

                        if(column_sum_correct) {
                            sum_cols -= col_maxima[0];
                            sum_cols -= max_sim_new_node;//is not negated
                            double temp = -sum_cols / k_double;

                            if(temp<upper_bound_sim) {//This bound is not always tighter
                                upper_bound_sim = temp;
                            }
                        }

                        if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
                            ub_sum = o_k_square_bound(window) / k_double;
                            //The sum bound is not necessarily tighter, we need the tightest bound for bound cascade of the *next* window
                            upper_bound_sim = (ub_sum<upper_bound_sim) ? ub_sum : upper_bound_sim;

                            if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
                                sim = -solver.solve(col_maxima, window);//Note the minus-trick for the Hungarian
                                //normalize
                                sim /= k_double;

                                if(sim>=threshold) {
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

        return time_elapsed.count();
    }

    /**
     *
     * @param my_candidates <line, L<run_start,run_stop>>
     * @return
     */
    double run_solution_corpus(const vector<pair<int,vector<pair<int, int>>>>& my_candidates){
        if (verbose) out_config("run_solution_corpus()");
        HungarianDeep solver(k);

        //Not needed later
        vector<const double*> window(k);//Can't use a vector to point into an existing buffer.
        fill_similarity_matrix_deep();

        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

        //Check candidate runs
        for(const auto& entry : my_candidates) {
            const int line= entry.first;
            vector<double>& alignment_matrix_line = alignment_matrix[line];

            for(const auto& current_run : entry.second) {//Contains start and stop index. Thus, c+=2.
                const int run_start = current_run.first;
                const int run_stop  = current_run.second;

                double ub_sum, sim, prior_cell_similarity, prev_min_value;
                bool prior_cell_updated_matrix, column_sum_correct;

                int column=run_start;
                {//First element in run: Here we have no O(1) bound
                    for(int i = 0;i<k;i++){//Init sliding window
                        const double* temp = &book_matrix[line+i][column];
                        window[i] = temp;
                    }
                    ub_sum = o_k_square_bound(window) / k_double;

                    if(ub_sum+DOUBLE_PRECISION_BOUND>=threshold) {
                        sim = -solver.solve(col_maxima, window);//Note the minus-trick for the Hungarian
                        sim /= k_double;
                        if(sim>=threshold) {
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
                        window[i]++;// = temp;//TODO
                    }

                    double upper_bound_sim = prior_cell_similarity + MAX_SIM_ADDITION_NEW_NODE;// O(1) bound
                    if(prior_cell_updated_matrix) {
                        upper_bound_sim-= (prev_min_value / k_double);// (1) O(k) bound : part of the O(k) bound in case the prior cell updated the matrix, i.e., we know the minimum similarity of the leaving node
                    }

                    if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
                        const double max_sim_new_node = min(window);//(2) O(k) bound
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
                            ub_sum = o_k_square_bound(window) / k_double;
                            //The sum bound is not necessarily tighter, we need the tightest bound for bound cascade of the *next* window
                            upper_bound_sim = (ub_sum<upper_bound_sim) ? ub_sum : upper_bound_sim;

                            if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
                                sim = -solver.solve(col_maxima, window);//Note the minus-trick for the Hungarian
                                //normalize
                                sim /= k_double;

                                if(sim>=threshold) {
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

        return time_elapsed.count();
    }

    void get_candidates(vector<vector<pair<int,int>>>& all_candidates) const {
        vector<BitSet> inverted_window_index_bit_set(global_similarity_matrix.size(), BitSet(static_cast<int>(k_with_windows_b2.size())));
        create_indexes_bit_vectors(inverted_window_index_bit_set);
        vector<BitSet> all_bit_candidates(k_with_windows_b1.size(), BitSet(static_cast<int>(k_with_windows_b2.size())));

        /*if (global_article_id==1 && global_query_id == 0) {
            cout << "****************** get_candidates()" << endl;
            cout << inverted_window_index_bit_set.at(62).words.at(0) << endl;
            cout << "******************" << endl;
//            exit(0);
        }*/

        for(int line=0;line<alignment_matrix.size();line++) {
            const vector<int>& window_b1 = k_with_windows_b1[line];
            BitSet& my_candidates = all_bit_candidates.at(line);
            /*if (line==79) {
                cout << line << endl;//TODO remove me
            }*/
            my_candidates.logic_or(inverted_window_index_bit_set, window_b1);

            //Manually inlined condense transforms the bit vector into runs of candidates
            vector<pair<int,int>> candidates_condensed_bit_set;
            condense(my_candidates, candidates_condensed_bit_set);
            all_candidates.emplace_back(candidates_condensed_bit_set);
        }
    }

    double run_solution(){
        if (verbose) out_config("run_solution()");
        HungarianDeep solver(k);
        /**
         * Indicates for token i whether the corresponding windows of the other sequence is a candidate.
         */
        //vector<vector<bool>> inverted_window_index(global_similarity_matrix.size(), vector<bool>(k_with_windows_b2.size()));
        vector<BitSet> inverted_window_index_bit_set(global_similarity_matrix.size(), BitSet(static_cast<int>(k_with_windows_b2.size())));
        //Not needed later
        vector<const double*> window(k);//Can't use a vector to point into an existing buffer.
        fill_similarity_matrix_deep();
        vector<BitSet> all_bit_candidates(k_with_windows_b1.size(), BitSet(static_cast<int>(k_with_windows_b2.size())));

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
                candidates_condensed_bit_set.push_back(static_cast<int>(start_alt));//XXX The casting should be removed
                candidates_condensed_bit_set.push_back(static_cast<int>(stop_alt)-1);
                start_alt = stop_alt;
            }

            const vector<int>& candidates_condensed = candidates_condensed_bit_set;

            const int size = static_cast<int>(candidates_condensed.size());
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
        if (verbose) {
            double check_sum = sum(alignment_matrix);
            auto size = alignment_matrix.size()*alignment_matrix.at(0).size();
            cout << "run_solution(k=" << k << ") time: " << time_elapsed.count() << " idx_gen= " << index_generation.count() << " time= " << time_elapsed.count() << "\t sum=" << check_sum << "\t size=" << size << "\t |C|=" << count_candidates << "\t |O(1)|" << count_survived_o_1 << "\t |O(k)|" << count_survived_o_k << "\tO(k*k)" << count_survived_o_k_square << "\t" << count_cells_exceeding_threshold << endl;
        }
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
        vector<BitSet> inverted_window_index_bit_set(global_similarity_matrix.size(), BitSet(static_cast<int>(k_with_windows_b2.size())));
        //Not needed later
        //vector<const double*> window(k);//Can't use a vector to point into an existing buffer.
        //fill_similarity_matrix_deep();
        vector<BitSet> all_bit_candidates(k_with_windows_b1.size(), BitSet(static_cast<int>(k_with_windows_b2.size())));

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
                candidates_condensed_bit_set.push_back(static_cast<int>(start_alt));
                candidates_condensed_bit_set.push_back(static_cast<int>(stop_alt)-1);
                start_alt = stop_alt;
            }

            const vector<int>& candidates_condensed = candidates_condensed_bit_set;

            const int size = static_cast<int>(candidates_condensed.size());
            for(int c=0;c<size;c+=2) {//Contains start and stop index. Thus, c+=2.
                const int run_start = candidates_condensed[c];
                const int run_stop  = candidates_condensed[c+1];

                double sim, prior_cell_similarity, prev_min_value;

                count_candidates+=run_stop-run_start+1;
                int column=run_start;
                {
                    double ub_sum;
                    //First element in run: Here we have no O(1) bound
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
        cout << "run_solution(k=" << k << ") time: " << time_elapsed.count() << " idx_gen= " << index_generation.count() << " time= " << time_elapsed.count() << "\tsum=" << check_sum << "\t size=" << size << "\t |C|=" << count_candidates << "\t |O(1)|" << count_survived_o_1 << "\t |O(k)|" << count_survived_o_k << "\tO(k*k)" << count_survived_o_k_square << "\t" << count_cells_exceeding_threshold << endl;
        return time_elapsed.count();
    }
};

class Corpus {
    const vector<vector<int>> raw_corpus;
    const vector<vector<double>> embedding_vector_index;
    vector<vector<int>> unique_tokens_per_article;
    vector<vector<vector<int>>> k_width_windows;
    const vector<vector<int>> candidate_producing_token_pairs;
    const vector<unordered_set<int>> candidate_producing_token_pairs_hashed;
    const int k;
    const double threshold;

    /**
     * at(article_id).at(token_id)->positions in raw_corpus.at(article_id)
     */
    vector<unordered_map<int, vector<int>>> token_positions_articles;
    vector<vector<int>> inverted_token_index;

    static void out_short(const vector<vector<double>>& to_display) {
        constexpr int num_articles_to_display = 3;
        for (int article = 0;article<num_articles_to_display;article++) {
            auto& vec = to_display.at(article);
            cout << "id="<< article++ << " ";
            cout << "size=" << vec.size() << "\t[";
            for (int j=0;j<5;j++) {
                cout << vec.at(j) << ", ";
            }
            cout << "..." <<endl;
        }
        cout << "..." << endl;
        for (int article = static_cast<int>(to_display.size())-num_articles_to_display;article<to_display.size();article++) {
            auto& vec = to_display.at(article);
            cout << "id="<< article++ << " ";
            cout << "size=" << vec.size() << "\t[";
            for (int j=0;j<5;j++) {
                cout << vec.at(j) << ", ";
            }
            cout << "..." <<endl;
        }
    }

    static void out_short(const vector<vector<int>>& to_display) {
        constexpr int num_articles_to_display = 3;
        for (int article = 0;article<num_articles_to_display;article++) {
            auto& vec = to_display.at(article);
            cout << "id="<< article++ << " ";
            cout << "size=" << vec.size() << "\t[";
            for (int j=0;j<min(static_cast<int>(vec.size()),5);j++) {
                cout << vec.at(j) << ", ";
            }
            cout << "..." <<endl;
        }
        cout << "..." << endl;
        for (int article = static_cast<int>(to_display.size())-num_articles_to_display;article<to_display.size();article++) {
            auto& vec = to_display.at(article);
            cout << "id="<< article++ << " ";
            cout << "size=" << vec.size() << "\t[";
            for (int j=0;j<min(static_cast<int>(vec.size()),5);j++) {
                cout << vec.at(j) << ", ";
            }
            cout << "..." <<endl;
        }
    }

    static double get_similarity(const int set_id1, const int set_id2, const vector<double>& vec_1, const vector<double>& vec_2) {
        if (set_id1==set_id2) {
            return 1;
        }
        double dotProduct = 0.0;
        for (int i = 0; i < vec_1.size(); i++) {
            dotProduct += vec_1[i] * vec_2[i];
        }

        dotProduct = (dotProduct < 0) ? 0 : dotProduct;
        dotProduct = (dotProduct > 1) ? 1 : dotProduct;
        return dotProduct;
    }
public:
    explicit Corpus(const int _k, const double _threshold, vector<vector<int>> _raw_corpus_int
        , vector<vector<double>> _embeddings
        , vector<vector<int>> _candidate_producing_token_pairs
        , vector<unordered_set<int>> _candidate_producing_token_pairs_hashed
    ) : raw_corpus(std::move(_raw_corpus_int)),
        embedding_vector_index(std::move(_embeddings)),
        candidate_producing_token_pairs(std::move(_candidate_producing_token_pairs)),
        candidate_producing_token_pairs_hashed(std::move(_candidate_producing_token_pairs_hashed)),
        k(_k),
        threshold(_threshold),
        inverted_token_index(embedding_vector_index.size()) {
        cout << "Indexing Corpus" << endl;
        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        for (int article_id = 0; article_id < raw_corpus.size(); article_id++) {
            //cout << "id="<< article_id << endl;
            auto &article = raw_corpus.at(article_id);
            vector<int> my_unique_tokens = Solutions::get_tokens(article);
            unordered_map<int, vector<int> > token_positions;

            for (int token_id: my_unique_tokens) {
                vector<int> token_id_index;
                for (int position = 0; position < article.size(); position++) {
                    if (article.at(position) == token_id) {
                        token_id_index.push_back(position);
                    }
                }
                token_positions.insert({token_id, token_id_index});
                inverted_token_index.at(token_id).push_back(article_id);
            }

            unique_tokens_per_article.emplace_back(my_unique_tokens);
            token_positions_articles.emplace_back(token_positions);
            auto my_windows = Solutions::create_windows(article, k);
            k_width_windows.push_back(my_windows);
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;
        cout << "Indexing Corpus [Done] " << time_elapsed.count() << endl;
    }

    pair<int,int> has_candidate(const vector<int>& article_window, const vector<int>& query_sequence) const {
        for (int article_token : article_window) {
            for (int query_token: query_sequence) {
                if (candidate_producing_token_pairs_hashed.at(article_token).count(query_token) == 1) {
                    pair<int,int> result = make_pair(article_token, query_token);
                    return result;
                }
            }
        }

        return make_pair(-1,-1);
    }

    /**
     *
     * @param query_id
     * @param all_candidates_corpus
     * @param aggregated_runtime
     */
    void filter(const int query_id
                //, vector<vector<pair<int,int>>>& line_runs
                , vector<unordered_map<int,vector<pair<int,int>>>>& all_candidates_corpus
                , vector<double>& aggregated_runtime) const {
        cout << "Computing corpus filter for query " << query_id;
        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();

        const vector<int>& query = raw_corpus.at(query_id);
        const auto& query_windows = k_width_windows.at(query_id);
        const int num_doubles_for_bit_vector = static_cast<int>((query_windows.size()/8)+1);

        vector<BitSet> candidate_documents;//TODO move to constructor
        candidate_documents.reserve(raw_corpus.size());
        for (int article_id=0;article_id<raw_corpus.size();article_id++) {
            candidate_documents.emplace_back(k_width_windows.at(article_id).size());//Must have as many bits as windows
        }

        for(int token_id : unique_tokens_per_article.at(query_id)) {//For each unique token of the query
            const auto& N_token_id = candidate_producing_token_pairs.at(token_id);
            for(int other_token : N_token_id) {//Iterate over all pairs (token_id, other_token) having sim() > threshold
                const auto& my_index = inverted_token_index.at(other_token);
                for(int article_id : my_index) { // These are the articles that contain other_token
                    if(article_id==query_id) {
                        continue; // Ignore myself
                    }
                    BitSet& candidate_lines = candidate_documents.at(article_id);
                    const vector<int>& positions = token_positions_articles.at(article_id).at(other_token);
                    int index_last_k_width_windows = static_cast<int>(k_width_windows.at(article_id).size());
                    for(int position : positions) {//These are the token positions, not the windows
                        int from = max(0,position-k+1);
                        int to = min(position+1, index_last_k_width_windows);
                        /*if (article_id==1) {
                            cout << "(" <<token_id <<", "<< other_token << "): -> from,to= " << from <<" "<< to << endl;
                        }*/
                        candidate_lines.set(from, to);
                    }
                }
            }
        }

        //Condense the BitSets to line runs
        vector<vector<pair<int,int>>> line_runs;
        for(int article_id=0;article_id<raw_corpus.size();article_id++) {
            const BitSet& candidate_lines = candidate_documents.at(article_id);

            /*{
                //checked all lines
                if (article_id!=query_id) {
                    const auto& my_windows = k_width_windows.at(article_id);
                    for (int window=0;window<my_windows.size();window++) {
                        bool line_contains_candidate = candidate_lines.get(window);
                        auto my_pair = has_candidate(my_windows.at(window), query);
                        bool line_contains_candidate_2;
                        if (my_pair.first==-1) {
                            line_contains_candidate_2 = false;
                        }else{
                            line_contains_candidate_2 = true;
                        }
                        if(line_contains_candidate!=line_contains_candidate_2) {
                            cout << endl;
                            cout <<" aid="<< article_id << " w=" << window << endl;
                            cout << "Index=" << line_contains_candidate << " N=" << line_contains_candidate_2 << endl;
                            cout << my_pair.first << " " << my_pair.second << endl;
                            cout << endl;
                        }
                    }
                }
            }*/
            //Manually inlined condense transforms the bit vector into runs of candidates
            vector<pair<int,int>> candidates_condensed_bit_set;
            Solutions::condense(candidate_lines, candidates_condensed_bit_set);
            line_runs.push_back(candidates_condensed_bit_set);
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;
        aggregated_runtime.push_back(time_elapsed.count());
        cout << " [Done]\t" << time_elapsed.count() <<endl;

        //Some statistics
        long sum_cells = 0l;
        long sum_lines = 0l;
        long sum_lines_pruned = 0l;
        long sum_cells_pruned = 0l;
        for(int article_id=0;article_id<raw_corpus.size();article_id++) {
            const BitSet& bs = candidate_documents.at(article_id);

            int size_m = static_cast<int>(k_width_windows.at(article_id).size() * query_windows.size());
            int count_w_pruned = 0;
            for(int w=0;w<k_width_windows.at(article_id).size();w++) {
                if(bs.get(w)==false) {
                    count_w_pruned++;
                }
            }
            int count_cells_pruned = static_cast<int>(count_w_pruned * query_windows.size());
            sum_cells += size_m;
            sum_lines += static_cast<int>(k_width_windows.at(article_id).size());
            sum_lines_pruned += count_w_pruned;
            sum_cells_pruned += count_cells_pruned;
            if(sum_cells<sum_lines_pruned) {
                cout << "sum_cells<sum_lines_pruned" << endl;
            }
            if(sum_lines<sum_lines_pruned) {
                cout << "sum_lines<sum_lines_pruned" << endl;
            }
        }
        cout << "Articles.filter " << query_id << " [DONE] in "<<time_elapsed.count()<<"\t"<< +line_runs.size()<<"\t"<<sum_cells<<"\t"<<sum_lines<<"\t"<<sum_lines_pruned<<"\t"<<sum_cells_pruned<<endl;

        start = std::chrono::high_resolution_clock::now();
        //Now start inlined column filter
        unordered_map<int,BitSet> candidate_vectors;

        unordered_set<int> index_N;//All the token that produce candidates //TODO make bit set
        for (int query_token : query) {
            for(int neighbor : candidate_producing_token_pairs.at(query_token)) {
                index_N.insert(neighbor);
            }
        }

        //(1) We compute for each token in the neighborhood index index_N the column vector once
        for(int other_token_id : index_N) {
            //cout << "Creating Vector for token_id= " << other_token_id << endl;
            BitSet bs(num_doubles_for_bit_vector);//one bit per window, if true this is a candidate column
            {
                //inlined create_bit_vector()
                // That's the index of other_token_id -> looking for pairs sim(other_token_id, query_token_at_pos)>=threshold
                const unordered_set<int>& my_neighborhood_index = candidate_producing_token_pairs_hashed[other_token_id];//FIX was copied
                for(int position=0; position<query.size();position++) {//Loop over each token of the query
                    int query_token_at_pos = query[position];
                    if(my_neighborhood_index.count(query_token_at_pos)) {//contains()
                        const int from = max(0, position-k+1);
                        const int to = min(static_cast<int>(query_windows.size())-1, position);
                        bs.set(from,to+1);
                    }
                }
            }
            candidate_vectors.insert({other_token_id, bs});
        }
        //(2) Now lets get the candidate runs for Line of the alignment matrix
        for(int article_id=0;article_id<raw_corpus.size();article_id++) {
            const vector<pair<int,int>>& article_line_runs = line_runs[article_id];
            const vector<vector<int>>& article_windows = k_width_windows[article_id];
            unordered_map<int,vector<pair<int,int>>> all_candidates_article;

            for(const pair<int,int>& run : article_line_runs) {
                const int run_start = run.first;
                const int run_stop  = run.second;

                for(int line=run_start; line<=run_stop;line++) {//refers to a line in the Alignment Matrix
                    const auto& article_window = article_windows[line];
                    BitSet window_bit_vectors(num_doubles_for_bit_vector);//TODO pre-allocate and nullify

                    for(int token_id : article_window) {//Not all tokens in the window have a vector. They only have one if they create at least one candidate pair
                        const auto& temp = candidate_vectors.find(token_id);
                        if(temp!=candidate_vectors.end()) {//contains()
                            //const BitSet& my_bit_vector=candidate_vectors.at(token_id);
                            window_bit_vectors.logic_or(temp->second);
                        }
                    }
                    /*{
                        // checked candidate runs
                        for(int q_window=0;q_window<query_windows.size();q_window++) {
                            bool index_candidate = window_bit_vectors.get(q_window);
                            auto my_pair = has_candidate(article_window, query_windows.at(q_window));
                            bool line_contains_candidate_2;
                            if (my_pair.first==-1) {
                                line_contains_candidate_2 = false;
                            }else{
                                line_contains_candidate_2 = true;
                            }
                            if(index_candidate!=line_contains_candidate_2) {
                                cout << endl;
                                cout <<" aid="<< article_id << " line=" << line << " w=" << q_window << endl;
                                cout << "Index=" << index_candidate << " N=" << line_contains_candidate_2 << endl;
                                cout << my_pair.first << " " << my_pair.second << endl;
                                cout << endl;
                            }
                        }
                    }*/
                    //condense
                    vector<pair<int,int>> candidates_condensed;
                    Solutions::condense(window_bit_vectors, candidates_condensed);
                    all_candidates_article.insert({line, candidates_condensed});
                }
            }
            all_candidates_corpus.push_back(all_candidates_article);
        }
        time_elapsed = std::chrono::high_resolution_clock::now() - start;
        long long sum_num_cells = 0;
        long long cells_remaining = 0;

        for(int article_id=0;article_id<raw_corpus.size();article_id++) {
            const auto& article_candidates = all_candidates_corpus.at(article_id);
            sum_num_cells+= static_cast<int>(k_width_windows.at(article_id).size() * query_windows.size());
            for (const auto& line_candidates : article_candidates) {
                for(const auto& run : line_candidates.second) {
                    int run_start = run.first;
                    int run_stop = run.second;
                    cells_remaining += run_stop-run_start;
                }
            }
        }
        aggregated_runtime.push_back(time_elapsed.count());
        cout << "Candidate filter \t"<<sum_num_cells << "\t" << cells_remaining << "\tin\t" << time_elapsed.count()<< endl;
    }


    static void to_vector(const unordered_map<int, vector<pair<int, int>>>& hash_map
        , vector<pair<int,vector<pair<int, int>>>>& vector_to_create, const int num_article_windows) {

        vector_to_create.reserve(num_article_windows);

        for(int window=0;window<num_article_windows;window++) {
            if(hash_map.count(window)) {//This window has candidates
                vector_to_create.emplace_back(window,hash_map.at(window));
            }/*else {
                vector<pair<int,int>> dummy;
                vector_to_create.emplace_back(dummy);
            }*/
        }
    }

    double query(const int query_id, vector<double>& aggregated_runtime, const int approach_to_run) const {
        //int k = 10;
        //double threshold = 0.7;
        global_query_id = query_id;//for debug

        vector<double> runtimes;
        runtimes.reserve(raw_corpus.size());
        const unordered_map<int, vector<double>> sim = get_sim(raw_corpus.at(query_id), embedding_vector_index, aggregated_runtime);
        const auto& my_unique_tokens = unique_tokens_per_article.at(query_id);
        const vector<int>& query = raw_corpus.at(query_id);

        //Compute the corpus filter
        //vector<vector<pair<int,int>>> line_runs;
        vector<unordered_map<int,vector<pair<int,int>>>> all_candidates_corpus;
        filter(query_id, all_candidates_corpus, aggregated_runtime);

        if (approach_to_run == run_seda){
            cout << "Running SeDA" << endl;
        }else if(approach_to_run == run_c_seda_2) {
            cout << "Running run_solution_corpus_2()" << endl;
        }else if(approach_to_run == run_c_seda) {
            cout << "Running run_solution_corpus()" << endl;
        }else if(approach_to_run == run_naive) {
            cout << "Running run_naive()" << endl;
        }else if(approach_to_run == run_basem) {
            cout << "Running run_baseline()" << endl;
        }else{
            cout << "Running Nothing in else branch" << endl;
        }

        for (int article_id=0;article_id<raw_corpus.size();article_id++) {
            global_article_id = article_id;//for debug
            if (query_id == article_id) {
                continue; // do not query myself
            }
            const auto& article_unique_tokens = unique_tokens_per_article.at(article_id);
            //XXX This copies the object s
            Solutions s = map_to_new_alphabet(k, threshold, sim, query, raw_corpus.at(article_id), embedding_vector_index, my_unique_tokens, article_unique_tokens);
            double runtime;
            if (approach_to_run == run_seda){
                runtime = s.run_solution();
            }else if(approach_to_run == run_c_seda) {
                vector<pair<int,vector<pair<int, int>>>> all_candidates_as_vector;//For better debugging
                const int num_article_windows = static_cast<int>(k_width_windows.at(article_id).size());
                to_vector(all_candidates_corpus.at(article_id), all_candidates_as_vector, num_article_windows);
                runtime = s.run_solution_corpus(all_candidates_as_vector);
                //runtime = s.run_solution_corpus_2(all_candidates_corpus.at(article_id));
            }else if(approach_to_run == run_c_seda_2) {
                runtime = s.run_solution_corpus_2(all_candidates_corpus.at(article_id));
            }else if(approach_to_run == run_basem) {
                runtime = s.run_baseline();
            }else if(approach_to_run == run_naive) {
                runtime = s.run_naive();
            }else{
                runtime = 0;//Do nothing
            }

            runtimes.push_back(runtime);
            if (article_id % 100 == 0) {
                cout << article_id << " ";
            }
        }
        cout << endl;
        double sum_runtime = Solutions::sum(runtimes);
        aggregated_runtime.push_back(sum_runtime);
        return  sum_runtime;
    }

    static unordered_map<int, vector<double>> get_sim(const vector<int>& query, const vector<vector<double>>& embeddings, vector<double>& aggregated_runtime) {
        chrono::time_point<std::chrono::high_resolution_clock> start = std::chrono::high_resolution_clock::now();
        unordered_map<int, vector<double>> sim;
        for (int token : query) {
            if (sim.find(token) == sim.end()) {//not seen this token before
                vector<double> sim_line(embeddings.size());//similarity to any token
                const vector<double>& my_embedding = embeddings.at(token);
                for (int other_token=0;other_token<embeddings.size();other_token++) {
                    const vector<double>& other_embedding = embeddings.at(other_token);
                    double token_sim = get_similarity(token, other_token, my_embedding, other_embedding);
                    sim_line.at(other_token) = token_sim;
                }
                sim.insert(make_pair(token, sim_line));
            }
        }
        chrono::duration<double> time_elapsed = std::chrono::high_resolution_clock::now() - start;
        cout << "Creating sim() for query [Done]\t" << time_elapsed.count() <<endl;
        aggregated_runtime.push_back(time_elapsed.count());
        return sim;
    }

    static void get_N(vector<vector<int>>& fill_me, const string &path){
        cout << "Creating N from from = " << path <<  endl;

        ifstream in_file(path);
        if (in_file.is_open()) {
            string line;
            //cout << "line" << endl;
            int id = 0;
            while (getline(in_file, line)) {
                //cout << line << endl;
                vector<string> tokens = Environment::split(line, ' ');
                vector<int> my_vector(tokens.size()-1);
                for(int j=1;j<tokens.size();j++) {// We start j=1
                    int neighbor = stoi(tokens.at(j));
                    my_vector.at(j-1) = neighbor;
                }

                fill_me.emplace_back(my_vector);
                id++;
                if (id%10000==0) {
                    cout << id <<" of about 50k"<< endl;
                }
            }
        }else {
            cout << "Could not open " << path << endl;
        }
        in_file.close();

        cout << "Done creating N" << endl;
        out_short(fill_me);
    }

    static void get_articles_tokenized(const string& path, vector<vector<int>>& raw_corpus_int) {
        ifstream corpus_file(path);
        if (corpus_file.is_open()) {
            string line;
            //cout << "line" << endl;
            while (getline(corpus_file, line)) {
               // cout << line << endl;
                vector<string> tokens = Environment::split(line, ' ');
                vector<int> tokens_int;
                transform(
                    tokens.begin(), tokens.end(), back_inserter(tokens_int),
               [](const string& str) { return stoi(str); }
                );
                raw_corpus_int.push_back(tokens_int);
            }
        }else {
            cout << "Could not open " << path << endl;
        }
        corpus_file.close();

        cout << "Done reading corpus" << endl;
        out_short(raw_corpus_int);
    }

    static void get_embeddings(const string& path, vector<vector<double>>& embeddings) {
        cout << "Reading embeddings from " << path << endl;
        ifstream embeddings_file(path);

        if (embeddings_file.is_open()) {
            string line;
            //cout << "line" << endl;
            int id = 0;
            while (getline(embeddings_file, line)) {
                // cout << line << endl;
                vector<string> tokens = Environment::split(line, ' ');
                vector<double> my_vector;
                transform(
                    tokens.begin(), tokens.end(), back_inserter(my_vector),
               [](const string& str) { return stod(str); }
                );
                embeddings.emplace_back(my_vector);
                id++;
                if (id%10000==0) {
                    cout << id <<" of about 50k"<< endl;
                }
            }
        }else {
            cout << "Could not open " << path << endl;
        }
        embeddings_file.close();

        cout << "Done reading embeddings" << endl;
        out_short(embeddings);
    }

    static Solutions map_to_new_alphabet(const int k, const double threshold, const unordered_map<int, vector<double>>& sim_query
        , const vector<int>& query, const vector<int>& article, const vector<vector<double>>& embeddings
        , const vector<int>& unique_tokens_query, const vector<int>& unique_tokens_article
        ) {

        unordered_set<int> all_unique_tokens;
        for (int token : unique_tokens_query) {
            all_unique_tokens.insert(token);
        }
        for (int token : unique_tokens_article) {
            all_unique_tokens.insert(token);
        }
        vector<int> all_tokens_ordered(all_unique_tokens.size()); //Allocate size
        {
            int position = 0;
            for (int token : all_unique_tokens) {
                all_tokens_ordered.at(position++) = token;
            }
        }
        sort(all_tokens_ordered.begin(), all_tokens_ordered.end());//optional? nope!
        //const int max_id = all_tokens_ordered.size();//Last element has max id
        unordered_map<int, int> new_token_ids;
        //unordered_map<int, vector<double>> new_embedding_vector_index;
        for(int new_id=0;new_id<all_tokens_ordered.size();new_id++){
            const int old_id = all_tokens_ordered.at(new_id);
            new_token_ids.insert({old_id, new_id});
            //const vector<double>& my_vector = embeddings.at(old_id);
            //new_embedding_vector_index.insert({new_id, my_vector});//TODO add to run_fast_text()
        }

        //Now create the wrappers for the Solution class.
        vector<int> raw_paragraph_b1(article.size());
        for(int i=0;i<raw_paragraph_b1.size();i++) {
            const int old_id = article.at(i);
            const int new_id = new_token_ids.at(old_id);
            raw_paragraph_b1.at(i) = new_id;
        }

        vector<int> raw_paragraph_b2(query.size());
        for(int i=0;i<raw_paragraph_b2.size();i++) {
            const int old_id = query.at(i);
            const int new_id = new_token_ids.at(old_id);
            raw_paragraph_b2.at(i) = new_id;
        }
        const int max_id_query   = unique_tokens_query.at(unique_tokens_query.size()-1);
        const int max_id_article = unique_tokens_article.at(unique_tokens_article.size()-1);
        const int array_length = new_token_ids.at(max_id_query)+1;
        vector<vector<double>> sim(new_token_ids.at(max_id_article)+1,vector<double>(array_length));//TODO to large!? Seems to work

        //materialize sim()
        for(int old_id_q : unique_tokens_query){
            const int new_id_q = new_token_ids.at(old_id_q);
            const vector<double>& sim_line = sim_query.at(old_id_q);

            for(int old_id_a : unique_tokens_article) {
                const int new_id_a = new_token_ids.at(old_id_a);
                sim.at(new_id_a).at(new_id_q) = sim_line.at(old_id_a);
            }
        }

        const auto k_width_windows_b1 = Solutions::create_windows(raw_paragraph_b1, k);
        const auto k_width_windows_b2 = Solutions::create_windows(raw_paragraph_b2, k);

        //TODO Remove need for copy objects
        Solutions s(k,threshold,raw_paragraph_b1, raw_paragraph_b2, sim, k_width_windows_b1, k_width_windows_b2);
        return s;
        //double runtime = s.run_baseline();
        //double runtime = s.run_solution();
        //return runtime;
    }

};

#endif //PRANAY_TEST_SOLUTIONS_H
