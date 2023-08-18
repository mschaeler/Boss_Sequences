package boss.hungarian;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;

import boss.embedding.MatchesWithEmbeddings;
import boss.util.MyArrayList;

public class Solutions {
	long count_candidates = 0;
	long count_survived_sum_bound = 0;
	long count_cells_exceeding_threshold = 0;
	
	static double[][] dense_global_matrix_buffer = null;
	static final double DOUBLE_PRECISION_BOUND = 0.0001d;
	private static final boolean SAVE_MODE = false;
	
	final int k;
	final double k_double;
	final int num_paragraphs;
	final double threshold;
	final int max_id;
	final double threshold_times_k;
	
	final int[][] k_with_windows_b1;
	final int[][] k_with_windows_b2;
	
	final int[] raw_paragraph_b1;
	final int[] raw_paragraph_b2;
	
	final double[][] alignement_matrixes;
	
	final HashMap<Integer, double[]> embedding_vector_index;
	
	/**
	 * Contains the maximum column similarity of current local similarity matrix. Note, since we negate the signum for the hungarian. It's the minimum....
	 */
	final double[] col_maxima;
	
	public Solutions(ArrayList<int[]> raw_paragraphs_b1, ArrayList<int[]> raw_paragraphs_b2, final int k, final double threshold, HashMap<Integer, double[]> embedding_vector_index) {
		this.k = k;
		this.k_double = (double) k;
		this.threshold = threshold;
		this.threshold_times_k = threshold * k;
		this.num_paragraphs = raw_paragraphs_b1.size();
		if(num_paragraphs!=1) {
			System.err.println("Solutions() - Expecting book granularity");
		}
		
		this.raw_paragraph_b1 = raw_paragraphs_b1.get(0);
		this.raw_paragraph_b2 = raw_paragraphs_b2.get(0);
		this.k_with_windows_b1 = create_windows(raw_paragraph_b1, k);
		this.k_with_windows_b2 = create_windows(raw_paragraph_b2, k);
		this.embedding_vector_index = embedding_vector_index;
		
		//Prepare the buffers for the alignment matrixes
		this.alignement_matrixes = new double[k_with_windows_b1.length][k_with_windows_b2.length];
		
		int max_id = 0;
		for(int[] p : raw_paragraphs_b1) {
			for(int id : p) {
				if(id>max_id) {
					max_id = id;
				}
			}
		}
		for(int[] p : raw_paragraphs_b2) {//Second paragraph
			for(int id : p) {
				if(id>max_id) {
					max_id = id;
				}
			}
		}
		this.max_id = max_id;
		if(dense_global_matrix_buffer==null) {
			create_dense_matrix();
		}
		
		this.col_maxima = new double[k];
	}
	
	/**
	 * 
	 * @param raw_paragraphs all the paragraphs
	 * @param k - window size
	 * @return
	 */
	private int[][] create_windows(int[] raw_paragraph, final int k) {	
		int[][] windows = new int[raw_paragraph.length-k+1][k];//pre-allocate the storage space for the
		for(int i=0;i<windows.length;i++){
			//create one window
			for(int j=0;j<k;j++) {
				windows[i][j] = raw_paragraph[i+j];
			}
		}
		return windows;
	}
	
	private void print_special_configurations() {
		if(!MatchesWithEmbeddings.NORMALIZE_VECTORS) {
			System.err.println("!MatchesWithEmbeddings.NORMALIZE_VECTORS");
		}
		if(SAVE_MODE) {
			System.err.println("SAVE_MODE");
		}
	}
	
	public double[] run_baseline(){
		print_special_configurations();
		HungarianKevinStern solver = new HungarianKevinStern(k);
		System.out.println("Solutions.run_baseline() k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] local_similarity_matrix = new double[k][k];
		USE_GLOBAL_MATRIX = true;
		
		double[] run_times = new double[num_paragraphs];
			
		double stop,start;
		long count_computed_cells   = 0;
		long count_survived_pruning = 0;
		
		//Allocate space for the alignment matrix
		final double[][] alignment_matrix = this.alignement_matrixes;//get the pre-allocated buffer. Done in Constructor
		
		start = System.currentTimeMillis();
		final double[][] global_cost_matrix_book = fill_similarity_matrix();
		//For each pair of windows
		for(int line=0;line<alignment_matrix.length;line++) {							
			for(int column=0;column<alignment_matrix[0].length;column++) {	
				//Fill local matrix of the current window combination from global matrix 
				fill_local_similarity_matrix(local_similarity_matrix, global_cost_matrix_book, line, column);
				final double upper_bound_sim = get_sum_of_column_row_minima(local_similarity_matrix);
				if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold_times_k) {
					count_survived_pruning++;
					//That's the important line
					double similarity = -solver.solve(local_similarity_matrix, threshold);
					//normalize costs: Before it was distance. Now it is similarity.
					if(similarity>=threshold_times_k) {
						alignment_matrix[line][column] = similarity/(double)k;//normalize
						count_computed_cells++;
					}//else keep it zero
				}
			}
		}
		stop = System.currentTimeMillis();
		run_times[0] = (stop-start);		
		int size = size(alignment_matrix);
		double check_sum = sum(alignment_matrix);
		System.out.println("k="+k+"\t"+(stop-start)+"\tms\t"+check_sum+"\t"+size+"\t"+count_survived_pruning+"\t"+count_computed_cells);

		return run_times;
	}
	
	private double get_sum_of_column_row_minima(final double[][] similarity_matrix) {
		double row_sum = 0;
		Arrays.fill(this.col_maxima, Double.MAX_VALUE);
		for(int i=0;i<this.k;i++) {
			final double[] line = similarity_matrix[i];
			double row_min = Double.MAX_VALUE;
			for(int j=0;j<this.k;j++) {
				final double val = line[j];
				if(val<row_min) {
					row_min = val;
				}
				if(val<col_maxima[j]) {
					col_maxima[j] = val;
				}
			}
			row_sum += row_min;
		}
		double col_sum = sum(col_maxima);
		double max_similarity = -Math.max(row_sum, col_sum);		
		
		return max_similarity;
	}
	
	public double[] run_naive(){
		print_special_configurations();
		HungarianKevinStern solver = new HungarianKevinStern(k);
		System.out.println("Solutions.run_naive() k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] local_similarity_matrix = new double[k][k];
		USE_GLOBAL_MATRIX = false;
		
		double[] run_times = new double[num_paragraphs];
			
		double stop,start;
		long count_computed_cells = 0;
		
		//Allocate space for the alignment matrix
		final double[][] alignment_matrix = this.alignement_matrixes;//get the pre-allocated buffer. Done in Constructor
		
		start = System.currentTimeMillis();
		final double[][] global_cost_matrix_book = fill_similarity_matrix();
		//For each pair of windows
		for(int line=0;line<alignment_matrix.length;line++) {							
			for(int column=0;column<alignment_matrix[0].length;column++) {	
				//Fill local matrix of the current window combination from global matrix 
				fill_local_similarity_matrix(local_similarity_matrix, global_cost_matrix_book, line, column);
				//That's the important line
				final double similarity = -solver.solve(local_similarity_matrix, threshold);
				//normalize costs: Before it was distance. Now it is similarity.
				if(similarity>=threshold_times_k) {
					alignment_matrix[line][column] = similarity/(double)k;//normalize
					count_computed_cells++;
				}//else keep it zero
			}
		}
		stop = System.currentTimeMillis();
		run_times[0] = (stop-start);		
		int size = size(alignment_matrix);
		double check_sum = sum(alignment_matrix);
		System.out.println("k="+k+"\t"+(stop-start)+"\tms\t"+check_sum+"\t"+size+"\t"+count_computed_cells);

		return run_times;
	}
	
	
	private boolean is_equal(double val_1, double val_2) {
		if(val_1-DOUBLE_PRECISION_BOUND<val_2 && val_1+DOUBLE_PRECISION_BOUND>val_2) {
			return true;
		}
		return false;
	}
	private boolean is_equal(double[][] cost_matrix, double[][] cost_matrix_copy) {
		if(cost_matrix.length!=cost_matrix_copy.length) {
			return false;
		}
		if(cost_matrix[0].length!=cost_matrix_copy[0].length) {
			return false;
		}
		for(int line=0;line<cost_matrix.length;line++) {
			for(int column=0;column<cost_matrix[0].length;column++) {
				if(cost_matrix[line][column]!=cost_matrix_copy[line][column]) {
					return false;
				}
			}
		}
		return true;
	}
	//final boolean SAFE_MODE = true;
	
	static final boolean LOGGING_MODE = true;
	
	
	
	public double[] run_candidates() {
		print_special_configurations();
		HungarianKevinStern solver = new HungarianKevinStern(k);
		System.out.println("Solutions.run_candidates() k="+k+" threshold="+threshold+" "+solver.get_name());
		USE_GLOBAL_MATRIX = true;
		
		System.out.println("HungarianExperiment.run_candidates() k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] local_similarity_matrix = new double[k][k];
		
		double[] run_times = new double[num_paragraphs];
		
		final int[][] inverted_window_index_ranges = create_indexes();
		
		//System.err.println("sum(index) "+sum(inverted_window_index_ranges));
		
		double stop,start;
		
		//Allocate space for the alignment matrix
		final double[][] alignment_matrix = this.alignement_matrixes;//get the pre-allocated buffer. Done in Constructor
		final boolean[][] candidates = new boolean[alignment_matrix.length][alignment_matrix[0].length];
		
		count_candidates = 0;
		count_survived_sum_bound = 0;
		count_cells_exceeding_threshold = 0;
		
		start = System.currentTimeMillis();
		final double[][] global_matrix_book = fill_similarity_matrix();//For k<5 this does not pay off
		double start_candidates = System.currentTimeMillis();
		ArrayList<MyArrayList> all_candidates = new ArrayList<MyArrayList>(candidates.length);
		
		for(int line=0;line<alignment_matrix.length;line++) {
			final boolean[] candidates_line = candidates[line];
			final int[] window_p1 = k_with_windows_b1[line];
			
			//Fill all_candidates
			get_candidates(window_p1, candidates_line, inverted_window_index_ranges);
			
			MyArrayList candidates_condensed = new MyArrayList(candidates_line.length);
			int q = 0;
			boolean found_run = false;
			while(q<candidates_line.length) {
				if(candidates_line[q]) {//start of a run
					candidates_condensed.add(q);
					q++;
					found_run = true;
					while(q<candidates_line.length) {
						if(!candidates_line[q]){//end of run
							candidates_condensed.add(q-1);	
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
				candidates_condensed.add(candidates_line.length-1);
			}
			all_candidates.add(candidates_condensed);
		}
		double stop_candidates = System.currentTimeMillis();
		
		for(int line=0;line<alignment_matrix.length;line++) {
			//Validate candidates
			final double[] alignment_matrix_line = alignment_matrix[line];
			final MyArrayList candidates_condensed = all_candidates.get(line);
			
			final int size = candidates_condensed.size();
			final int[] raw_candidates = candidates_condensed.ARRAY;
			for(int c=0;c<size;c+=2) {
				final int run_start = raw_candidates[c];
				final int run_stop = raw_candidates[c+1];
				
				validate_run(solver, line, run_start, run_stop, local_similarity_matrix, global_matrix_book, alignment_matrix_line);
			}
		}
		
		stop = System.currentTimeMillis();
		run_times[0] = stop-start;
		
		int size = size(alignment_matrix);
		double check_sum = sum(alignment_matrix);
		System.out.println("k="+k+"\t"+(stop-start)+"\tms\t"+check_sum+"\t"+size+"\t"+count_candidates+"\t"+count_survived_sum_bound+"\t"+count_cells_exceeding_threshold+"\t"+(stop_candidates-start_candidates));
		
		return run_times;
	}
	
	/**
	 * inverted_index.get(token_id)[] -> List of all other other_token_id's with sim(token_id, other_token_id) > threshold, ordered asc by token_id.
	 * @param matrix global similarity matrix of the Books. 
	 */
	private ArrayList<MyArrayList> create_neihborhood_index(final double[][] matrix) {
		System.out.println("create_neihborhood_index() BEGIN");
		double start = System.currentTimeMillis();
		
		ArrayList<MyArrayList> indexes = new ArrayList<MyArrayList>(matrix.length);
		for(final double[] line : matrix) {
			MyArrayList index = new MyArrayList(line.length);//Ensure the list has enough memory reserved
			for(int id=0;id<line.length;id++) {
				final double sim = line[id];
				if(sim>=threshold){
					index.add(id);
				}
			}
			//TODO trim to size?
			indexes.add(index);
		}
		System.out.println("create_neihborhood_index() END in\t"+(System.currentTimeMillis()-start));
		return indexes;
	}
	
	//TODO exploit running window property: Maybe order ids by frequency
	private boolean is_in(MyArrayList neihborhood_index, int[] curr_window) {
		final int[] arr = neihborhood_index.ARRAY;
		for(int i=0;i<neihborhood_index.size();i++) {
			final int neighbor = arr[i];
			for(int t : curr_window) {
				if(t==neighbor) {
					return true;
				}
			}
		}
		return false;
	}
	
	/**
	 * indexes.get(token_id) -> int[] of window (positions) with token having sim(token_id, some token) > threshold
	 * @param k_with_windows
	 * @return
	 */
	private ArrayList<MyArrayList> create_inverted_window_index(final int[][] k_with_windows, final ArrayList<MyArrayList> neihborhood_indexes) {
		System.out.println("ArrayList<ArrayList<int[]>> create_inverted_window_index() BEGIN");
		final double start = System.currentTimeMillis();
		final ArrayList<MyArrayList> indexes = new ArrayList<MyArrayList>(max_id+1);//one index per token
		//For each token
		for(int token_id = 0;token_id<neihborhood_indexes.size();token_id++) {
			//Create the list of occurrences for token: token_id
			final MyArrayList neihborhood_index = neihborhood_indexes.get(token_id);
			
			final MyArrayList index_this_paragraph = new MyArrayList();
			for(int pos=0;pos<k_with_windows.length;pos++) {
				int[] curr_window = k_with_windows[pos];
				if(is_in(neihborhood_index, curr_window)) {
					index_this_paragraph.add(pos);
				}
			}
			indexes.add(index_this_paragraph);
		}
		System.out.println("ArrayList<ArrayList<int[]>> create_inverted_window_index() END in\t"+(System.currentTimeMillis()-start));
		return indexes;
	}
	
	private int[][] to_inverted_window_index_ranges(ArrayList<MyArrayList> inverted_window_index) {
		int[][] index = new int[inverted_window_index.size()][];
		for(int token_id=0;token_id<index.length;token_id++) {
			MyArrayList cells_above_threshold = inverted_window_index.get(token_id);
			int[] temp = make_dense(cells_above_threshold);
			index[token_id] = temp;
		}
		
		return index;
	}
	
	private int[] make_dense(final MyArrayList cells_above_threshold) {
		//start range
		int index = 0;
		final int size = cells_above_threshold.size();
		MyArrayList temp = new MyArrayList();
		
		while(index<size) {
			int start_range_index = cells_above_threshold.get(index);
			index++;
			//find end of range
			while(index<size && cells_above_threshold.get(index-1)+1==cells_above_threshold.get(index)) {
				index++;
			}
			int stop_range_index = cells_above_threshold.get(index-1);//The one before was the final element of the current range.
			temp.add(start_range_index);
			temp.add(stop_range_index);
			//index++;
		}
		return temp.getTrimmedArray();//XXX should we do that?
	}
	
	int[][] create_indexes() {
		/**
		 * inverted_index.get(my_token_id) -> ordered list of token_id's with sim(my_token_id, token_id) >= threshold 
		 */
		final ArrayList<MyArrayList> neighborhood_index = create_neihborhood_index(dense_global_matrix_buffer);
		//System.err.println("sum(neighborhood_index) "+sum(neighborhood_index));
		/**
		 * inverted_window_index.get(my_token_id) -> ordered list of cells containing some other token, s.t.  sim(my_token_id, token_id) >= threshold. I.e., this is a candidate. 
		 */
		final ArrayList<MyArrayList> inverted_window_index = create_inverted_window_index(k_with_windows_b2, neighborhood_index);
		//System.err.println("sum(inverted_window_index) "+sum(inverted_window_index));
		/**
		 * inverted_window_index_ranges[token_id][0] -> Start of first run (if there is one). The respective end is at inverted_window_index_ranges[token_id][1] etc.
		 */
		final int[][] inverted_window_index_ranges = to_inverted_window_index_ranges(inverted_window_index);
		//System.err.println("sum(inverted_window_index_ranges) "+sum(inverted_window_index_ranges));
		
		return inverted_window_index_ranges;
	}
	
	void get_candidates(final int[] window_b1, final boolean[] candidates_line, final int[][] inverted_window_index_ranges) {
		//Get candidates
		for(int id : window_b1) {
			final int[] index = inverted_window_index_ranges[id];
			
			for(int i=0;i<index.length;i+=2) {//Note the +=2 because we have pairs (start,stop)
				final int start_range = index[i];
				final int stop_range = index[i+1];//including this one
				for(int pos = start_range;pos<=stop_range;pos++) {
					candidates_line[pos] = true;
				}
			}
		}
	}
	
	void validate_run(HungarianKevinStern solver, final int line, final int run_start, final int run_stop
			, final double[][] local_similarity_matrix, final double[][] global_cost_matrix_book, final double[] alignment_matrix_line) {
		for(int column=run_start;column<=run_stop;column++) {
			count_candidates++;
			
			fill_local_similarity_matrix(local_similarity_matrix, global_cost_matrix_book, line, column);
			final double upper_bound_sim = get_sum_of_column_row_minima(local_similarity_matrix);
			if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold_times_k) {
				count_survived_sum_bound++;
				//That's the important line
				double similarity = -solver.solve(local_similarity_matrix, threshold);
				//normalize costs: Before it was distance. Now it is similarity.
				if(similarity>=threshold_times_k) {
					alignment_matrix_line[column] = similarity/k_double;//normalize
					count_cells_exceeding_threshold++;
				}//else keep it zero
			}
		}
	}
	
	
	public double[] run_incremental_cell_pruning_deep(){
		HungarianDeep solver = new HungarianDeep(k);
		System.out.println("Solutions.run_incremental_cell_pruning_deep() k="+k+" threshold="+threshold+" "+solver.get_name());
		
		double[] run_times = new double[num_paragraphs];
		
		final double MAX_SIM_ADDITION_NEW_NODE = 1.0/k_double;
		
		double stop,start;

		//Allocate space for the alignment matrix
		final double[][] alignment_matrix = alignement_matrixes;
		
		double prior_cell_similarity;
		double prev_min_value;
		
		int count_survived_pruning = 0;
		int count_survived_second_pruning = 0;
		int count_survived_third_pruning = 0;
		int count_cells_exceeding_threshold = 0;
		boolean prior_cell_updated_matrix;
		
		double ub_sum;
		double sim;
		
		USE_GLOBAL_MATRIX = true;
		final double[][] global_cost_matrix_book = fill_similarity_matrix_deep();//can be re-used for any k. Thus not part of the runtime.
		
		final double[][] current_lines = new double[k][];
		
		start = System.currentTimeMillis();
		
		//For each pair of windows
		for(int line=0;line<alignment_matrix.length;line++) {
			for(int i=0;i<k;i++) {
				current_lines[i] = global_cost_matrix_book[line+i];
			}
			solver.set_matrix(current_lines);
			if(LOGGING_MODE) count_survived_pruning++;
			if(LOGGING_MODE) count_survived_second_pruning++;
			//get the line to get rid of 2D array resolution
			final double[] alignment_matrix_line = alignment_matrix[line];
			
			int column=0;			
			{//Here we have no bound
				ub_sum = sum_bound_similarity(current_lines, column)/k_double;
				
				if(ub_sum+DOUBLE_PRECISION_BOUND>=threshold) {
					sim = -solver.solve(column, col_maxima);//Note the minus-trick for the Hungarian
					sim /= k_double;
					if(sim>=threshold) {
						if(LOGGING_MODE) count_cells_exceeding_threshold++;
						alignment_matrix_line[column] = sim;
					}//else keep it zero
					prior_cell_similarity = sim;
				}else{
					prior_cell_similarity = ub_sum;
				}
				
				prev_min_value = max(current_lines, column);
				prior_cell_updated_matrix = true;
				
			}
			
			//For all other columns: Here we have a bound
			for(column=1;column<alignment_matrix[0].length;column++) {		
				double upper_bound_sim = prior_cell_similarity + MAX_SIM_ADDITION_NEW_NODE;
				if(prior_cell_updated_matrix) {
					upper_bound_sim-= (prev_min_value / k_double);
				}				
				
				if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
					if(LOGGING_MODE) count_survived_pruning++;  
					
					double max_sim_new_node = min(current_lines, column);
					upper_bound_sim-=MAX_SIM_ADDITION_NEW_NODE;
					upper_bound_sim+=(max_sim_new_node/k_double);
					 
					if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
						if(LOGGING_MODE) count_survived_second_pruning++;
						
						ub_sum = sum_bound_similarity(current_lines, column)/k_double;
						upper_bound_sim = (ub_sum<upper_bound_sim) ? ub_sum : upper_bound_sim;//The some bound is not necessarily tighter
						
						if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {	
							if(LOGGING_MODE) count_survived_third_pruning++;
							//That's the important line
							sim = -solver.solve(column, col_maxima);//Note the minus-trick for the Hungarian
							//normalize 
							sim /= k_double;
							
							if(sim>=threshold) {
								if(LOGGING_MODE) count_cells_exceeding_threshold++;
								alignment_matrix_line[column] = sim;
							}//else keep it zero
							prior_cell_similarity = sim;
							
						}else{
							prior_cell_similarity = upper_bound_sim;
						}
					}else{
						prior_cell_similarity = upper_bound_sim;
					}
					prev_min_value = max(current_lines, column);
					prior_cell_updated_matrix = true;
				}else{
					prior_cell_similarity = upper_bound_sim;
					prior_cell_updated_matrix = false;
				}
			}
		}
		stop = System.currentTimeMillis();
		run_times[0] = (stop-start);
		int size = size(alignment_matrix);
		double check_sum = sum(alignment_matrix);
		System.out.println("k="+k+"\t"+(stop-start)+"\tms\t"+check_sum+"\t"+size+"\t"+count_survived_pruning+"\t"+count_survived_second_pruning+"\t"+count_survived_third_pruning+"\t"+count_cells_exceeding_threshold);
	
		return run_times;
	}
	
	private void change_signum(double[][] matrix) {
		for(double[] arr : matrix) {
			for(int i=0;i<arr.length;i++) {
				arr[i] *= -1;
			}
		}
		
	}

	public double[] run_incremental_cell_pruning(){
		HungarianKevinStern solver = new HungarianKevinStern(k);
		HungarianKevinStern solver_baseline = new HungarianKevinStern(k);
		
		System.out.println("Solutions.run_incremental_cell_pruning() k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] local_similarity_matrix = new double[k][k];
		
		double[] run_times = new double[num_paragraphs];
		
		final double MAX_SIM_ADDITION_NEW_NODE = 1.0/k;
		
		double stop,start;

		//Allocate space for the alignment matrix
		final double[][] alignment_matrix = alignement_matrixes;
		final int[][] k_windows_p1 = k_with_windows_b1;
		final int[][] k_windows_p2 = k_with_windows_b2;	
		
		double prior_cell_similarity;
		double prev_min_value;
		
		int count_survived_pruning = 0;
		int count_survived_second_pruning = 0;
		int count_survived_third_pruning = 0;
		int count_cells_exceeding_threshold = 0;
		boolean prior_cell_updated_matrix;
		
		double ub_sum;
		double sim;
		
		start = System.currentTimeMillis();
		//For each pair of windows
		for(int line=0;line<alignment_matrix.length;line++) {
			count_survived_pruning++;
			count_survived_second_pruning++;
			//get the line to get rid of 2D array resolution
			final double[] alignment_matrix_line = alignment_matrix[line];
			
			int column=0;			
			{//Here we have no bound
				fill_local_similarity_matrix(k_windows_p1[line], k_windows_p2[column], local_similarity_matrix); 
				ub_sum = sum_bound_similarity(local_similarity_matrix)/(double)k;
				
				if(ub_sum+DOUBLE_PRECISION_BOUND>=threshold) {
					sim = -solver.solve_inj(local_similarity_matrix, threshold, col_maxima);//Note the minus-trick for the Hungarian
					sim /= k;
					if(sim>=threshold) {
						count_cells_exceeding_threshold++;
						alignment_matrix_line[column] = sim;
					}//else keep it zero
					prior_cell_similarity = sim;
				}else{
					prior_cell_similarity = ub_sum;
				}
				
				prev_min_value = max(local_similarity_matrix);
				prior_cell_updated_matrix = true;
			}
			
			//For all other columns: Here we have a bound
			for(column=1;column<alignment_matrix[0].length;column++) {		
				double upper_bound_sim = prior_cell_similarity + MAX_SIM_ADDITION_NEW_NODE;
				if(prior_cell_updated_matrix) {
					upper_bound_sim-= (prev_min_value / k);
				}				
				
				if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
					count_survived_pruning++;
					
					fill_local_similarity_matrix(k_windows_p1[line], k_windows_p2[column], local_similarity_matrix);  
					
					double max_sim_new_node = min(local_similarity_matrix);
					upper_bound_sim-=MAX_SIM_ADDITION_NEW_NODE;
					upper_bound_sim+=(max_sim_new_node/k);
					 
					if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
						count_survived_second_pruning++;
						
						ub_sum = sum_bound_similarity(local_similarity_matrix)/k;
						upper_bound_sim = (ub_sum<upper_bound_sim) ? ub_sum : upper_bound_sim;//The some bound is not necessarily tighter
						
						if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {	
							count_survived_third_pruning++;
							//That's the important line
							sim = -solver.solve_inj(local_similarity_matrix, threshold, col_maxima);//Note the minus-trick for the Hungarian
							//normalize 
							sim /= k;
							
							if(sim>=threshold) {
								count_cells_exceeding_threshold++;
								alignment_matrix_line[column] = sim;
							}//else keep it zero
							prior_cell_similarity = sim;
							
						}else{
							prior_cell_similarity = upper_bound_sim;
						}
					}else{
						prior_cell_similarity = upper_bound_sim;
					}
					prev_min_value = max(local_similarity_matrix);
					prior_cell_updated_matrix = true;
				}else{
					prior_cell_similarity = upper_bound_sim;
					prior_cell_updated_matrix = false;
				}
			}
		}
		stop = System.currentTimeMillis();
		run_times[0] = (stop-start);
		int size = size(alignment_matrix);
		double check_sum = sum(alignment_matrix);
		System.out.println("k="+k+"\t"+(stop-start)+"\tms\t"+check_sum+"\t"+size+"\t"+count_survived_pruning+"\t"+count_survived_second_pruning+"\t"+count_survived_third_pruning+"\t"+count_cells_exceeding_threshold);
	
		return run_times;
	}
	
	final void fill_local_similarity_matrix_incrementally(final int[] k_window_p1, final int[] k_window_p2, final double[][] local_similarity_matrix){
		final int copy_length = k-1;
		
		col_sum-=col_maxima[0];
		System.arraycopy(col_maxima, 1, col_maxima, 0, copy_length);
		col_maxima[copy_length] = Double.MAX_VALUE;
		
		for(int i=0;i<k;i++) {
			System.arraycopy(local_similarity_matrix[i], 1, local_similarity_matrix[i], 0, copy_length);
			final int token_id_1 = k_window_p1[i];
			final int token_id_2 = k_window_p2[copy_length];
			double sim = sim_cached(token_id_1, token_id_2);
			local_similarity_matrix[i][copy_length] = -sim;//Note the minus-trick for the Hungarian
			if(-sim<col_maxima[copy_length]) {
				col_maxima[copy_length]=-sim;
			}
			
		}
		col_sum+=col_maxima[copy_length];
	}
	
	final double sim_cached(final int token_id_1, final int token_id_2) {
		return (token_id_1==token_id_2) ? EQUAL : dense_global_matrix_buffer[token_id_1][token_id_2]; 
	}
	
	double col_sum;
	private double sum_bound_similarity(final double[][] similarity_matrix) {
		double row_sum = 0;
		Arrays.fill(col_maxima, Double.POSITIVE_INFINITY);
		for(int i=0;i<this.k;i++) {
			final double[] line = similarity_matrix[i];
			double row_min = Double.POSITIVE_INFINITY;
			for(int j=0;j<this.k;j++) {
				final double val = line[j];
				if(val<row_min) {
					row_min = val;
				}
				if(val<col_maxima[j]) {
					col_maxima[j] = val;
				}
			}
			row_sum += row_min;
		}
		col_sum = sum(col_maxima);
		double min_cost = Math.max(row_sum, col_sum);		
		
		return -min_cost;
	}
	private double sum_bound_similarity(final double[][] current_lines, final int offset) {
		double row_sum = 0;
		Arrays.fill(col_maxima, Double.POSITIVE_INFINITY);
		for(int i=0;i<this.k;i++) {
			final double[] line = current_lines[i];
			double row_min = Double.POSITIVE_INFINITY;
			for(int j=0;j<this.k;j++) {
				final double val = line[offset+j];
				if(val<row_min) {
					row_min = val;
				}
				if(val<col_maxima[j]) {
					col_maxima[j] = val;
				}
			}
			row_sum += row_min;
		}
		col_sum = sum(col_maxima);
		double min_cost = Math.max(row_sum, col_sum);		
		
		return -min_cost;
	}
	
	private final double max(final double[][] current_lines, final int offset) {
		double max = Double.NEGATIVE_INFINITY;//TODO remove this line?
		for(double[] line : current_lines) {
			if(max<line[offset+0]) {//similarity of the deleted token
				max=line[offset+0];
			}
		}
		return -max;
	}
	
	private final double max(final double[][] local_similarity_matrix) {
		double max = Double.NEGATIVE_INFINITY;//TODO remove this line?
		for(double[] line : local_similarity_matrix) {
			if(max<line[0]) {//similarity of the deleted token
				max=line[0];
			}
		}
		return -max;
	}
	
	private final double min(final double[][] current_lines, final int offset) {
		double min = current_lines[0][offset+k-1];
		for(int line=1;line<k;line++) {
			if(min>current_lines[line][offset+k-1]) {
				min=current_lines[line][offset+k-1];
			}
		}
		return -min;
	}
	
	private final double min(final double[][] window) {
		double min = window[0][k-1];
		for(int line=1;line<window.length;line++) {
			if(min>window[line][k-1]) {
				min=window[line][k-1];
			}
		}
		return -min;
	}
	
	final void fill_local_similarity_matrix(final int[] k_window_p1, final int[] k_window_p2, final double[][] local_similarity_matrix){
		for(int i=0;i<k;i++) {
			final int token_id_1 = k_window_p1[i];
			final double[] matrix_line = dense_global_matrix_buffer[token_id_1]; 
			for(int j=0;j<this.k;j++) {
				final int token_id_2 = k_window_p2[j];
				double sim = (token_id_1==token_id_2) ? EQUAL : matrix_line[token_id_2];
				local_similarity_matrix[i][j] = -sim;//Note the minus-trick for the Hungarian
			}
		}
	}
	
	
	private void out(String name, double[][] cost_matrix) {
		System.out.println(name);
		for(double[] arr : cost_matrix) {
			for(double d: arr) {
				System.out.print(d+"\t");
			}
			System.out.println();
		}
	}
	
	private void out(int[] arr) {
		for(int d : arr) {
			System.out.print(d+"\t");
		}
		System.out.println();
	}
	
	private void out(double[] arr) {
		for(double d : arr) {
			System.out.println(d+"\t");
		}
	}
	
	private double min_cost_first_node(double[][] local_similarity_matrix) {
		double min = local_similarity_matrix[0][0];
		for(int line=1;line<k;line++) {
			final double val = local_similarity_matrix[line][0];
			if(val<min) {
				min= val;
			}
		}
		return min;
	}
	
	private double min_cost_new_node(double[][] local_similarity_matrix) {
		double min = local_similarity_matrix[0][k-1];
		for(int line=1;line<k;line++) {
			final double val = local_similarity_matrix[line][k-1];
			if(val<min) {
				min= val;
			}
		}
		return min;
	}
	
	private void fill_local_similarity_matrix(final double[][] local_cost_matrix, final double[][] global_cost_matrix_book, final int line, final int column) {
		for(int i=0;i<this.k;i++) {
			for(int j=0;j<this.k;j++) {
				local_cost_matrix[i][j] = -global_cost_matrix_book[line+i][column+j];//XXX - Note the minus for the Hungarian
			}
		}
	}
	
	private int size(double[][] alignment_matrix) {
		return alignment_matrix.length*alignment_matrix[0].length;
	}
	private long sum(ArrayList<MyArrayList> to_sum_up) {
		long sum = 0;
		for(MyArrayList array : to_sum_up) {
			for(int i=0;i<array.size();i++) {
				sum+=array.get(i);
			}
		}
		return sum;
	}
	private long sum(int[][] alignment_matrix) {
		long sum = 0;
		for(int[] array : alignment_matrix) {
			for(long d : array) {
				sum+=d;
			}
		}
		return sum;
	}
	private double sum(double[][] alignment_matrix) {
		double sum = 0;
		for(double[] array : alignment_matrix) {
			for(double d : array) {
				sum+=d;
			}
		}
		return sum;
	}
	private double sum(double[] array) {
		double sum = 0;
		for(double d : array) {
			sum+=d;
		}
		return sum;
	}
	
	static boolean USE_GLOBAL_MATRIX = false;
	private double[][] fill_similarity_matrix() {
		final double[][] global_cost_matrix = new double[raw_paragraph_b1.length][raw_paragraph_b2.length];
		
		if(USE_GLOBAL_MATRIX) {
			for(int line=0;line<raw_paragraph_b1.length;line++) {
				final int set_id_window_p1 = raw_paragraph_b1[line];
				final double[] sim_matrix_line = dense_global_matrix_buffer[set_id_window_p1];
				for(int column=0;column<raw_paragraph_b2.length;column++) {
					final int set_id_window_p2 = raw_paragraph_b2[column];	
					final double sim = sim_matrix_line[set_id_window_p2];
					global_cost_matrix[line][column] = sim;
				}
			}
		}else{
			for(int line=0;line<raw_paragraph_b1.length;line++) {
				final int set_id_window_p1 = raw_paragraph_b1[line];
				final double[] vec_1 = this.embedding_vector_index.get(set_id_window_p1);
				for(int column=0;column<raw_paragraph_b2.length;column++) {
					final int set_id_window_p2 = raw_paragraph_b2[column];
					final double[] vec_2 = this.embedding_vector_index.get(set_id_window_p2);
					final double sim = sim(set_id_window_p1,set_id_window_p2,vec_1,vec_2);
					global_cost_matrix[line][column] = sim;
				}
			}
		}
		
		return global_cost_matrix;
	}
	
	private double[][] fill_similarity_matrix_deep() {
		final double[][] matrix = new double[raw_paragraph_b1.length][raw_paragraph_b2.length];
		
		for(int line=0;line<raw_paragraph_b1.length;line++) {
			final int set_id_window_p1 = raw_paragraph_b1[line];
			final double[] sim_matrix_line = dense_global_matrix_buffer[set_id_window_p1];
			for(int column=0;column<raw_paragraph_b2.length;column++) {
				final int set_id_window_p2 = raw_paragraph_b2[column];	
				final double sim = sim_matrix_line[set_id_window_p2];
				matrix[line][column] = -sim;//XXX - that is the difference to the method above
			}
		}

		return matrix;
	}
	
	private static final double EQUAL = 1;
	private static final double MIN_SIM = 0;
	
	private static double sim(final int set_id1, final int set_id2, final double[] vec_1, final double[] vec_2) {
		if(set_id1==set_id2) {
			return EQUAL;
		}else if(vec_1==null || vec_2==null){//may happen e.g., for stop words
			return MIN_SIM;
		}
		return cosine_similarity(vec_1, vec_2);
	}
	
	/**
	 * Expects that vectors are normalized to unit length.
	 * 
	 * @param vectorA
	 * @param vectorB
	 * @return
	 */
	private static double cosine_similarity(final double[] vectorA, final double[] vectorB) {
		double dotProduct = 0.0;
	    for (int i = 0; i < vectorA.length; i++) {
	        dotProduct += vectorA[i] * vectorB[i];
	    }
	    
	    dotProduct = (dotProduct < 0) ? 0 : dotProduct;
	    dotProduct = (dotProduct > 1) ? 1 : dotProduct;
	    return dotProduct;
	}
	
	private void create_dense_matrix() {
		double start = System.currentTimeMillis();
		dense_global_matrix_buffer = new double[max_id+1][max_id+1];//This is big....
		for(int line_id=0;line_id<dense_global_matrix_buffer.length;line_id++) {
			dense_global_matrix_buffer[line_id][line_id] = EQUAL;
			final double[] vec_1 = this.embedding_vector_index.get(line_id);
			for(int col_id=line_id+1;col_id<dense_global_matrix_buffer[0].length;col_id++) {//Exploits symmetry
				final double[] vec_2 = this.embedding_vector_index.get(col_id);
				double sim = sim(line_id, col_id, vec_1, vec_2);
				dense_global_matrix_buffer[line_id][col_id] = sim;
				dense_global_matrix_buffer[col_id][line_id] = sim;
			}
		}
		double stop = System.currentTimeMillis();
		double check_sum = sum(dense_global_matrix_buffer);
		int size = dense_global_matrix_buffer.length*dense_global_matrix_buffer[0].length;
		
		System.out.println("create_dense_matrix()\t"+(stop-start)+" check sum=\t"+check_sum+" size="+size);
	}
}
