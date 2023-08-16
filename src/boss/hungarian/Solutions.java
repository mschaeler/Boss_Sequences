package boss.hungarian;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import boss.embedding.MatchesWithEmbeddings;

public class Solutions {
	static double[][] dense_global_matrix_buffer = null;
	static final double DOUBLE_PRECISION_BOUND = 0.0001d;
	private static final boolean SAVE_MODE = false;
	
	final int k;
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
		final double[][] local_cost_matrix = new double[k][k];
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
				fill_local_similarity_matrix(local_cost_matrix, global_cost_matrix_book, line, column);
				final double upper_bound_sim = get_sum_of_column_row_minima(local_cost_matrix);
				if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold_times_k) {
					count_survived_pruning++;
					//That's the important line
					double similarity = -solver.solve(local_cost_matrix, threshold);
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
		final double[][] local_cost_matrix = new double[k][k];
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
				fill_local_similarity_matrix(local_cost_matrix, global_cost_matrix_book, line, column);
				//That's the important line
				final double similarity = -solver.solve(local_cost_matrix, threshold);
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
	
	public double[] run_incremental_cell_pruning_deep(){
		HungarianDeep solver = new HungarianDeep(k);
		HungarianKevinStern solver_baseline = new HungarianKevinStern(k);
		
		System.out.println("Solutions.run_incremental_cell_pruning_deep() k="+k+" threshold="+threshold+" "+solver.get_name());
		//final double[][] local_similarity_matrix = new double[k][k];
		
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
			count_survived_pruning++;
			count_survived_second_pruning++;
			//get the line to get rid of 2D array resolution
			final double[] alignment_matrix_line = alignment_matrix[line];
			
			int column=0;			
			{//Here we have no bound
				ub_sum = sum_bound_similarity(current_lines, column)/(double)k;
				
				if(ub_sum+DOUBLE_PRECISION_BOUND>=threshold) {
					sim = -solver.solve(column, col_maxima);//Note the minus-trick for the Hungarian
					sim /= k;
					if(sim>=threshold) {
						count_cells_exceeding_threshold++;
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
					upper_bound_sim-= (prev_min_value / k);
				}				
				
				if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
					count_survived_pruning++;  
					
					double max_sim_new_node = min(current_lines, column);
					upper_bound_sim-=MAX_SIM_ADDITION_NEW_NODE;
					upper_bound_sim+=(max_sim_new_node/k);
					 
					if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
						count_survived_second_pruning++;
						
						ub_sum = sum_bound_similarity(current_lines, column)/k;
						upper_bound_sim = (ub_sum<upper_bound_sim) ? ub_sum : upper_bound_sim;//The some bound is not necessarily tighter
						
						if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {	
							count_survived_third_pruning++;
							//That's the important line
							sim = -solver.solve(column, col_maxima);//Note the minus-trick for the Hungarian
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
