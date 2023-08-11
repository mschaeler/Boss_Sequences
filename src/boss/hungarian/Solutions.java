package boss.hungarian;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

import boss.embedding.MatchesWithEmbeddings;

public class Solutions {
	static double[][] dense_global_matrix_buffer = null;
	static final double DOUBLE_PRECISION_BOUND = 0.0001d;
	
	final int k;
	final int num_paragraphs;
	final double threshold;
	final int max_id;
	final double dist_threshold;
	
	final int[][] k_with_windows_b1;
	final int[][] k_with_windows_b2;
	
	final int[] raw_paragraph_b1;
	final int[] raw_paragraph_b2;
	
	final double[][] alignement_matrixes;
	
	final HashMap<Integer, double[]> embedding_vector_index;
	
	final double[] col_minima;
	
	public Solutions(ArrayList<int[]> raw_paragraphs_b1, ArrayList<int[]> raw_paragraphs_b2, final int k, final double threshold, HashMap<Integer, double[]> embedding_vector_index) {
		this.k = k;
		this.threshold = threshold;
		this.dist_threshold = (1-threshold) * k;
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
		if(dense_global_matrix_buffer==null) {//XXX Has extra runtime measurement inside method
			create_dense_matrix();
		}
		
		this.col_minima = new double[k];
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
	
	void print_special_configurations() {
		if(!MatchesWithEmbeddings.NORMALIZE_VECTORS) {
			System.err.println("!MatchesWithEmbeddings.NORMALIZE_VECTORS");
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
		final double[][] global_cost_matrix_book = fill_cost_matrix();
		//For each pair of windows
		for(int line=0;line<alignment_matrix.length;line++) {							
			for(int column=0;column<alignment_matrix[0].length;column++) {	
				//Fill local matrix of the current window combination from global matrix 
				fill_local_cost_matrix(local_cost_matrix, global_cost_matrix_book, line, column);
				final double lower_bound_cost = get_sum_of_column_row_minima(local_cost_matrix);
				if(lower_bound_cost+DOUBLE_PRECISION_BOUND<=dist_threshold) {
					count_survived_pruning++;
					//That's the important line
					double cost = solver.solve(local_cost_matrix, threshold);
					//normalize costs: Before it was distance. Now it is similarity.
					if(cost<=dist_threshold) {
						alignment_matrix[line][column] = cost_to_normalized_similarity(cost);
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
	
	private double get_sum_of_column_row_minima(final double[][] cost_matrix) {
		double row_sum = 0;
		Arrays.fill(this.col_minima, Double.MAX_VALUE);
		for(int i=0;i<this.k;i++) {
			final double[] line = cost_matrix[i];
			double row_min = Double.MAX_VALUE;
			for(int j=0;j<this.k;j++) {
				final double val = line[j];
				if(val<row_min) {
					row_min = val;
				}
				if(val<col_minima[j]) {
					col_minima[j] = val;
				}
			}
			row_sum += row_min;
		}
		double col_sum = sum(col_minima);
		double min_cost = Math.max(row_sum, col_sum);		
		
		return min_cost;
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
		final double[][] global_cost_matrix_book = fill_cost_matrix();
		//For each pair of windows
		for(int line=0;line<alignment_matrix.length;line++) {							
			for(int column=0;column<alignment_matrix[0].length;column++) {	
				//Fill local matrix of the current window combination from global matrix 
				fill_local_cost_matrix(local_cost_matrix, global_cost_matrix_book, line, column);
				//That's the important line
				double cost = solver.solve(local_cost_matrix, threshold);
				//normalize costs: Before it was distance. Now it is similarity.
				if(cost<=dist_threshold) {
					alignment_matrix[line][column] = cost_to_normalized_similarity(cost);
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
	
	void fill_local_cost_matrix(final double[][] local_cost_matrix, final double[][] global_cost_matrix_book, final int line, final int column) {
		for(int i=0;i<this.k;i++) {
			for(int j=0;j<this.k;j++) {
				local_cost_matrix[i][j] = global_cost_matrix_book[line+i][column+j];
			}
		}
	}
	
	double cost_to_normalized_similarity(final double cost) {
		return 1.0 - (cost / (double)k);
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
	private double[][] fill_cost_matrix() {
		final double[][] global_cost_matrix = new double[raw_paragraph_b1.length][raw_paragraph_b2.length];
		
		if(USE_GLOBAL_MATRIX) {
			for(int line=0;line<raw_paragraph_b1.length;line++) {
				final int set_id_window_p1 = raw_paragraph_b1[line];
				final double[] cost_matrix_line = dense_global_matrix_buffer[set_id_window_p1];
				for(int column=0;column<raw_paragraph_b2.length;column++) {
					final int set_id_window_p2 = raw_paragraph_b2[column];	
					final double dist = cost_matrix_line[set_id_window_p2];
					global_cost_matrix[line][column] = dist;
				}
			}
		}else{
			for(int line=0;line<raw_paragraph_b1.length;line++) {
				final int set_id_window_p1 = raw_paragraph_b1[line];
				final double[] vec_1 = this.embedding_vector_index.get(set_id_window_p1);
				for(int column=0;column<raw_paragraph_b2.length;column++) {
					final int set_id_window_p2 = raw_paragraph_b2[column];
					final double[] vec_2 = this.embedding_vector_index.get(set_id_window_p2);
					final double dist = dist(set_id_window_p1,set_id_window_p2,vec_1,vec_2);
					global_cost_matrix[line][column] = dist;
				}
			}
		}
		
		return global_cost_matrix;
	}
	
	private static final double EQUAL = 0;
	private static final double MAX_DIST = 1;
	public static double dist(final int set_id1, final int set_id2, final double[] vec_1, final double[] vec_2) {
		if(set_id1==set_id2) {
			return EQUAL;
		}else if(vec_1==null || vec_2==null){//may happen e.g., for stop words
			return MAX_DIST;
		}
		return cosine_distance(vec_1, vec_2);
	}
	
	/**
	 * Expects that vectors are normalized to unit length.
	 * 
	 * @param vectorA
	 * @param vectorB
	 * @return
	 */
	static double cosine_distance(final double[] vectorA, final double[] vectorB) {
		double dotProduct = 0.0;
	    for (int i = 0; i < vectorA.length; i++) {
	        dotProduct += vectorA[i] * vectorB[i];
	    }
	    double dist = 1-dotProduct;
	    dist = (dist < 0) ? 0 : dist;
	    dist = (dist > 1) ? 1 : dist;
	    return dist;
	}
	
	private void create_dense_matrix() {
		double start = System.currentTimeMillis();
		dense_global_matrix_buffer = new double[max_id+1][max_id+1];//This is big....
		for(int line_id=0;line_id<dense_global_matrix_buffer.length;line_id++) {
			final double[] vec_1 = this.embedding_vector_index.get(line_id);
			for(int col_id=line_id+1;col_id<dense_global_matrix_buffer[0].length;col_id++) {//Exploits symmetry
				final double[] vec_2 = this.embedding_vector_index.get(col_id);
				double dist = dist(line_id, col_id, vec_1, vec_2);
				dense_global_matrix_buffer[line_id][col_id] = dist;
				dense_global_matrix_buffer[col_id][line_id] = dist;
			}
		}
		double stop = System.currentTimeMillis();
		double check_sum = sum(dense_global_matrix_buffer);
		int size = dense_global_matrix_buffer.length*dense_global_matrix_buffer[0].length;
		
		System.out.println("create_dense_matrix()\t"+(stop-start)+" check sum=\t"+check_sum+" size="+size);
	}
}
