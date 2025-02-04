package boss.hungarian;

import java.util.ArrayList;
import java.util.Arrays;
import boss.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;

import boss.embedding.MatchesWithEmbeddings;
import boss.util.Config;
import boss.util.MyArrayList;
import boss.util.Util;

public class Solutions {
	//final boolean SAFE_MODE = true;
	
	static final boolean LOGGING_MODE = true;
	
	public long count_candidates 				= 0;
	public long count_survived_sum_bound 		= 0;
	public long count_cells_exceeding_threshold = 0;
	public long count_survived_pruning 		= 0;
	public long count_survived_second_pruning 	= 0;
	public long count_survived_third_pruning 	= 0;
	
	public static double[][] dense_global_matrix_buffer = null;
	static final double DOUBLE_PRECISION_BOUND = 0.0001d;
	private static final boolean SAVE_MODE = false;
	
	final double MAX_SIM_ADDITION_NEW_NODE; 
	
	public final int k;
	final double k_double;
	final int num_paragraphs;
	public final double threshold;
	final int max_id;
	final double threshold_times_k;
	
	public final int[][] k_with_windows_b1;
	public final int[][] k_with_windows_b2;
	
	public final int[] raw_paragraph_b1;
	public final int[] raw_paragraph_b2;
	
	public final double[][] alignement_matrix;
	
	final HashMap<Integer, double[]> embedding_vector_index;
	public final int[] tokens_b1;
	public final int[] tokens_b2;
	
	final MatrixRingBuffer mrb; 
	
	/**
	 * Contains the maximum column similarity of current local similarity matrix. Note, since we negate the signum for the hungarian. It's the minimum....
	 */
	final double[] col_maxima;	
	double sum_cols;
	
	public Solutions(ArrayList<int[]> raw_paragraphs_b1, ArrayList<int[]> raw_paragraphs_b2, final int k, final double threshold, HashMap<Integer, double[]> embedding_vector_index) {
		this.k = k;
		this.k_double = (double) k;
		this.threshold = threshold;
		this.threshold_times_k = threshold * k;
		this.num_paragraphs = raw_paragraphs_b1.size();
		if(num_paragraphs!=1) {
			System.err.println("Solutions() - Expecting book granularity");
		}
		MAX_SIM_ADDITION_NEW_NODE = 1.0 / k_double; 
		
		this.raw_paragraph_b1 = raw_paragraphs_b1.get(0);
		this.raw_paragraph_b2 = raw_paragraphs_b2.get(0);
		this.k_with_windows_b1 = create_windows(raw_paragraph_b1, k);
		this.k_with_windows_b2 = create_windows(raw_paragraph_b2, k);
		this.embedding_vector_index = embedding_vector_index;
		this.mrb = new MatrixRingBuffer(k);
		
		//Prepare the buffers for the alignment matrixes
		this.alignement_matrix = new double[k_with_windows_b1.length][k_with_windows_b2.length];
		
		HashSet<Integer> tokens_b1 = new HashSet<Integer>();
		HashSet<Integer> tokens_b2 = new HashSet<Integer>();
		
		int max_id = 0;
		for(int[] p : raw_paragraphs_b1) {
			for(int id : p) {
				tokens_b1.add(id);
				if(id>max_id) {
					max_id = id;
				}
			}
		}
		this.tokens_b1 = new int[tokens_b1.size()];
		int i=0;
		for(int id : tokens_b1) {
			this.tokens_b1[i++] = id;
		}
		Arrays.sort(this.tokens_b1);
		
		for(int[] p : raw_paragraphs_b2) {//Second paragraph
			for(int id : p) {
				tokens_b2.add(id);
				if(id>max_id) {
					max_id = id;
				}
			}
		}
		
		this.tokens_b2 = new int[tokens_b2.size()];
		i=0;
		for(int id : tokens_b2) {
			this.tokens_b2[i++] = id;
		}
		Arrays.sort(this.tokens_b2);
		
		this.max_id = max_id;
		if(dense_global_matrix_buffer==null) {
			create_dense_matrix();
		}
		
		this.col_maxima = new double[k];
	}
	
	/**
	 * For Jaccard only
	 * 
	 * @param raw_paragraphs_b1
	 * @param raw_paragraphs_b2
	 * @param k
	 * @param threshold
	 * @param embedding_vector_index
	 */
	public Solutions(ArrayList<int[]> raw_paragraphs_b1, ArrayList<int[]> raw_paragraphs_b2, final int k, final double threshold) {
		this.k = k;
		this.k_double = (double) k;
		this.threshold = threshold;
		this.threshold_times_k = threshold * k;
		this.num_paragraphs = raw_paragraphs_b1.size();
		if(num_paragraphs!=1) {
			System.err.println("Solutions() - Expecting book granularity");
		}
		MAX_SIM_ADDITION_NEW_NODE = 1.0 / k_double; 
		
		this.raw_paragraph_b1 = raw_paragraphs_b1.get(0);
		this.raw_paragraph_b2 = raw_paragraphs_b2.get(0);
		this.k_with_windows_b1 = create_windows(raw_paragraph_b1, k);
		this.k_with_windows_b2 = create_windows(raw_paragraph_b2, k);
		this.embedding_vector_index = null;
		this.mrb = new MatrixRingBuffer(k);
		
		//Prepare the buffers for the alignment matrixes
		this.alignement_matrix = new double[k_with_windows_b1.length][k_with_windows_b2.length];
		
		HashSet<Integer> tokens_b1 = new HashSet<Integer>();
		HashSet<Integer> tokens_b2 = new HashSet<Integer>();
		
		int max_id = 0;
		for(int[] p : raw_paragraphs_b1) {
			for(int id : p) {
				tokens_b1.add(id);
				if(id>max_id) {
					max_id = id;
				}
			}
		}
		this.tokens_b1 = new int[tokens_b1.size()];
		int i=0;
		for(int id : tokens_b1) {
			this.tokens_b1[i++] = id;
		}
		Arrays.sort(this.tokens_b1);
		
		for(int[] p : raw_paragraphs_b2) {//Second paragraph
			for(int id : p) {
				tokens_b2.add(id);
				if(id>max_id) {
					max_id = id;
				}
			}
		}
		
		this.tokens_b2 = new int[tokens_b2.size()];
		i=0;
		for(int id : tokens_b2) {
			this.tokens_b2[i++] = id;
		}
		Arrays.sort(this.tokens_b2);
		
		this.max_id = max_id;
		
		this.col_maxima = new double[k];
	}
	
	/**
	 * 
	 * @param raw_paragraphs all the paragraphs
	 * @param k - window size
	 * @return
	 */
	private int[][] create_windows(int[] raw_paragraph, final int k) {	
		int[][] windows; 
		if(raw_paragraph.length-k+1<0) {
			System.err.println("Solutions.create_windows(): raw_paragraph.length-k+1<0");
			windows = new int[1][];
			windows[0] = raw_paragraph.clone();
		}else{
			windows = new int[raw_paragraph.length-k+1][k];//pre-allocate the storage space for the
			for(int i=0;i<windows.length;i++){
				//create one window
				for(int j=0;j<k;j++) {
					windows[i][j] = raw_paragraph[i+j];
				}
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
	
	public double[] run_bound_tightness_exp(){
		print_special_configurations();
		HungarianKevinStern solver = new HungarianKevinStern(k);
		System.out.println("Solutions.run_bound_tightness_exp() k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] local_similarity_matrix = new double[k][k];
		USE_GLOBAL_MATRIX = true;
		
		double[] run_times = new double[num_paragraphs];
			
		double stop,start;
		long count_computed_cells   = 0;
		long count_survived_pruning = 0;
		long count_num_candidates   = 0;
		long count_survived_o_1_bound = 0;
		long count_survived_o_k_bound = 0;
		double prior_cell_similarity = 1.0;//Some value
		double prev_min_value = 0.0;
		//Allocate space for the alignment matrix
		final double[][] alignment_matrix = this.alignement_matrix;//get the pre-allocated buffer. Done in Constructor
		
		ArrayList<Double> candidate_bound_differences = new ArrayList<Double>(size(alignment_matrix));
		ArrayList<Double> o_1_bound_differences = new ArrayList<Double>(size(alignment_matrix));
		ArrayList<Double> o_k_bound_differences = new ArrayList<Double>(size(alignment_matrix));
		ArrayList<Double> o_k_square_bound_differences = new ArrayList<Double>(size(alignment_matrix));
		
		start = System.currentTimeMillis();
		final double[][] global_cost_matrix_book = fill_similarity_matrix();
		//For each pair of windows
		for(int line=0;line<alignment_matrix.length;line++) {							
			for(int column=0;column<alignment_matrix[0].length;column++) {	
				//Fill local matrix of the current window combination from global matrix 
				fill_local_similarity_matrix(local_similarity_matrix, global_cost_matrix_book, line, column);
				
				//Bound used for candidates
				double max_sim_value_in_matrix = max_value(local_similarity_matrix);
				
				if(max_sim_value_in_matrix+DOUBLE_PRECISION_BOUND>=threshold) {
					count_num_candidates++;
				}
				
				//O(1) bound
				double o_1_bound = prior_cell_similarity + MAX_SIM_ADDITION_NEW_NODE;// O(1) bound
				
				if(o_1_bound+DOUBLE_PRECISION_BOUND>=threshold) {
					count_survived_o_1_bound++;
				}
				
				//O(k) bound
				double o_k_bound = prior_cell_similarity - (prev_min_value / k_double);
				double max_sim_new_node = min(local_similarity_matrix);//(2) O(k) bound
				o_k_bound += max_sim_new_node / k_double;
				
				if(o_k_bound+DOUBLE_PRECISION_BOUND>=threshold) {
					count_survived_o_k_bound++;
				}
						
				//O(k²) bound
				double o_k_square_bound = get_sum_of_column_row_minima(local_similarity_matrix);
				if(o_k_square_bound+DOUBLE_PRECISION_BOUND>=threshold_times_k) {
					count_survived_pruning++;
				}
				o_k_square_bound /= k_double;
				
				//That's the important line
				double similarity = -solver.solve(local_similarity_matrix, threshold);
				similarity /= k_double;
				//normalize costs: Before it was distance. Now it is similarity.
				if(similarity>=threshold) {
					alignment_matrix[line][column] = similarity;//normalize
					count_computed_cells++;
				}//else keep it zero
				prior_cell_similarity = similarity;//normalize XXX only for ideal case
				prev_min_value = max(local_similarity_matrix);
				
				//System.out.println(line+" "+column+" "+similarity);
				
				if(similarity-DOUBLE_PRECISION_BOUND>=o_1_bound && column!=0) {
					System.err.println("similarity+DOUBLE_PRECISION_BOUND>=o_1_bound");
					System.err.println(prior_cell_similarity+"+"+MAX_SIM_ADDITION_NEW_NODE+"<"+similarity);
				}
				if(similarity-DOUBLE_PRECISION_BOUND>=o_k_bound && column!=0) {
					System.err.println("similarity+DOUBLE_PRECISION_BOUND>=o_k_bound");
					System.err.println(prior_cell_similarity+" "+(prev_min_value / k_double)+" "+max_sim_new_node / k_double+" < "+similarity);
					System.err.println("prior_cell_similarity+(prev_min_value / k_double)+max_sim_new_node / k_double < similarity");
				}
				if(similarity-DOUBLE_PRECISION_BOUND>=o_k_square_bound) {
					System.err.println("similarity+DOUBLE_PRECISION_BOUND>=o_k_square_bound");
				}
				if(column!=0) {
					double d_0_1 = o_1_bound - similarity;
					o_1_bound_differences.add(d_0_1);
					double d_0_k = o_k_bound - similarity;
					o_k_bound_differences.add(d_0_k);
				}
				double o_k_square_d = o_k_square_bound - similarity;
				o_k_square_bound_differences.add(o_k_square_d);
				candidate_bound_differences.add(max_sim_value_in_matrix - similarity);
			}
		}
		stop = System.currentTimeMillis();
		run_times[0] = (stop-start);		
		int size = size(alignment_matrix);
		double check_sum = sum(alignment_matrix);
		System.out.println("k="+k+"\t"+(stop-start)+"\tms\t"+check_sum+"\t"+size+"\t"+count_num_candidates+"\t"+count_survived_o_1_bound+"\t"+count_survived_o_k_bound+"\t"+count_survived_pruning+"\t"+count_computed_cells);
		
		System.out.println("Candidate bound mean overestimation\t"+mean(candidate_bound_differences));
		System.out.println("O(1) bound mean overestimation\t"+mean(o_1_bound_differences));
		System.out.println("O(k) bound mean overestimation\t"+mean(o_k_bound_differences));
		System.out.println("O(k²) bound mean overestimation\t"+mean(o_k_square_bound_differences));

		return run_times;
	}
	
	private double max_value(double[][] local_similarity_matrix) {
		double min_value = Double.MAX_VALUE;
		for(double[] array : local_similarity_matrix) {
			for(double d : array) {
				if(d < min_value) {
					min_value = d;
				}
			}
		}
		return -min_value;//XXX -trick for the Hungarian
	}

	private double mean(ArrayList<Double> list) {
		double sum = 0.0d;
		for(double d : list) {
			sum+=d;
		}
		return sum/(double)list.size();
	}

	public double[] run_baseline(){
		print_special_configurations();
		HungarianKevinStern solver = new HungarianKevinStern(k);
		System.out.println("Solutions.run_baseline() k="+k+" threshold="+threshold+" "+solver.get_name());
		MatrixRingBuffer mrb = new MatrixRingBuffer(k);
		USE_GLOBAL_MATRIX = true;
		
		double[] run_times = new double[num_paragraphs];
			
		double stop,start;
		long count_computed_cells   = 0;
		long count_survived_pruning = 0;
		
		//Allocate space for the alignment matrix
		final double[][] alignment_matrix = this.alignement_matrix;//get the pre-allocated buffer. Done in Constructor
		
		start = System.currentTimeMillis();
		//For each pair of windows
		for(int line=0;line<alignment_matrix.length;line++) {
			mrb.fill(line, 0, dense_global_matrix_buffer, raw_paragraph_b1, raw_paragraph_b2);
			for(int column=0;column<alignment_matrix[0].length;column++) {	
				if(column!=0) {
					mrb.update(line, column, dense_global_matrix_buffer, raw_paragraph_b1, raw_paragraph_b2);
				}
				
				final double upper_bound_sim = mrb.get_sum_of_column_row_minima();
				if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold_times_k) {
					count_survived_pruning++;
					//That's the important line
					double similarity = -solver.solve(mrb.buffer, threshold);
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
	
	/**
	 * Run FastText Sentence Algo
	 * @return
	 */
	public double[] run_fast_text(){
		print_special_configurations();
		System.out.println("Solutions.run_avg_word_2_vec() k="+k+" threshold="+threshold);
				
		double[] run_times = new double[num_paragraphs];
			
		double stop,start;
		long count_computed_cells = 0;
		
		//Allocate space for the alignment matrix
		final double[][] alignment_matrix = this.alignement_matrix;//get the pre-allocated buffer. Done in Constructor
		
		int vector_size = embedding_vector_index.entrySet().iterator().next().getValue().length;
		start = System.currentTimeMillis();
		
		final double[] avg_vec_window_line	= new double[vector_size];//300?
		final double[][] avg_vec_window_column= new double[k_with_windows_b2.length][vector_size];
		//final double[] avg_vec_window_column_copy= new double[vector_size];
		
		
		for(int column=0;column<alignment_matrix[0].length;column++) {
			get_avg_vector(k_with_windows_b2[column], avg_vec_window_column[column]);
		}
		
		//For each pair of windows
		for(int line=0;line<alignment_matrix.length;line++) {
			final int[] window_line = k_with_windows_b1[line];
			get_avg_vector(window_line, avg_vec_window_line);			
//			get_avg_vector(k_with_windows_b2[0], avg_vec_window_column);//init the column
			
			for(int column=0;column<alignment_matrix[0].length;column++) {	
				final int[] window_column = k_with_windows_b2[column];
				//get_avg_vector(window_column, avg_vec_window_column);
				/*if(column!=0) {//update the averaged vector
					//System.out.println(Util.outTSV(avg_vec_window_column));
					update_avg_vector(k_with_windows_b2[column-1], window_column, avg_vec_window_column);//TODO Check me
				}*/
				/*//TODO remove me
				for(int dim=0;dim<avg_vec_window_column_copy.length;dim++) {
					if(!is_equal(avg_vec_window_column_copy[dim],avg_vec_window_column[dim])) {
						System.err.println("dim="+dim);
						System.out.println(Util.outTSV(avg_vec_window_column));
						System.out.println(Util.outTSV(avg_vec_window_column_copy));
						System.out.println(Util.outTSV(k_with_windows_b2[column-1]));
						System.out.println(Util.outTSV(k_with_windows_b2[column]));
						System.out.println(k_with_windows_b2[column-1][0]+" "+Util.outTSV(embedding_vector_index.get(k_with_windows_b2[column-1][0])));
						System.out.println(k_with_windows_b2[column][0]+" "+Util.outTSV(embedding_vector_index.get(k_with_windows_b2[column-1][0])));
						System.out.println(k_with_windows_b2[column][1]+" "+Util.outTSV(embedding_vector_index.get(k_with_windows_b2[column][1])));
						System.out.println(k_with_windows_b2[column][2]+" "+Util.outTSV(embedding_vector_index.get(k_with_windows_b2[column][2])));
						System.err.println("dim="+dim);
					}
				}*/
				//final double similarity = cosine(avg_vec_window_line,avg_vec_window_column[column]);//TODO materialize norm of line window? squrared_nom vec b?
				final double similarity = cosine_similarity(avg_vec_window_line,avg_vec_window_column[column]);
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
	
	
	private static double cosine(final double[] vec_1, final double[] vec_2) {
		 double dotProduct = 0.0;
		 double normA = 0.0;
		 double normB = 0.0;
		 for (int i = 0; i < vec_1.length; i++) {
			 dotProduct += vec_1[i] * vec_2[i];
		     normA += Math.pow(vec_1[i], 2);
		     normB += Math.pow(vec_2[i], 2);
		 }   
		 return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
	}

	private void update_avg_vector(final int[] prior_window, final int[] current_window, final double[] avg_vec) {
		//(1) remove old vector of leaving token
		int leaving_token = prior_window[0];
		double[] my_vector = embedding_vector_index.get(leaving_token);
		if(my_vector!=null) {
			for(int i=0;i<avg_vec.length;i++) {
				avg_vec[i]-= my_vector[i];//Note the minus
			}
		}
		
		//(2) add vector of incoming token
		int incoming_token = current_window[current_window.length-1];
		my_vector = embedding_vector_index.get(incoming_token);
		if(my_vector!=null) {
			for(int i=0;i<avg_vec.length;i++) {
				avg_vec[i]+= my_vector[i];//Note the plus
			}
		}
		
	}

	private void get_avg_vector(int[] window, double[] avg_vec_window_line) {
		Arrays.fill(avg_vec_window_line, 0);
		for(int token : window){
			double[] my_vector = embedding_vector_index.get(token);
			if(my_vector!=null) {
				for(int i=0;i<avg_vec_window_line.length;i++) {
					avg_vec_window_line[i]+= my_vector[i];
				}
			}else {
				System.err.println("get_avg_vector() empty vector");
			}
		}

		
		if(MatchesWithEmbeddings.NORMALIZE_VECTORS) {//to unit length
			double length = 0;
			for(double v : avg_vec_window_line) {
				length += (v*v);
			}
			length = Math.sqrt(length);
			for(int i=0;i<avg_vec_window_line.length;i++) {
				avg_vec_window_line[i] = avg_vec_window_line[i]/length;
			}
		}
/*		//Normalize by window size
		double winodw_size = window.length;
		for(int i=0;i<avg_vec_window_line.length;i++) {
			avg_vec_window_line[i]/= winodw_size;
		}
*/
	}

	/**
	 * Does not compute every window, but jumps k cells. I.e., the cells are not overlapping
	 * @return
	 */
	@Deprecated
	public double[] run_naive_jump_k(){//XXX geht so nicht
		print_special_configurations();
		HungarianKevinStern solver = new HungarianKevinStern(k);
		System.out.println("Solutions.run_naive_jump_k() k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] local_similarity_matrix = new double[k][k];
		USE_GLOBAL_MATRIX = true;
		
		double[] run_times = new double[num_paragraphs];
			
		double stop,start;
		long count_computed_cells = 0;
		
		//Allocate space for the alignment matrix
		final double[][] alignment_matrix = this.alignement_matrix;//get the pre-allocated buffer. Done in Constructor
		
		start = System.currentTimeMillis();
		final double[][] global_cost_matrix_book = fill_similarity_matrix();
		//For each pair of windows
		for(int line=0;line<alignment_matrix.length;line++) {//XXX here to jump by k
			for(int column=0;column<alignment_matrix[0].length;column+=k) {//Note the +=k	
				//Fill local matrix of the current window combination from global matrix 
				fill_local_similarity_matrix(local_similarity_matrix, global_cost_matrix_book, line, column);

				//mrb.compare(local_similarity_matrix, column);
				
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
	
	public double[] run_naive(){
		print_special_configurations();
		HungarianKevinStern solver = new HungarianKevinStern(k);
		if(Config.verbose) System.out.println("Solutions.run_naive() k="+k+" threshold="+threshold+" "+solver.get_name());
		//final double[][] local_similarity_matrix = new double[k][k];
		USE_GLOBAL_MATRIX = false;
		
		double[] run_times = new double[num_paragraphs];
			
		double stop,start;
		long count_computed_cells = 0;
		
		//Allocate space for the alignment matrix
		final double[][] alignment_matrix = this.alignement_matrix;//get the pre-allocated buffer. Done in Constructor
		final MatrixRingBuffer mrb = new MatrixRingBuffer(k);
		
		start = System.currentTimeMillis();
		//final double[][] global_cost_matrix_book = fill_similarity_matrix();
		//For each pair of windows
		for(int line=0;line<alignment_matrix.length;line++) {
			mrb.fill(line, 0, dense_global_matrix_buffer, raw_paragraph_b1, raw_paragraph_b2);
			for(int column=0;column<alignment_matrix[0].length;column++) {	
				//Fill local matrix of the current window combination from global matrix 
				//fill_local_similarity_matrix(local_similarity_matrix, global_cost_matrix_book, line, column);
				if(column!=0) {
					mrb.update(line, column, dense_global_matrix_buffer, raw_paragraph_b1, raw_paragraph_b2);
				}
				//mrb.compare(local_similarity_matrix, column);
				
				//That's the important line
				final double similarity = -solver.solve(mrb.buffer, threshold);
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
		if(Config.verbose) System.out.println("k="+k+"\t"+(stop-start)+"\tms\t"+check_sum+"\t"+size+"\t"+count_computed_cells);

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
	
	MyArrayList condense(final BitSet candidates_line) {
		MyArrayList candidates_condensed = new MyArrayList(candidates_line.size());
		//int q = 0;
		//boolean found_run = false;
		int start_alt=0, stop_alt=0;
		
		while((start_alt = candidates_line.nextSetBit(start_alt))!=-1) {
			stop_alt = candidates_line.nextClearBit(start_alt);
			candidates_condensed.add(start_alt);
			candidates_condensed.add(stop_alt-1);
			start_alt = stop_alt;
		}
		
		/*while(q<candidates_line.length()) {
			start_alt = candidates_line.nextSetBit(q);
			stop_alt = candidates_line.nextClearBit(start_alt);
			
			if(candidates_line.get(q)) {//start of a run
				if(q!=start_alt) {
					System.err.println("q!=start_alt");
				}
				candidates_condensed.add(q);
				q++;
				found_run = true;
				while(q<candidates_line.size()) {
					if(!candidates_line.get(q)){//end of run
						if(q!=stop_alt) {
							System.err.println("q!=start_alt");
						}
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
			System.err.println("At end stop ="+stop_alt);
			candidates_condensed.add(candidates_line.size()-1);
		}*/
		return candidates_condensed;
	}
	
	MyArrayList condense(final boolean[] candidates_line) {
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
		return candidates_condensed;
	}
	
	/**
	 * Suggests that candidate generation still makes sense
	 * @return
	 */
	public double[] run_solution_no_candidates() {
		print_special_configurations();
		HungarianDeep solver = new HungarianDeep(k);
		System.out.println("Solutions.run_solution_no_candidates() k="+k+" threshold="+threshold+" "+solver.get_name());
		USE_GLOBAL_MATRIX = true;
		
		//Some variable
		double[] run_times = new double[num_paragraphs];
		double stop,start;
		final double[][] current_lines = new double[k][];
		USE_GLOBAL_MATRIX = true;
//		final ArrayList<MyArrayList> all_candidates = new ArrayList<MyArrayList>(alignement_matrix.length);
//		final BitSet candidates = new BitSet(alignement_matrix[0].length);
		
//		final BitSet[] inverted_window_index = create_indexes_bit_vectors();//TODO include into run time below?
		final double[][] global_matrix_book  = fill_similarity_matrix_deep();//can be re-used for any k. Thus not part of the runtime. TODO Buffer to make it fair
		
		start = System.currentTimeMillis();

		double stop_candidates = System.currentTimeMillis();
		
		for(int line=0;line<alignement_matrix.length;line++) {
			for(int i=0;i<k;i++) {//Let current_lines point to the right window
				current_lines[i] = global_matrix_book[line+i];
			}
			solver.set_matrix(current_lines);//This way, we avoid to pass the window again and again for each window in a line of the matrix
			//Validate candidates
			final double[] alignment_matrix_line   = alignement_matrix[line];
			int run_start = 0;
			int run_stop = alignment_matrix_line.length-1;//to the end inclusively
			validate_run_deep_2(solver, line, run_start, run_stop, current_lines, alignment_matrix_line);
		}
		
		stop = System.currentTimeMillis();
		run_times[0] = stop-start;
		
		int size = size(alignement_matrix);
		double check_sum = sum(alignement_matrix);
		System.out.println("k="+k+"\t"+(stop-start)+"\tms\t"+check_sum+"\t"+size+"\t"+count_candidates+"\t"+count_survived_pruning+"\t"+count_survived_second_pruning+"\t"+count_survived_third_pruning+"\t"+count_cells_exceeding_threshold+"\t"+(stop_candidates-start));
		
		return run_times;
	}
	
	int all_done = 0;
	MyArrayList merge_runs(MyArrayList[] candidates) {
		MyArrayList condensed_candidates = new MyArrayList(200);//guess
		int[] pointer_lists = new int[candidates.length];
		all_done = 0;
		
		int min_start_list = min_start_list(pointer_lists, candidates);
		int pointer = pointer_lists[min_start_list];
		int start = candidates[min_start_list].get(pointer);
		int stop = candidates[min_start_list].get(pointer+1);
		condensed_candidates.add(start);
		condensed_candidates.add(stop);
		
		while(all_done==candidates.length) {
			min_start_list = min_start_list(pointer_lists, candidates);
			pointer = pointer_lists[min_start_list];
			start = candidates[min_start_list].get(pointer);
			stop = candidates[min_start_list].get(pointer+1);	
			
			int size = condensed_candidates.size();
			int last_stop  = condensed_candidates.get(size-1);
			if(start<=last_stop+1) {//Overlap in intervals
				condensed_candidates.ARRAY[size-1] = stop;//Overwrite stop position
			}else {
				condensed_candidates.add(start);
				condensed_candidates.add(stop);	
			}
		}
		return condensed_candidates;
	}
	
	int min_start_list(int[] pointer_lists, MyArrayList[] candidates) {
		int min_pointer = Integer.MAX_VALUE;
		int idx_min_pointer = -1;
		for(int list=0;list<pointer_lists.length;list++) {
			int my_start = candidates[list].get(pointer_lists[list]); 
			if(my_start<min_pointer) {
				min_pointer = my_start;
				idx_min_pointer = list;
			}
		}
		pointer_lists[idx_min_pointer] +=2;//increment position
		if(candidates[idx_min_pointer].size() <= pointer_lists[idx_min_pointer]) {//reached end of this candidate list
			all_done++;
		}
		return idx_min_pointer;
	}

	public static ArrayList<String[]> memory_consumptions = null;
	public double[] run_memory_consumption_measurement() {
		print_special_configurations();
		HungarianDeep solver = new HungarianDeep(k);
		System.out.println("Solutions.run_memory_consumption_measurement() k="+k+" threshold="+threshold+" "+solver.get_name());
		USE_GLOBAL_MATRIX = true;
		if(memory_consumptions==null) {
			memory_consumptions = new ArrayList<String[]>();
		}
		
		//Some variable
		double[] run_times = new double[num_paragraphs];
		double stop,start;
		final double[][] current_lines = new double[k][];
		USE_GLOBAL_MATRIX = true;
		final ArrayList<MyArrayList> all_candidates = new ArrayList<MyArrayList>(alignement_matrix.length);
		final BitSet candidates = new BitSet(alignement_matrix[0].length);
		
		final double[][] global_matrix_book  = fill_similarity_matrix_deep();//can be re-used for any k. Thus not part of the runtime. TODO Buffer to make it fair
		
		
		final MyArrayList[] neighborhood_index = create_neihborhood_index(dense_global_matrix_buffer);
		int size_neighborhood_index = 0;
		for(MyArrayList list : neighborhood_index) {
			if(list!=null) {
				size_neighborhood_index+=list.size();
			}
		}
		
		start = System.currentTimeMillis();
		final BitSet[] inverted_window_index = create_indexes_bit_vectors();//XXX includes neighborhood computation, which can be re-used
		double stop_idx_creation = System.currentTimeMillis();
		
		int size_bit_vector = 0;
		for(BitSet bs : inverted_window_index) {
			if(bs != null) {//not dense, only for all tokens of b1
				size_bit_vector += bs.size();
			}
		}
		
		for(int line=0;line<alignement_matrix.length;line++) {
			candidates.clear();//may contain result from prior line
			for(int token_id : k_with_windows_b1[line]) {
				final BitSet temp = inverted_window_index[token_id];
				candidates.or(temp);
			}
			//condense bool[] to runs with from to
			MyArrayList candidates_condensed = condense(candidates);
			all_candidates.add(candidates_condensed);
		}
		double stop_candidates = System.currentTimeMillis();
		
		int size_all_candidates = 0;
		for(MyArrayList cand : all_candidates) {
			size_all_candidates += cand.size();
		}
		
		int size_global_matrix = size(global_matrix_book); 

		
		stop = System.currentTimeMillis();
		run_times[0] = stop-start;
		
		int size = size(alignement_matrix);
		
		int size_double = 8;
		int size_int = 4;
		System.out.println("theta="+threshold+"\t"+(stop-start)+"\tms\t"+size*size_double+"\t"+size_global_matrix*size_double+"\t"+size_bit_vector/8+"\t"+size_neighborhood_index*size_int+"\t"+size_all_candidates);
		String[] temp = {
			threshold+""
			,size*size_double+""
			,size_global_matrix*size_double+""
			,size_bit_vector/8+""
			,size_neighborhood_index*size_int+""
			,size_all_candidates+""
		};
		memory_consumptions.add(temp);
		//clean up
		count_candidates = 0;
		count_survived_pruning = 0;
		count_survived_second_pruning = 0;
		count_survived_third_pruning = 0;
		count_cells_exceeding_threshold = 0;
		return run_times;
	}
	
	public double[] run_solution() {
		print_special_configurations();
		HungarianDeep2 solver = new HungarianDeep2(k);
		solver.set_matrix(mrb.buffer);
		System.out.println("Solutions.run_solution() k="+k+" threshold="+threshold+" "+solver.get_name());
		USE_GLOBAL_MATRIX = true;
		
		//Some variable
		double[] run_times = new double[num_paragraphs];
		double stop,start;
		
		USE_GLOBAL_MATRIX = true;
		final ArrayList<MyArrayList> all_candidates = new ArrayList<MyArrayList>(alignement_matrix.length);
		final BitSet candidates = new BitSet(alignement_matrix[0].length);
		
		start = System.currentTimeMillis();
		final BitSet[] inverted_window_index = create_indexes_bit_vectors();//XXX includes neighborhood computation, which can be re-used
		double stop_idx_creation = System.currentTimeMillis();
		
		for(int line=0;line<alignement_matrix.length;line++) {
			candidates.clear();//may contain result from prior line
			candidates.or(inverted_window_index, k_with_windows_b1[line]);
			MyArrayList candidates_condensed = condense(candidates);
			
			all_candidates.add(candidates_condensed);
		}
		double stop_candidates = System.currentTimeMillis();
		
		for(int line=0;line<alignement_matrix.length;line++) {
			//Validate candidates
			final double[] alignment_matrix_line   = alignement_matrix[line];
			final MyArrayList candidates_condensed = all_candidates.get(line);
			
			final int size = candidates_condensed.size();
			final int[] raw_candidates = candidates_condensed.ARRAY;
			for(int c=0;c<size;c+=2) {//Contains start and stop index. Thus, c+=2.
				final int run_start = raw_candidates[c];
				final int run_stop = raw_candidates[c+1];
				
				validate_run_deep(solver, line, run_start, run_stop, alignment_matrix_line);
			}
		}
		
		stop = System.currentTimeMillis();
		run_times[0] = stop-start;
		
		int size = size(alignement_matrix);
		double check_sum = sum(alignement_matrix);
		System.out.println("k="+k+"\t"+(stop-start)+"\tms\tcheck_sum=\t"+check_sum+"\t"+size+"\tcandidates\t"+count_candidates+"\tO(1)\t"+count_survived_pruning+"\t"+count_survived_second_pruning+"\t"+count_survived_third_pruning+"\t"+count_cells_exceeding_threshold+"\t"+(stop_candidates-start)+"\t"+(stop_idx_creation-start));
		//clean up
		/*count_candidates = 0;
		count_survived_pruning = 0;
		count_survived_second_pruning = 0;
		count_survived_third_pruning = 0;
		count_cells_exceeding_threshold = 0;*/
		return run_times;
	}
	
	MyArrayList merge(BitSet[] my_candidates) {//, MyArrayList candidates_correct) {
		final int size = my_candidates[0].size();
		final int max_stop = k_with_windows_b2.length-1;
		int index = 0;
		boolean found_run = false;
		MyArrayList candidates_condensed = new MyArrayList(size/2);
		
		while(index<size) {
			//find next run start
			if(has_run(my_candidates,index)) {
				int start_run = index;
				found_run = true;
				if(index>k) {
					index +=k;//TODO hier vlt k ausnutzen	
				}else {
					index++;
				}
				
				while(has_run(my_candidates,index)) {
					index++;
				}
				int stop_run = index-1;//index is the first one not having a candidate
				if(stop_run>=max_stop) {
					stop_run = max_stop;
				}
				candidates_condensed.add(start_run);
				candidates_condensed.add(stop_run);
				found_run = false;
			}else{
				index++;
			}
		}
		if(found_run) {
			candidates_condensed.add(size-1);	
		}

		/*if(candidates_correct.size()!=candidates_condensed.size()) {
			System.err.println(candidates_correct);
			System.err.println(candidates_condensed);
		}else{
			for(int i=0;i<candidates_correct.size();i++) {
				if(candidates_correct.get(i)!=candidates_condensed.get(i)) {
					System.err.println("candidates_correct.get(i)!=candidates_condensed.get("+i+")");
					System.err.println(candidates_correct);
					System.err.println(candidates_condensed);
					System.err.println("i");
				}
			}
		}*/
		return candidates_condensed;
	}

	private boolean has_run(BitSet[] my_candidates, int index) {
		for(BitSet bs : my_candidates) {
			if(bs.get(index)) {
				return true;
			}
		}
		return false;
	}

	public double[] run_candidates_deep() {
		print_special_configurations();
		HungarianDeep solver = new HungarianDeep(k);
		System.out.println("Solutions.run_candidates_deep() k="+k+" threshold="+threshold+" "+solver.get_name());
		USE_GLOBAL_MATRIX = true;
		
		//Some variable
		double[] run_times = new double[num_paragraphs];
		double stop,start;
		//Allocate space for the alignment matrix
		final boolean[][] candidates   = new boolean[alignement_matrix.length][alignement_matrix[0].length];
		final double[][] current_lines = new double[k][];
		USE_GLOBAL_MATRIX = true;
		final ArrayList<MyArrayList> all_candidates = new ArrayList<MyArrayList>(candidates.length);
		
		final int[][] inverted_window_index_ranges = create_indexes();//TODO include into run time below?
		final double[][] global_matrix_book        = fill_similarity_matrix_deep();//can be re-used for any k. Thus not part of the runtime. TODO Buffer to make it fair
		
		start = System.currentTimeMillis();
		for(int line=0;line<alignement_matrix.length;line++) {
			final boolean[] candidates_line = candidates[line];
			
			//Fill all_candidates
			get_candidates(k_with_windows_b1[line], candidates_line, inverted_window_index_ranges);
			//condense bool[] to runs with from to
			MyArrayList candidates_condensed = condense(candidates_line);
			all_candidates.add(candidates_condensed);
		}
		double stop_candidates = System.currentTimeMillis();
		
		for(int line=0;line<alignement_matrix.length;line++) {
			for(int i=0;i<k;i++) {
				current_lines[i] = global_matrix_book[line+i];
			}
			solver.set_matrix(current_lines);
			//Validate candidates
			final double[] alignment_matrix_line = alignement_matrix[line];
			final MyArrayList candidates_condensed = all_candidates.get(line);
			
			final int size = candidates_condensed.size();
			final int[] raw_candidates = candidates_condensed.ARRAY;
			for(int c=0;c<size;c+=2) {
				final int run_start = raw_candidates[c];
				final int run_stop = raw_candidates[c+1];
				
				validate_run_deep_2(solver, line, run_start, run_stop, current_lines, alignment_matrix_line);
			}
		}
		
		stop = System.currentTimeMillis();
		run_times[0] = stop-start;
		
		int size = size(alignement_matrix);
		double check_sum = sum(alignement_matrix);
		System.out.println("k="+k+"\t"+(stop-start)+"\tms\t"+check_sum+"\t"+size+"\t"+count_candidates+"\t"+count_survived_pruning+"\t"+count_survived_second_pruning+"\t"+count_survived_third_pruning+"\t"+count_cells_exceeding_threshold+"\t"+(stop_candidates-start));
		
		return run_times;
	}
	
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
		final double[][] alignment_matrix = this.alignement_matrix;//get the pre-allocated buffer. Done in Constructor
		final boolean[][] candidates = new boolean[alignment_matrix.length][alignment_matrix[0].length];
		
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
	private MyArrayList[] create_neihborhood_index(final double[][] matrix) {
		//System.out.println("create_neihborhood_index() BEGIN");
		//double start = System.currentTimeMillis();
		
		MyArrayList[] indexes = new MyArrayList[matrix.length];
		for(int token_id : this.tokens_b1) {
			final double[] line = matrix[token_id];
			MyArrayList index = new MyArrayList(line.length);//Ensure the list has enough memory reserved
			for(int id : tokens_b2) {
				final double sim = line[id];
				if(sim>=threshold){
					index.add(id);
				}
			}
			//TODO trim to size?
			indexes[token_id]=index;
		}
		//System.out.println("create_neihborhood_index() END in\t"+(System.currentTimeMillis()-start));
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
	
	/**
	 * indexes.get(token_id) -> int[] of window (positions) with token having sim(token_id, some token) > threshold
	 * @param k_with_windows
	 * @return
	 */
	private BitSet[] create_inverted_window_index_bit_vector(final int[][] k_with_windows, final MyArrayList[] neihborhood_indexes) {
		System.out.println("BitSet[] create_inverted_window_index() BEGIN");
		final double start = System.currentTimeMillis();
		
		final BitSet[] indexes = new BitSet[dense_global_matrix_buffer.length];//one index per token
		//For each token
		for(int token_id = 0;token_id<neihborhood_indexes.length;token_id++) {
			//Create the list of occurrences for token: token_id
			final MyArrayList neihborhood_index = neihborhood_indexes[token_id];
			if(neihborhood_index==null) {
				continue;//Token not in b_1
			}
			
			final BitSet index_this_token = new BitSet(k_with_windows.length);
			for(int pos=0;pos<k_with_windows.length;pos++) {
				int[] curr_window = k_with_windows[pos];
				if(is_in(neihborhood_index, curr_window)) {
					index_this_token.set(pos);
				}
			}
			indexes[token_id] = index_this_token;
		}
		System.out.println("BitSet[] create_inverted_window_index() END in\t"+(System.currentTimeMillis()-start));
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
		final MyArrayList[] neighborhood_index = create_neihborhood_index(dense_global_matrix_buffer);
		//System.err.println("sum(neighborhood_index) "+sum(neighborhood_index));
		/**
		 * inverted_window_index.get(my_token_id) -> ordered list of cells containing some other token, s.t.  sim(my_token_id, token_id) >= threshold. I.e., this is a candidate. 
		 */
		System.err.println("Removed");
		//final MyArrayList[] inverted_window_index = create_inverted_window_index(k_with_windows_b2, neighborhood_index);
		//System.err.println("sum(inverted_window_index) "+sum(inverted_window_index));
		/**
		 * inverted_window_index_ranges[token_id][0] -> Start of first run (if there is one). The respective end is at inverted_window_index_ranges[token_id][1] etc.
		 */
		System.err.println("Removed");
		//final int[][] inverted_window_index_ranges = to_inverted_window_index_ranges(inverted_window_index);
		//System.err.println("sum(inverted_window_index_ranges) "+sum(inverted_window_index_ranges));
		
		System.err.println("Removed");
		return null;
		//return inverted_window_index_ranges;
	}
	
	public BitSet[] create_indexes_bit_vectors() {
		/**
		 * inverted_index.get(my_token_id) -> ordered list of token_id's with sim(my_token_id, token_id) >= threshold 
		 */
		final MyArrayList[] neighborhood_index = create_neihborhood_index(dense_global_matrix_buffer);

		/**
		 * inverted_window_index.get(my_token_id) -> ordered list of cells containing some other token, s.t.  sim(my_token_id, token_id) >= threshold. I.e., this is a candidate. 
		 */
		//final BitSet[] inverted_window_index = create_inverted_window_index_bit_vector(k_with_windows_b2, neighborhood_index);
		final BitSet[] inverted_window_index = create_inverted_window_index_bit_vector_2(neighborhood_index);

		//test(neighborhood_index, inverted_window_index);
		
		return inverted_window_index;
	}
	
	BitSet[] create_inverted_window_index_bit_vector_2(final MyArrayList[] neihborhood_indexes) {
		//System.out.println("BitSet[] create_inverted_window_index_bit_vector_2() BEGIN");
		//final double start_time = System.currentTimeMillis();
		final BitSet[] indexes = new BitSet[dense_global_matrix_buffer.length];//one index per token
		
		for(int token_id : tokens_b1) {
			final BitSet index_this_token = new BitSet(k_with_windows_b2.length);
			final MyArrayList my_neighborhood_index = neihborhood_indexes[token_id];
			
			for(int i=0;i<raw_paragraph_b2.length;i++) {
				final int token_id_in_b2 = raw_paragraph_b2[i];
				if(my_neighborhood_index.isIn(token_id_in_b2)) {
					final int start = Math.max(0, i-k+1);
					final int stop = Math.min(k_with_windows_b2.length-1, i);
					index_this_token.set(start,stop+1);
					/*for(int pos=start;pos<=stop;pos++) {
						index_this_token.set(pos);
					}*/					
				}
			}
			indexes[token_id] = index_this_token;
		}
		//final double stop_time = System.currentTimeMillis();
		//System.out.println("BitSet[] create_inverted_window_index_bit_vector_2() END "+(stop_time-start_time)+" ms");
		return indexes;
	}

	MyArrayList[] test(final MyArrayList[] neighborhood_index, BitSet[] indexes_corect) {
		System.out.println("MyArrayList[] test() BEGIN");
		final double start_time = System.currentTimeMillis();
		
		final MyArrayList[] candidate_runs = new MyArrayList[dense_global_matrix_buffer.length];
		
		for(int token_id : tokens_b1) {
			if(token_id==30) {
				System.err.println("token_id=="+token_id);
			}
			final MyArrayList my_neighborhood_index = neighborhood_index[token_id];
			
			//BitSet idx_correct = indexes_corect[token_id];
			MyArrayList condensed_idx_correct = condense(indexes_corect[token_id]);
			MyArrayList condensed_idx = new MyArrayList();
			for(int i=0;i<raw_paragraph_b2.length;i++) {
				final int token_id_in_b2 = raw_paragraph_b2[i];
				if(my_neighborhood_index.isIn(token_id_in_b2)) {
					int start = Math.max(0, i-k+1);
					int stop = Math.min(k_with_windows_b2.length-1, i);
					
					if(condensed_idx.size()>=2) {
						int size = condensed_idx.size();
						int last_stop  = condensed_idx.get(size-1);
						if(start<=last_stop+1) {//Overlap in intervals
							condensed_idx.ARRAY[size-1] = stop;//Overwrite stop position
						}else {
							condensed_idx.add(start);
							condensed_idx.add(stop);	
						}
					}else {
						condensed_idx.add(start);
						condensed_idx.add(stop);
					}					
				}
			}
			
			if(condensed_idx_correct.size()!=condensed_idx.size()) {
				System.err.println(condensed_idx_correct);
				System.err.println(condensed_idx);
			}else{
				for(int i=0;i<condensed_idx_correct.size();i++) {
					if(condensed_idx.get(i)!=condensed_idx.get(i)) {
						System.err.println("condensed_idx.get(i)!=condensed_idx.get(i)");
					}
				}
			}
			candidate_runs[token_id] = condensed_idx;
		}
		
		final double stop_time = System.currentTimeMillis();
		System.out.println("BitSet[] test() END "+(stop_time-start_time)+" ms");

		return candidate_runs;
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
	public void validate_run_deep(HungarianDeep2 hunga, final int line, final int run_start, final int run_stop
			, final double[] alignment_matrix_line) {
		
		double ub_sum, sim, prior_cell_similarity, prev_min_value; 
		
		if(LOGGING_MODE) {count_candidates+=run_stop-run_start+1;}
		
		int column=run_start;			
		{//Here we have no bound O(1) bound
			if(LOGGING_MODE) count_survived_pruning++;
			if(LOGGING_MODE) count_survived_second_pruning++;
			
			mrb.fill(line, column, dense_global_matrix_buffer, raw_paragraph_b1, raw_paragraph_b2);
			ub_sum = mrb.get_sum_of_column_row_minima()/k_double;
			
			if(ub_sum+DOUBLE_PRECISION_BOUND>=threshold) {
				if(LOGGING_MODE) count_survived_third_pruning++;
				sim = -hunga.solve(mrb.col_maxima);
				sim /= k_double;
				if(sim>=threshold) {
					if(LOGGING_MODE) count_cells_exceeding_threshold++;
					alignment_matrix_line[column] = sim;
				}//else keep it zero
				prior_cell_similarity = sim;
			}else{
				prior_cell_similarity = ub_sum;
			}
			prev_min_value = mrb.max(column);
		}
	
		//For all other columns: Here we have a O(1) and O(k) bound
		for(column=run_start+1;column<=run_stop;column++) {
			mrb.update_with_bound(line, column, dense_global_matrix_buffer, raw_paragraph_b1, raw_paragraph_b2);
			double upper_bound_sim = prior_cell_similarity + MAX_SIM_ADDITION_NEW_NODE;// O(1) bound
			upper_bound_sim-= (prev_min_value / k_double);// (1) O(k) bound : part of the O(k) bound in case the prior cell updated the matrix, i.e., we know the minimum similarity of the leaving node
			
			if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
				if(LOGGING_MODE) count_survived_pruning++;  
				
				//(2) O(k) bound 
				double max_sim_new_node = mrb.min(column);
				
				upper_bound_sim-=MAX_SIM_ADDITION_NEW_NODE;
				upper_bound_sim+=(max_sim_new_node/k_double);
									
				double temp = -mrb.col_sum / k_double;
				
				if(temp<upper_bound_sim) {
					upper_bound_sim = temp;
				}
				 
				if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {						
					if(LOGGING_MODE) count_survived_third_pruning++;
					//That's the important line
					sim = -hunga.solve(mrb.col_maxima);
					//normalize 
					sim /= k_double;
					if(sim>=threshold) {
						if(LOGGING_MODE) count_cells_exceeding_threshold++;
						alignment_matrix_line[column] = sim;
					}//else keep it zero
					prior_cell_similarity = sim;
				}
			}
			prev_min_value = mrb.max(column);
			prior_cell_similarity = upper_bound_sim;
		}
	}
	
	public void validate_run_deep_2(HungarianDeep solver, final int line, final int run_start, final int run_stop
			, final double[][] current_lines, final double[] alignment_matrix_line) {
		
		double ub_sum, sim, prior_cell_similarity, prev_min_value; 
		boolean prior_cell_updated_matrix, column_sum_correct;
		
		if(LOGGING_MODE) {count_candidates+=run_stop-run_start+1;}
		
		int column=run_start;			
		{//Here we have no bound O(1) bound
			if(LOGGING_MODE) count_survived_pruning++;
			if(LOGGING_MODE) count_survived_second_pruning++;
			ub_sum = o_k_square_bound(current_lines, column)/k_double;
			
			if(ub_sum+DOUBLE_PRECISION_BOUND>=threshold) {
				if(LOGGING_MODE) count_survived_third_pruning++;
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
			column_sum_correct = true;
		}
	
		//For all other columns: Here we have a O(1) and O(k) bound
		for(column=run_start+1;column<=run_stop;column++) {	
			double upper_bound_sim = prior_cell_similarity + MAX_SIM_ADDITION_NEW_NODE;// O(1) bound
			if(prior_cell_updated_matrix) {
				upper_bound_sim-= (prev_min_value / k_double);// (1) O(k) bound : part of the O(k) bound in case the prior cell updated the matrix, i.e., we know the minimum similarity of the leaving node
			}		
			
			if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
				if(LOGGING_MODE) count_survived_pruning++;  
				
				double max_sim_new_node = min(current_lines, column);//(2) O(k) bound 
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
					if(LOGGING_MODE) count_survived_second_pruning++;
					ub_sum = o_k_square_bound(current_lines, column)/k_double;
					upper_bound_sim = (ub_sum<upper_bound_sim) ? ub_sum : upper_bound_sim;//The sum bound is not necessarily tighter. Need the tightest bound for next cell.
					
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
					column_sum_correct = true;
				}else{
					prior_cell_similarity = upper_bound_sim;
					column_sum_correct = false;
				}
				prev_min_value = max(current_lines, column);
				prior_cell_updated_matrix = true;
			}else{
				prior_cell_similarity = upper_bound_sim;
				prior_cell_updated_matrix = false;
				column_sum_correct = false;
			}
		}
	}
	
	
	private boolean is_equal(double[] col_maxima2, double[] temp_arr) {
		for(int i=0;i<col_maxima2.length;i++) {
			if(col_maxima2[i] != temp_arr[i]) return false;
		}
		return true;
	}

	private double sum_bound_row_only_and_shift(final double[][] current_lines, final int offset, final double max_sim_new_node) {
		double row_sum = 0;
		System.arraycopy(col_maxima, 1, col_maxima, 0, k-1);
		col_maxima[k-1] = max_sim_new_node;
		
		for(int i=0;i<this.k;i++) {
			final double[] line = current_lines[i];
			double row_min = Double.POSITIVE_INFINITY;
			for(int j=0;j<this.k;j++) {
				final double val = line[offset+j];
				if(val<row_min) {
					row_min = val;
				}
			}
			row_sum += row_min;
		}
		double min_cost = Math.max(row_sum, sum_cols);		
		
		return -min_cost;
	}

	private double min_col_i(double[][] current_lines, int column, int i) {
		double min = Double.MAX_VALUE;
		for(int j=0;j<k;j++) {
			double val = current_lines[j][column+i];
			if(val<min) {
				min=val;
			}
		}
		return min;
	}

	private void out_matrix(double[][] current_lines, int column) {
		for(double[] line : current_lines) {
			for(int i=0;i<column+k;i++) {
				System.out.print(line[i]+"\t");
			}
			System.out.println();
		}
		System.out.println("col/row maxime");
		System.out.println(Arrays.toString(col_maxima));
		//System.out.println(Arrays.toString(row_maxima));
		
	}

	public double[] run_incremental_cell_pruning_deep(){
		HungarianDeep solver = new HungarianDeep(k);
		System.out.println("Solutions.run_incremental_cell_pruning_deep() k="+k+" threshold="+threshold+" "+solver.get_name());
		
		double[] run_times = new double[num_paragraphs];
	
		double stop,start;

		//Allocate space for the alignment matrix
		final double[][] alignment_matrix = alignement_matrix;
		
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
				ub_sum = o_k_square_bound(current_lines, column)/k_double;
				
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
						
						ub_sum = o_k_square_bound(current_lines, column)/k_double;
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
		
		System.out.println("Solutions.run_incremental_cell_pruning() k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] local_similarity_matrix = new double[k][k];
		
		double[] run_times = new double[num_paragraphs];
		
		double stop,start;

		//Allocate space for the alignment matrix
		final double[][] alignment_matrix = alignement_matrix;
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

	
	final double sim_cached(final int token_id_1, final int token_id_2) {
		return (token_id_1==token_id_2) ? EQUAL : dense_global_matrix_buffer[token_id_1][token_id_2]; 
	}
	
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
		sum_cols = sum(col_maxima);
		double min_cost = Math.max(row_sum, sum_cols);		
		
		return -min_cost;
	}
	
	
	/**
	 * O(k²) bound computing min(row/column)
	 * @param current_lines
	 * @param offset
	 * @return
	 */
	private double o_k_square_bound(final double[][] current_lines, final int offset) {
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
		sum_cols = sum(col_maxima);
		double min_cost = Math.max(row_sum, sum_cols);		
		
		return -min_cost;
	}	
	
	private double min(double[] array, int offset) {
		double min = array[offset];
		for(int i=1;i<k;i++) {
			double val = array[offset+i];
			if(val<min) {
				min=val;
			}
		}
		return min;
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
	
	public static void out(int[] arr) {
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
				local_cost_matrix[i][j] = -global_cost_matrix_book[line+i][column+j];//Note the minus for the Hungarian
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
	public static double sum(double[][] alignment_matrix) {
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
	
	public double[][] fill_similarity_matrix_deep() {
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
	
	public static double sim(final int set_id1, final int set_id2, final double[] vec_1, final double[] vec_2) {
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
	public static double cosine_similarity(final double[] vectorA, final double[] vectorB) {
		double dotProduct = 0.0;
	    for (int i = 0; i < vectorA.length; i++) {
	        dotProduct += vectorA[i] * vectorB[i];
	    }
	    
	    dotProduct = (dotProduct < 0) ? 0 : dotProduct;
	    dotProduct = (dotProduct > 1) ? 1 : dotProduct;
	    return dotProduct;
	}
	
	public double[][] jaccard_windows(){
		if(!Config.STEM_WORDS) {//In case we do not stem, replace all sets having maximal similarity s.t. it counts for jaccard as overlap
			HashSet<Integer> duplicates = new HashSet<Integer>();
			final double[][] global_cost_matrix_book = fill_similarity_matrix();
			for(int id=0;id<global_cost_matrix_book.length;id++) {
				if(duplicates.contains(id)) {
					continue;
				}
				final double[] line = global_cost_matrix_book[id];
				for(int other_id=id+1;other_id<line.length;other_id++) {
					if(line[other_id]==1.0d) {
						duplicates.add(other_id);
						replace(k_with_windows_b1, id, other_id);
						replace(k_with_windows_b2, id, other_id);
					}
				}
			}
		}
		
		double[][] matrix = new double[k_with_windows_b1.length][k_with_windows_b2.length];
		for(int row=0;row<matrix.length;row++) {
			int[] w_r = k_with_windows_b1[row];
			for(int colum=0;colum<matrix[0].length;colum++) {
				int[] w_c = k_with_windows_b2[colum];
				double jaccard_sim = jaccard(w_r, w_c);
				matrix[row][colum] = jaccard_sim;
			}
		}
		return matrix;
	}
	
	private void replace(int[][] windows, int id, int replace_me_id) {
		for(int[] w : windows) {
			for(int i=0;i<w.length;i++) {
				if(w[i]==replace_me_id){
					w[i] = id; 
				}
			}
		}
		
	}

	static double jaccard(int[] tokens_t1, int[] tokens_t2) {
		HashSet<Integer> tokens_hashed = new HashSet<Integer>(tokens_t1.length);
		for(int t : tokens_t1) {
			tokens_hashed.add(t);
		}
		
		int size_intersection = 0;
		for(int t : tokens_t2) {
			if(tokens_hashed.contains(t)) {
				size_intersection++;
			}
		}
		//size union
		for(int t : tokens_t2) {
			tokens_hashed.add(t);
		}
		int size_union = tokens_hashed.size();
		
		double jaccard_sim = (double) size_intersection / (double) size_union;
		
		/*double jaccard_sim_c = (double) size_intersection / (double)(tokens_t1.length+tokens_t2.length-size_intersection);
		
		if(jaccard_sim!=jaccard_sim_c) {
			System.err.println("jaccard_sim!=jaccard_sim_c");
		}*/
		
		if(jaccard_sim>=0.8) {
			System.out.println(Arrays.toString(tokens_t1));
			System.out.println(Arrays.toString(tokens_t2));
		}
		
		return jaccard_sim;
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

	public double[] run_dummy() {
		return col_maxima;
	}

	public double size_alignment_matrix() {
		return size(alignement_matrix);
	}
}
