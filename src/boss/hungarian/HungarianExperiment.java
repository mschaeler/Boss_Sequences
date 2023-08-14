package boss.hungarian;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map.Entry;

import boss.embedding.MatchesWithEmbeddings;
import boss.test.SemanticTest;
import boss.util.Histogram;
import boss.util.HistogramDouble;

public class HungarianExperiment {
	final int num_paragraphs;
	final int k;
	final double threshold;
	final int max_id;
	
	final ArrayList<int[][]> k_with_windows_b1;
	final ArrayList<int[][]> k_with_windows_b2;
	
	final ArrayList<int[]> raw_paragraphs_b1;
	final ArrayList<int[]> raw_paragraphs_b2;
	
	final ArrayList<double[][]> alignement_matrixes;
	
	final HashMap<Integer, double[]> embedding_vector_index;
	
	Solver solver = null;
	HungarianKevinStern solver_baseline;
	
	final double[] col_minima;
	final double[] row_minima;
	double col_sum;
	
	//final double[][] cost_matrix_buffer;
	
	public HungarianExperiment(ArrayList<int[]> raw_paragraphs_b1, ArrayList<int[]> raw_paragraphs_b2, final int k, final double threshold, HashMap<Integer, double[]> embedding_vector_index){
		this.k = k;
		this.threshold = threshold;
		this.raw_paragraphs_b1 = raw_paragraphs_b1;
		this.raw_paragraphs_b2 = raw_paragraphs_b2;
		this.k_with_windows_b1 = create_windows(raw_paragraphs_b1, k);
		this.k_with_windows_b2 = create_windows(raw_paragraphs_b2, k);
		this.num_paragraphs = k_with_windows_b1.size();
		if(k_with_windows_b2.size()!=num_paragraphs) {
			System.err.println("k_with_windows_b2.size()!=size");
		}
		this.embedding_vector_index = embedding_vector_index;
		
		/*int max_length_line = max(raw_paragraphs_b1);
		int max_length_column = max(raw_paragraphs_b2);
		this.cost_matrix_buffer = new double[max_length_column][max_length_line];*/
		
		//out(raw_paragraphs_b1, k_with_windows_b1);
		//out(raw_paragraphs_b2, k_with_windows_b2);
		
		//Prepare the buffers for the alignment matrixes
		this.alignement_matrixes = new ArrayList<double[][]>(num_paragraphs);
		for(int p=0;p<num_paragraphs;p++) {
			//For each paragraph - get the k-width windows
			final int[][] k_windows_p1 = this.k_with_windows_b1.get(p);
			final int[][] k_windows_p2 = this.k_with_windows_b2.get(p);
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = new double[k_windows_p1.length][k_windows_p2.length];
			this.alignement_matrixes.add(alignment_matrix);
		}
		
		this.col_minima = new double[k];
		this.row_minima = new double[k];
		
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
		
		this.solver_baseline = new HungarianKevinStern(k);
	}
	
	private int max(ArrayList<int[]> raw_pragraph) {
		int max_length = 0;
		for(int[] array : raw_pragraph) {
			if(array.length>max_length) {
				max_length = array.length;
			}
		}
		return max_length;
	}

	public void set_solver(Solver s){
		this.solver = s;
	}
	
	/**
	 * 
	 * @param raw_paragraphs all the paragraphs
	 * @param k - window size
	 * @return
	 */
	private ArrayList<int[][]> create_windows(ArrayList<int[]> raw_paragraphs, final int k) {
		ArrayList<int[][]> windows = new ArrayList<int[][]>(raw_paragraphs.size());
		for(int[] paragraph : raw_paragraphs) {//for each paragraph
			
			int[][] paragraph_windows = new int[paragraph.length-k+1][k];//pre-allocate the storage space for the
			for(int i=0;i<paragraph_windows.length;i++){
				//create one window
				for(int j=0;j<k;j++) {
					paragraph_windows[i][j] = paragraph[i+j];
				}
			}
			windows.add(paragraph_windows);
		}
		return windows;
	}
	
	public final boolean VERBOSE = false;
	public final boolean TO_FILE = false;
	FileWriter f;
	BufferedWriter output;
	
	/**
	 * The main difference to the run() method below is the way how we compute the similarity (or distance) of two k-width windows. 
	 * Normally we use the assignment problem. Here, we average the vectors in one k-width window to kind of represent the average semantics....
	 */
	void run_idea_nikolaus() {
		System.out.println("HungarianExperiment.run_idea_nikolaus() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold);
		if(this.solver==null) {
			System.err.println("Solver is null: Using StupidSolver");
			this.solver = new StupidSolver(k);
		}
		
		//get embedding vector size 
		int embedding_vector_size = -1;
		double[] vec = this.embedding_vector_index.get(1);//some id
		if(vec!=null) {
			embedding_vector_size = vec.length;
		}else{
			System.err.println("run_idea_nikolaus(): vec == null");
		}
		
		final double[] avg_vec_window_1 = new double[embedding_vector_size];
		final double[] avg_vec_window_2 = new double[embedding_vector_size];
		
		double stop,start;
		for(int p=0;p<num_paragraphs;p++) {
			start = System.currentTimeMillis();
			//For each paragraph - get the k-width windows
			final int[][] k_windows_p1 = this.k_with_windows_b1.get(p);
			final int[][] k_windows_p2 = this.k_with_windows_b2.get(p);
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = new double[k_windows_p1.length][k_windows_p2.length];
			//For each pair of windows
			for(int line=0;line<k_windows_p1.length;line++) {
				//get the line to get rid of 2D array resolution
				final double[] alignment_matrix_line = alignment_matrix[line];
				for(int column=0;column<k_windows_p2.length;column++) {
					final int[] current_window_p1 = k_windows_p1[line];
					final int[] current_window_p2 = k_windows_p2[column];
					
					double cost = get_dist_avg_vectors(current_window_p1,current_window_p2,avg_vec_window_1,avg_vec_window_2);
					//normalize costs: Before it was distance. Now it is similarity.
					double normalized_similarity = 1.0 - (cost / (double)k);
					if(normalized_similarity<threshold) {
						normalized_similarity = 0;
					}
					alignment_matrix_line[column] = normalized_similarity;
				}
			}
			stop = System.currentTimeMillis();
			System.out.println("P="+p+"\t"+(stop-start)+"\tms");
			this.alignement_matrixes.add(alignment_matrix);
		}
		
		String experiment_name = "idea_nikolaus";//default experiment, has no special name
		print_results(experiment_name, null);//TODO
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

	private double[] get_avg_vector(final int[] window, final double[] buffer) {
		final int vector_size = buffer.length;
		//clear the buffer
		for(int dim=0;dim<vector_size;dim++) {
			buffer[dim] = 0;
		}
		
		double num_vecotrs_p1 = 0;
		for(int id : window) {
			double[] temp = this.embedding_vector_index.get(id);
			if(temp!=null) {
				for(int dim=0;dim<vector_size;dim++) {
					buffer[dim] += temp[dim];
				}
				num_vecotrs_p1++;
			}
		}
		if(num_vecotrs_p1==0) {
			return null;
		}else{
			for(int dim=0;dim<vector_size;dim++) {
				buffer[dim] /= num_vecotrs_p1;
			}
		}
		return buffer;
	}
	
	
	private final double get_dist_avg_vectors(final int[] window_p1, final int[] window_p2, final double[] buffer_1, final double[] buffer_2) {
		final double[] avg_vector_1 = get_avg_vector(window_p1, buffer_1);
		final double[] avg_vector_2 = get_avg_vector(window_p2, buffer_2);
		if(avg_vector_1==null || avg_vector_2==null ) {
			return MAX_DIST;
		}
		final double dist = cosine_distance(avg_vector_1, avg_vector_2);
		return dist;
	}

	public void run_cost_matrix_experiment(){
		System.out.println("HungarianExperiment.run_cost_matrix_experiment() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold);
		final double[][] cost_matrix = new double[k][k];
		if(this.solver==null) {
			System.err.println("Solver is null: Using StupidSolver");
			this.solver = new StupidSolver(k);
		}
		System.out.println("k="+k+"\t t="+threshold);
		System.out.println("Paragraph\tAll\tCost Matrix");
		double stop,start;
		for(int p=0;p<num_paragraphs;p++) {
			start = System.currentTimeMillis();
			//For each paragraph - get the k-width windows
			final int[][] k_windows_p1 = this.k_with_windows_b1.get(p);
			final int[][] k_windows_p2 = this.k_with_windows_b2.get(p);
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = new double[k_windows_p1.length][k_windows_p2.length];
			
			double start_cost_matrix = System.currentTimeMillis(); 
			final double[][] global_cost_matrix_buffer = fill_cost_matrix(p);
			double stop_cost_matrix = System.currentTimeMillis();
			//System.out.println("p="+p+"\t Cost matrix computation\t"+(System.currentTimeMillis()-start_cost_matrix));
			
			double start_max_matrix = System.currentTimeMillis();
			final double[][] max_costs = new double[alignment_matrix.length][alignment_matrix[0].length];
			fill_max_costs(global_cost_matrix_buffer, max_costs);
			double stop_max_matrix = System.currentTimeMillis();
			
			double start_sum_matrix = System.currentTimeMillis();
			final double[][] sum_costs = new double[alignment_matrix.length][alignment_matrix[0].length];
			fill_sum_costs(global_cost_matrix_buffer, sum_costs);
			double stop_sum_matrix = System.currentTimeMillis();
			
			double start_cell_matrix = System.currentTimeMillis();
			//final boolean[][] compute_this_cell = new boolean[alignment_matrix.length][alignment_matrix[0].length];
			//fill_compute_this_cell(global_cost_matrix_buffer,compute_this_cell);
			final double[][] max_line_col_costs = new double[alignment_matrix.length][alignment_matrix[0].length];
			fill_max_line_col_costs(global_cost_matrix_buffer, max_line_col_costs);
			double stop_cell_matrix = System.currentTimeMillis();
			
			//For each pair of windows
			for(int line=0;line<k_windows_p1.length;line++) {
				//get the line to get rid of 2D array resolution
				final double[] alignment_matrix_line = alignment_matrix[line];
				
				
				for(int column=0;column<k_windows_p2.length;column++) {	
					//Fill local matrix of the current window combination from global matrix
					//Note it is cost matrix with cosine distance. I.e, not similarity. 
					for(int i=0;i<this.k;i++) {
						for(int j=0;j<this.k;j++) {
							cost_matrix[i][j] = global_cost_matrix_buffer[line+i][column+j];
						}
					}
					//That's the important line
					double cost = this.solver.solve(cost_matrix, threshold);
					//normalize costs: Before it was distance. Now it is similarity.
					double normalized_similarity = 1.0 - (cost / (double)k);
					if(normalized_similarity<threshold) {
						normalized_similarity = 0;
					}
					alignment_matrix_line[column] = normalized_similarity;
				}
			}
			stop = System.currentTimeMillis();
			System.out.print("P="+p+"\t"+(stop-start)+"\t"+(stop_cost_matrix-start_cost_matrix)+"\t"+(stop_max_matrix-start_max_matrix)+"\t"+(stop_sum_matrix-start_sum_matrix)+"\t"+(stop_cell_matrix-start_cell_matrix));
			this.alignement_matrixes.add(alignment_matrix);
			check(alignment_matrix, max_costs, sum_costs, max_line_col_costs);
		}
		String experiment_name = "";//default experiment, has no special name
		print_results(experiment_name, null);//TODO
	}
	
	/*//TODO umdrehen
	private void fill_sum_costs(final double[][] global_cost_matrix, final double[][] sum_costs) {
		final double dist_threshold = 1-this.threshold;//TODO check me
		final double[][] temp = new double[this.k][sum_costs[0].length];
		//XXX Boah....
		for(int line = 0;line<temp.length;line++) {
			final double[] line_temp = temp[line];
			final double[] line_global_cost_matrix = global_cost_matrix[line];
			double sum = 0.0;
			
			for(int i=0;i<k;i++) {
				sum+=line_global_cost_matrix[i];
			}
			line_temp[0] = sum;
		}
		
		for(int line = 0;line<temp.length;line++) {
			final double[] line_temp = temp[line];
			final double[] line_global_cost_matrix = global_cost_matrix[line+k];
			double sum = 0.0;
			
			for(int i=0;i<k;i++) {
				sum+=line_global_cost_matrix[i];
			}
			line_temp[0] = sum;
		}
		
		for(int line=0;line<sum_costs.length;line++) {
			//sum up first k values
			int column=0;
			double sum;
			for(;column<k;column++) {
				sum+= global_cost_matrix[line][column+j];
			}
			
			for(int column=0;column<sum_costs[0].length;column++) {	
				//check whether there is some value > threshold in the cost matrix
				for(int i=0;i<this.k;i++) {
					for(int j=0;j<this.k;j++) {
						double cost = global_cost_matrix[line+i][column+j];
						if(cost>dist_threshold) {
							compute_this_cell[i][j] = true;
						}
					}
				}
			}
		}
	}*/

	private void check(double[][] alignment_matrix, double[][] max_costs, double[][] sum_costs,
			double[][] max_line_column_costs) {
		final int num_lines = alignment_matrix.length;
		final int num_columns = alignment_matrix[0].length;
		
		boolean[][] baseline = get_filter_matrix(alignment_matrix, threshold,num_lines, num_columns);
		boolean[][] max_excludes = get_filter_matrix(max_costs, 1-threshold,num_lines, num_columns);//TODO 1-dist vs. <=
		boolean[][] sum_excludes = get_filter_matrix(sum_costs, 1-threshold,num_lines, num_columns);
		boolean[][] max_line_col_excludes = get_filter_matrix(max_line_column_costs, 1-threshold,num_lines, num_columns);
		
		final int num_cells = num_lines*num_columns;
		int baseline_cells_to_compute = 0;
		int max_cells_to_compute = 0;
		int sum_cells_to_compute = 0;
		int compute_cells_to_compute = 0;
		
		//System.out.println("Comparing baseline and max_excludes");
		for(int line = 0;line<baseline.length;line++) {
			for(int column= 0;column<baseline[0].length;column++) {
				boolean baseline_need_to_compute = baseline[line][column];
				boolean bound_need_to_compute 	 = max_excludes[line][column];
				if(baseline_need_to_compute) {
					baseline_cells_to_compute++;//only here
					max_cells_to_compute++;
					
					if(!bound_need_to_compute) {//This must not happen
						System.err.println("Wrong bound for max_excludes\t"+line+"\t"+column);
					}
				}else{
					if(bound_need_to_compute) {
						max_cells_to_compute++;
					}
				}
			}
		}
		//System.out.println("Comparing baseline and sum_excludes");
		for(int line = 0;line<baseline.length;line++) {
			for(int column= 0;column<baseline[0].length;column++) {
				boolean baseline_need_to_compute = baseline[line][column];
				boolean bound_need_to_compute 	 = sum_excludes[line][column];
				if(baseline_need_to_compute) {
					sum_cells_to_compute++;
					if(!bound_need_to_compute) {//This must not happen
						System.err.println("Wrong bound for sum_excludes\t"+line+"\t"+column);
					}
				}else{
					if(bound_need_to_compute) {
						sum_cells_to_compute++;
					}
				}
			}
		}
		//System.out.println("Comparing baseline and compute_this_cell");
		for(int line = 0;line<baseline.length;line++) {
			for(int column= 0;column<baseline[0].length;column++) {
				boolean baseline_need_to_compute = baseline[line][column];
				boolean bound_need_to_compute 	 = max_line_col_excludes[line][column];
				if(baseline_need_to_compute) {
					compute_cells_to_compute++;
					if(!bound_need_to_compute) {//This must not happen
						System.err.println("Wrong bound for compute_this_cell\t"+line+"\t"+column);
					}
				}else{
					if(bound_need_to_compute) {
						compute_cells_to_compute++;
					}
				}
			}
		}
		/*System.out.println("Statistics\tall\tneed");
		System.out.println("Baseline\t"+num_cells+"\t"+baseline_cells_to_compute);
		System.out.println("LineMax\t"+num_cells+"\t"+max_cells_to_compute);
		System.out.println("LineSum\t"+num_cells+"\t"+sum_cells_to_compute);
		System.out.println("GlobalMax\t"+num_cells+"\t"+compute_cells_to_compute);*/
		System.out.println("\t"+num_cells+"\t"+baseline_cells_to_compute+"\t"+compute_cells_to_compute+"\t"+max_cells_to_compute+"\t"+sum_cells_to_compute);
	}

	private boolean[][] get_filter_matrix(double[][] matrix, final double threshold, final int num_lines, final int num_columns) {
		boolean[][] ret = new boolean[num_lines][num_columns];
		for(int line = 0;line<num_lines;line++) {
			for(int column= 0;column<num_columns;column++) {
				double val = matrix[line][column];
				if(val>=threshold) {
					ret[line][column] = true;
				}
			}
		}
		return ret;
	}

	private void fill_max_costs(final double[][] global_cost_matrix, final double[][] array) {
		for(int line=0;line<array.length;line++) {
			for(int column=0;column<array[0].length;column++) {	
				//check whether there is some value > threshold in the cost matrix
				double sum = 0.0;
				for(int i=0;i<this.k;i++) {
					double line_max = 1-global_cost_matrix[line+i][column];
					for(int j=1;j<this.k;j++) {
						double cost = 1-global_cost_matrix[line+i][column+j];
						if(cost>line_max) {
							line_max = cost;
						}
					}
					sum+=line_max;
				}
				array[line][column] = sum/k;
			}
		}
	}
	
	 
	private void fill_max_line_col_costs(final double[][] global_cost_matrix, final double[][] array) {
		for(int line=0;line<array.length;line++) {
			for(int column=0;column<array[0].length;column++) {	
				//check whether there is some value > threshold in the cost matrix
				double line_sum = 0.0;
				for(int i=0;i<this.k;i++) {
					double line_max = 1-global_cost_matrix[line+i][column];
					for(int j=1;j<this.k;j++) {
						double cost = 1-global_cost_matrix[line+i][column+j];
						if(cost>line_max) {
							line_max = cost;
						}
					}
					line_sum+=line_max;
				}
				double col_sum = 0.0;
				for(int j=0;j<this.k;j++) {
					double col_max = 1-global_cost_matrix[line+0][column+j];
					for(int i=1;i<this.k;i++) {	
						double cost = 1-global_cost_matrix[line+i][column+j];
						if(cost>col_max) {
							col_max = cost;
						}
					}
					col_sum+=col_max;
				}
				double min_cost = Math.min(line_sum, col_sum);
				array[line][column] = min_cost/k;
			}
		}
	}
	
	private void fill_sum_costs(final double[][] global_cost_matrix, final double[][] array) {
		for(int line=0;line<array.length;line++) {
			for(int column=0;column<array[0].length;column++) {	
				//check whether there is some value > threshold in the cost matrix
				double sum = 0.0;
				for(int i=0;i<this.k;i++) {
					double line_sum = 0.0;
					for(int j=0;j<this.k;j++) {
						line_sum+= 1-global_cost_matrix[line+i][column+j];
					}
					sum+=line_sum;
				}
				array[line][column] = sum;
			}
		}
	}
	
	private void fill_compute_this_cell(final double[][] global_cost_matrix, final boolean[][] array) {
		final double dist_threshold = 1-this.threshold;//TODO check me
		for(int line=0;line<array.length;line++) {
			for(int column=0;column<array[0].length;column++) {	
				//check whether there is some value > threshold in the cost matrix
				for(int i=0;i<this.k;i++) {
					for(int j=0;j<this.k;j++) {
						double cost = global_cost_matrix[line+i][column+j];
						if(cost>dist_threshold) {
							array[line][column] = true;
							break;
						}
					}
				}
			}
		}
	}
	
	
	static final double DOUBLE_PRECISION_BOUND = 0.0001d;
	static final boolean SAFE_MODE    = false;
	static final boolean LOGGING_MODE = false;
	
	static final int USE_COLUMN_SUM = 0;
	static final int USE_MATRIX_MAX = 1;
	static final int USE_COLUMN_ROW_SUM = 2;
	static final int USE_WINDOW_SUM = 3;
	
	
	static final int PRUNING_APPROACH = USE_COLUMN_SUM;
	
	final double to_normalized_similarity(final double cost) {
		return 1.0 - (cost / (double)k);
	}
	
	@SuppressWarnings("unused")
	public void run_pruning(){
		System.out.println("HungarianExperiment.run_pruning() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold+" pruning="+PRUNING_APPROACH);
		if(SAFE_MODE) System.err.println("SAFE_MODE");
		if(LOGGING_MODE) System.err.println("LOGGING_MODE");
		final double[][] cost_matrix = new double[k][k];
		if(this.solver==null) {
			System.err.println("Solver is null: Using StupidSolver");
			this.solver = new StupidSolver(k);
		}
		
		//Below stuff for USE_WINDOW_SUM bound
		double[] index_nearest_neighbor = null;
		boolean[] exlude_me_p1 = null;
		boolean[] exlude_me_p2 = null;
		
		if(PRUNING_APPROACH == USE_WINDOW_SUM) {
			index_nearest_neighbor = get_nearest_neighbor();
		}
		
		double[] run_times = new double[num_paragraphs];
		long[][] counts = new long[num_paragraphs][2];
			
		double stop,start;
		for(int p=0;p<num_paragraphs;p++) {
			start = System.currentTimeMillis();
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = alignement_matrixes.get(p);
			final double[][] global_cost_matrix_buffer = fill_cost_matrix(p);
			
			double[][] max_window_line_buffer=null;
			double[][] max_window_column_buffer=null;
			if(PRUNING_APPROACH != USE_WINDOW_SUM) {//Mind the !=
				fill_max_window_buffer(global_cost_matrix_buffer,max_window_line_buffer,max_window_column_buffer);
			}
			
			if(PRUNING_APPROACH == USE_WINDOW_SUM) {
				final int[][] k_windows_p1 = this.k_with_windows_b1.get(p);
				final int[][] k_windows_p2 = this.k_with_windows_b2.get(p);	
				exlude_me_p1 = get_bound_window_sum(index_nearest_neighbor, k_windows_p1);
				exlude_me_p2 = get_bound_window_sum(index_nearest_neighbor, k_windows_p2);
			}
			
			long num_cels_geq_threshold = 0;
			long num_cels_geq_threshold_estimation = 0;
			
			//For each pair of windows
			for(int line=0;line<alignment_matrix.length;line++) {
				if(PRUNING_APPROACH == USE_WINDOW_SUM) {
					if(exlude_me_p1[line]) {
						continue;//We can exclude this line
					}
				}
				//get the line to get rid of 2D array resolution
				final double[] alignment_matrix_line = alignment_matrix[line];
				
				for(int column=0;column<alignment_matrix[0].length;column++) {
					if(PRUNING_APPROACH == USE_WINDOW_SUM) {
						if(exlude_me_p2[column]) {
							continue;//We can exclude this column
						}
					}
					//Fill local matrix of the current window combination from global matrix
					//Note it is cost matrix with cosine distance. I.e, not similarity. 
					for(int i=0;i<this.k;i++) {
						for(int j=0;j<this.k;j++) {
							cost_matrix[i][j] = global_cost_matrix_buffer[line+i][column+j];
						}
					}
					
					double lb_cost;
					if(PRUNING_APPROACH == USE_COLUMN_SUM) {
						lb_cost = get_column_sum(cost_matrix);	
					}else if(PRUNING_APPROACH == USE_MATRIX_MAX){
						lb_cost = k*get_matrix_min(cost_matrix);//we assume that this value occurs k times
					}else if(PRUNING_APPROACH == USE_COLUMN_ROW_SUM){
						lb_cost = get_column_row_sum(cost_matrix);
					}else if(PRUNING_APPROACH == USE_WINDOW_SUM){
						lb_cost = 0;//No bound here
					}else{
						System.err.println("Unknonw pruning apporach");
					}
					
					final double up_normalized_similarity = 1.0 - (lb_cost / (double)k);
					if(up_normalized_similarity+DOUBLE_PRECISION_BOUND>this.threshold) {
						if(LOGGING_MODE) {num_cels_geq_threshold_estimation++;}
						//That's the important line
						double cost = this.solver.solve(cost_matrix, threshold);
						//normalize costs: Before it was distance. Now it is similarity.
						double normalized_similarity = 1.0 - (cost / (double)k);
						if(normalized_similarity>=threshold) {
							if(LOGGING_MODE) {num_cels_geq_threshold++;}
							alignment_matrix_line[column] = normalized_similarity;
						}//else keep it zero
					}
										
					{
						if(SAFE_MODE) {//if false removed by compiler
							double cost = this.solver.solve(cost_matrix, threshold);
							//normalize costs: Before it was distance. Now it is similarity.
							double normalized_similarity = 1.0 - (cost / (double)k);
							if(normalized_similarity>up_normalized_similarity+DOUBLE_PRECISION_BOUND) {
								System.err.println("normalized_similarity>normalized_estimate_similarity "+up_normalized_similarity+" "+normalized_similarity);
							}else {
								//System.out.println("normalized_similarity<=normalized_estimate_similarity "+normalized_estimate_similarity+" "+normalized_similarity);
							}
						}
					}
					if(LOGGING_MODE) {
						counts[p][0]=num_cels_geq_threshold_estimation;
						counts[p][1]=num_cels_geq_threshold;
					}
				}
			}
			stop = System.currentTimeMillis();
			run_times[p] = (stop-start);
			System.out.println("P="+p+"\t"+(stop-start)+"\tms");
		}
		String experiment_name = "";//default experiment, has no special name
		print_results(experiment_name, run_times);//TODO
		if(LOGGING_MODE) {
			System.out.println("p\tnum_cells\tbounded\tcomputed");
			for(int p=0;p<num_paragraphs;p++) {
				System.out.println(p+"\t"+alignement_matrixes.get(p).length*alignement_matrixes.get(p)[0].length+"\t"+counts[p][0]+"\t"+counts[p][1]);
			}
		}
	}
	
	private void fill_max_window_buffer(double[][] global_cost_matrix_buffer, double[][] max_window_line_buffer,
			double[][] max_window_column_buffer) {
		final int num_lines = global_cost_matrix_buffer.length;
		final int num_columns = global_cost_matrix_buffer[0].length;
		final double[] value_buffer = new double[k];
		
		max_window_line_buffer = new double[num_lines-k][num_columns];
		for(int line=0;line<num_lines;line++) {
			final double[] cost_matrix_line = global_cost_matrix_buffer[line];
			final double[] max_window_line_buffer_line = max_window_line_buffer[line];
			
			{//first window is special
				for(int i=0;i<k;i++) {
					value_buffer[i] = cost_matrix_line[i];
				}
				double window_max = max(value_buffer);
				max_window_line_buffer_line[0] = window_max;
			}
			
			//The idea is that we replace the oldest value (i.e., the one not in the window anymore) with the new one in the window
			for(int window=1;window<max_window_line_buffer_line.length;window++) {
				int replace_position = (window-1)%k;
				value_buffer[replace_position] = cost_matrix_line[k-1+window];
				double window_max = max(value_buffer);
				max_window_line_buffer_line[window] = window_max;
			}
		}
		
	}

	private final double max(final double[] values) {
		double max = values[0];
		for(int i=1;i<values.length;i++) {
			if(max<values[i]) {
				max=values[i];
			}
		}
		return max;
	}

	private boolean[] get_bound_window_sum(final double[] index_nearest_neighbor, final int[][] k_windows) {
		boolean[] exclude_me = new boolean[k_windows.length];//TODO Optimize computation exploit running window
		for(int w=0;w<k_windows.length;w++) {
			int[] window = k_windows[w];
			double min_cost = 0;
			for(int id : window){
				min_cost += index_nearest_neighbor[id];
			}
			double max_sim = to_normalized_similarity(min_cost);
			if(max_sim<threshold) {
				exclude_me[w] = true;
			}
		}
		return exclude_me;
	}

	private double get_matrix_min(final double[][] cost_matrix) {
		double min = Double.MAX_VALUE;
		for(int i=0;i<this.k;i++) {
			final double[] line = cost_matrix[i];
			for(int j=0;j<this.k;j++) {
				final double val = line[j];
				if(val<min) {
					min = val;
				}
			}
		}	
		return min;
	}
	
	private void get_column_row_minima(final double[][] cost_matrix, final double[] min_cost_lines, final double[] min_cost_columns) {
		for(int i=0;i<this.k;i++) {//TODO remove branching
			double[] line = cost_matrix[i];
			for(int j=0;j<this.k;j++) {
				final double val = line[j];
				if(j==0) {//new line
					min_cost_lines[i] = line[0];
				}else{
					if(val<min_cost_lines[i]) {
						min_cost_lines[i] = val;
					}
				}
				if(i==0) {//new column
					min_cost_columns[j] = val;
				}else{
					if(val<min_cost_columns[j]) {
						min_cost_columns[j] = val;
					}
				}
			}
			
		}
	}
	
	private double get_row_sum(final double[][] current_window, final int offset) {
		double row_sum = 0;
		for(int i=0;i<this.k;i++) {
			final double[] line = current_window[i];
			double row_min = line[offset+0];
			for(int j=1;j<this.k;j++) {
				final double val = line[offset+j];
				if(val<row_min) {
					row_min = val;
				}
			}
			row_sum += row_min;
			row_minima[i] = row_min;
		}
		
		return row_sum;
	}
	
	private double get_row_sum(final double[][] cost_matrix) {
		double row_sum = 0;
		for(int i=0;i<this.k;i++) {
			final double[] line = cost_matrix[i];
			double row_min = line[0];
			for(int j=1;j<this.k;j++) {
				final double val = line[j];
				if(val<row_min) {
					row_min = val;
				}
			}
			row_sum += row_min;
		}
		
		return row_sum;
	}
	
	private double get_column_row_sum(final double[][] cost_matrix_window, final int offset) {
		double row_sum = 0;
		Arrays.fill(this.col_minima, Double.MAX_VALUE);
		for(int i=0;i<this.k;i++) {
			final double[] line = cost_matrix_window[i];
			double row_min = Double.MAX_VALUE;
			for(int j=0;j<this.k;j++) {
				final double val = line[offset+j];
				if(val<row_min) {
					row_min = val;
				}
				if(val<col_minima[j]) {
					col_minima[j] = val;
				}
			}
			row_sum += row_min;
			row_minima[i] = row_min;
		}
		col_sum = sum(col_minima);
		double min_cost = Math.max(row_sum, col_sum);		
		
		return min_cost;
	}
	
	private double get_column_row_sum(final double[][] cost_matrix) {
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
		col_sum = sum(col_minima);
		double min_cost = Math.max(row_sum, col_sum);		
		
		return min_cost;
	}
	
	private double sum_bound_similarity_incremental(final double[][] similarity_matrix) {
		double row_sum = 0;
		for(int i=0;i<this.k;i++) {
			final double[] line = similarity_matrix[i];
			double row_min = Double.POSITIVE_INFINITY;
			for(int j=0;j<this.k;j++) {
				final double val = line[j];
				if(val<row_min) {
					row_min = val;
				}
			}
			row_sum += row_min;
		}
		double min_cost = Math.max(row_sum, col_sum);		
		
		return -min_cost;
	}
	
	private double sum_bound_similarity(final double[][] similarity_matrix) {
		double row_sum = 0;
		Arrays.fill(this.col_minima, Double.POSITIVE_INFINITY);
		for(int i=0;i<this.k;i++) {
			final double[] line = similarity_matrix[i];
			double row_min = Double.POSITIVE_INFINITY;
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
		col_sum = sum(col_minima);
		double min_cost = Math.max(row_sum, col_sum);		
		
		return -min_cost;
	}
	
	private double get_column_row_sum_safe(final double[][] cost_matrix) {
		double row_sum = 0;
		double[] k_buffer = new double[k];
		Arrays.fill(k_buffer, Double.MAX_VALUE);
		for(int i=0;i<this.k;i++) {
			final double[] line = cost_matrix[i];
			double row_min = Double.MAX_VALUE;
			for(int j=0;j<this.k;j++) {
				final double val = line[j];
				if(val<row_min) {
					row_min = val;
				}
				if(val<k_buffer[j]) {
					k_buffer[j] = val;
				}
			}
			row_sum += row_min;
		}
		double col_sum = sum(k_buffer);
		double min_cost = Math.max(row_sum, col_sum);		
		
		return min_cost;
	}
	
	private double sum(final double[] array) {
		double sum = 0;
		for(double d : array) {
			sum+=d;
		}
		return sum;
	}

	private double get_column_sum(final double[][] cost_matrix) {
		double col_sum = 0.0;
		for(int j=0;j<this.k;j++) {
			double col_min = cost_matrix[0][j];
			for(int i=1;i<this.k;i++) {	
				double cost = cost_matrix[i][j];
				if(cost<col_min) {
					col_min = cost;
				}
			}
			col_sum+=col_min;
		}
				
		return col_sum;
	}

	static double[][] dense_global_matrix_buffer = null;
	void run_baseline_global_matrix_dense(){
		System.out.println("HungarianExperiment.run_baseline_global_matrix_dense() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold);
		final double[][] cost_matrix = new double[k][k];
		if(this.solver==null) {
			System.err.println("Solver is null: Using StupidSolver");
			this.solver = new StupidSolver(k);
		}
		
		if(dense_global_matrix_buffer==null) {//XXX - this is not fair. Just for testing
			create_dense_matrix();
		}
		
		double[] run_times = new double[num_paragraphs];
			
		double stop,start;
		for(int p=0;p<num_paragraphs;p++) {
			start = System.currentTimeMillis();
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = this.alignement_matrixes.get(p);//get the pre-allocated buffer. Done in Constructor
			final int[][] k_windows_p1 = this.k_with_windows_b1.get(p);
			final int[][] k_windows_p2 = this.k_with_windows_b2.get(p);	
			
			//For each pair of windows
			for(int line=0;line<alignment_matrix.length;line++) {
				//get the line to get rid of 2D array resolution
				final double[] alignment_matrix_line = alignment_matrix[line];
								
				for(int column=0;column<alignment_matrix[0].length;column++) {	
					//Fill local matrix of the current window combination from global matrix
					//Note it is cost matrix with cosine distance. I.e, not similarity. 
					for(int i=0;i<this.k;i++) {
						final int set_id_window_p1 = k_windows_p1[line][i];
						
						for(int j=0;j<this.k;j++) {
							final int set_id_window_p2 = k_windows_p2[column][j];
							double dist = dense_global_matrix_buffer[set_id_window_p1][set_id_window_p2];
							cost_matrix[i][j] = dist;
							
							if(SAFE_MODE) {
								double dist_safe = dist(set_id_window_p1, set_id_window_p2, this.embedding_vector_index.get(set_id_window_p1), this.embedding_vector_index.get(set_id_window_p2));
								if(dist_safe!=dist) {
									System.err.println("dist_safe!=dist:\t"+dist_safe+"\t"+dist);
								}
							}
						}
					}
					//That's the important line
					double cost = this.solver.solve(cost_matrix, threshold);
					//normalize costs: Before it was distance. Now it is similarity.
					double normalized_similarity = 1.0 - (cost / (double)k);
					if(normalized_similarity<threshold) {
						normalized_similarity = 0;
					}
					alignment_matrix_line[column] = normalized_similarity;
				}
			}
			stop = System.currentTimeMillis();
			run_times[p] = (stop-start);
			System.out.println("P="+p+"\t"+(stop-start)+"\tms");
			this.alignement_matrixes.add(alignment_matrix);
		}	
		
		String experiment_name = "";//default experiment, has no special name
		print_results(experiment_name, run_times);
	}
	
	
	
	HashMap<Integer, Double> sparse_global_matrix_buffer = new HashMap<Integer, Double>();
	/**
	 * Stores how often we re-use a previously computes distance.
	 */
	HashMap<Integer, Integer> sparse_global_matrix_access_log = new HashMap<Integer, Integer>();
	/**
	 * We use a sparse matrix for the pair-wise token similarities. The idea is to avoid excessive re computations of distances.
	 */
	void run_baseline_global_matrix_sparse(){
		System.out.println("HungarianExperiment.run_baseline_global_matrix_dense() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold);
		final double[][] cost_matrix = new double[k][k];
		if(this.solver==null) {
			System.err.println("Solver is null: Using StupidSolver");
			this.solver = new StupidSolver(k);
		}
		
		double[] run_times = new double[num_paragraphs];
			
		double stop,start;
		for(int p=0;p<num_paragraphs;p++) {
			start = System.currentTimeMillis();
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = this.alignement_matrixes.get(p);//get the pre-allocated buffer. Done in Constructor
			final int[][] k_windows_p1 = this.k_with_windows_b1.get(p);
			final int[][] k_windows_p2 = this.k_with_windows_b2.get(p);	
			
			//For each pair of windows
			for(int line=0;line<alignment_matrix.length;line++) {
				//get the line to get rid of 2D array resolution
				final double[] alignment_matrix_line = alignment_matrix[line];
								
				for(int column=0;column<alignment_matrix[0].length;column++) {	
					//Fill local matrix of the current window combination from global matrix
					//Note it is cost matrix with cosine distance. I.e, not similarity. 
					for(int i=0;i<this.k;i++) {
						final int set_id_window_p1 = k_windows_p1[line][i];
						
						for(int j=0;j<this.k;j++) {
							final int set_id_window_p2 = k_windows_p2[column][j];
							
							final int key = get_key(set_id_window_p1,set_id_window_p2);
							Double dist = sparse_global_matrix_buffer.get(key);
							if(LOGGING_MODE) {
								if(dist==null) {
									sparse_global_matrix_access_log.put(key, 0);//first re use
								}else{
									Integer re_use = sparse_global_matrix_access_log.get(key);
									sparse_global_matrix_access_log.put(key, re_use.intValue()+1);
								}
							}
								
							if(dist==null) {//we haven't seen this pair before: Let's compute and cache it
								final double[] vec_1 = this.embedding_vector_index.get(set_id_window_p1);
								final double[] vec_2 = this.embedding_vector_index.get(set_id_window_p2);
								dist = dist(set_id_window_p1, set_id_window_p2, vec_1, vec_2);
								sparse_global_matrix_buffer.put(key, dist);
							}
							
							cost_matrix[i][j] = dist.doubleValue();
						}
					}
					//That's the important line
					double cost = this.solver.solve(cost_matrix, threshold);
					//normalize costs: Before it was distance. Now it is similarity.
					double normalized_similarity = 1.0 - (cost / (double)k);
					if(normalized_similarity<threshold) {
						normalized_similarity = 0;
					}
					alignment_matrix_line[column] = normalized_similarity;
				}
			}
			stop = System.currentTimeMillis();
			run_times[p] = (stop-start);
			System.out.println("P="+p+"\t"+(stop-start)+"\tms");
			this.alignement_matrixes.add(alignment_matrix);
		}
		if(LOGGING_MODE) {
			System.out.println("Computes distance for num of pairs: "+sparse_global_matrix_access_log.size());
			ArrayList<Integer> counts = new ArrayList<Integer>(sparse_global_matrix_access_log.size());
			for(Entry<Integer, Integer> e : sparse_global_matrix_access_log.entrySet()) {
				counts.add(e.getValue());
			}
			boss.util.Histogram hist = new Histogram(counts);
			System.out.println(hist.toString());
			System.out.println(hist.getStatistics());
		}
		
		
		String experiment_name = "";//default experiment, has no special name
		print_results(experiment_name, run_times);
	}
	
	/**
	 * Creates an index for the 1-nearest neighbor bound of each set
	 * 
	 * @return
	 */
	private double[] get_nearest_neighbor(){
		if(this.dense_global_matrix_buffer==null) {
			create_dense_matrix();
		}
		final int size = dense_global_matrix_buffer.length;
		double[] index_nearest_neighbor = new double[size];
		
		for(int id=0;id<size;id++) {
			final double[] line = dense_global_matrix_buffer[id];
			double min_cost = Double.MAX_VALUE;
			for(int other_id=0;other_id<size;other_id++){
				if(id==other_id) continue;//FIXME This line is basically wrong.... works only if the id is not part of the other window
				double dist = line[other_id]; 
				if(dist<min_cost) {
					min_cost = dist;
				}
			}
			index_nearest_neighbor[id] = min_cost;
		}
		
		return index_nearest_neighbor;
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
		
		if(LOGGING_MODE) {
			ArrayList<Double> raw_data = new ArrayList<Double>(size);
			for(double[] arr : dense_global_matrix_buffer) {
				for(double d : arr) {
					raw_data.add(d);
				}
			}
			HistogramDouble hist = new HistogramDouble(raw_data);
			System.out.println("size\tsum\tmin\tmax");
			System.out.println(hist.size()+"\t"+hist.sum()+"\t"+hist.get_min()+"\t"+hist.get_max());
			System.out.println(hist.getStatistics());
			System.out.println(hist);
			
			//Now weights based on occurrence frequency: Works best for Book Granularity
			raw_data.clear();
			int[] book_1 = raw_paragraphs_b1.get(0);
			int[] book_2 = raw_paragraphs_b2.get(0);
			final double[][] matrix = new double[book_1.length][book_2.length];
			for(int line = 0; line < book_1.length; line++) {
				int token_id_1 = book_1[line];
				double[] vec_1 = embedding_vector_index.get(token_id_1);
				for(int column = 0; column < book_2.length; column++){
					int token_id_2 = book_2[column];	
					double[] vec_2 = embedding_vector_index.get(token_id_2);
					double dist = dist(token_id_1, token_id_2, vec_1, vec_2);
					raw_data.add(dist);
				}
			}
			hist = new HistogramDouble(raw_data);
			System.out.println("size\tsum\tmin\tmax");
			System.out.println(hist.size()+"\t"+hist.sum()+"\t"+hist.get_min()+"\t"+hist.get_max());
			System.out.println(hist.getStatistics());
			System.out.println(hist);
		}
	}

	private int get_key(int set_id_window_p1, int set_id_window_p2) {
		if(set_id_window_p1>set_id_window_p2) {//exploit symmetry
			return set_id_window_p1 |= (set_id_window_p2 << 16);
		}else{
			return set_id_window_p2 |= (set_id_window_p1 << 16);
		}
	}

	
	void test_hungarian_implementations(){
		System.out.println("HungarianExperiment.test_hunagrian_implementations() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold);
		final double[][] cost_matrix = new double[k][k];
		this.solver = new StupidSolver(k);
		
		double[] run_times = new double[num_paragraphs];
			
		Solver HAP = new HungarianAlgorithmPranayImplementation();
		Solver H_WIKI = new HungarianAlgorithmWiki(k);
		Solver HKS = new HungarianKevinStern(k);
		
		double stop,start;
		for(int p=0;p<num_paragraphs;p++) {
			start = System.currentTimeMillis();
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = this.alignement_matrixes.get(p);//get the pre-allocated buffer. Done in Constructor
			final double[][] global_cost_matrix_buffer = fill_cost_matrix(p);
			
			//For each pair of windows
			for(int line=0;line<alignment_matrix.length;line++) {
				//get the line to get rid of 2D array resolution
				final double[] alignment_matrix_line = alignment_matrix[line];
								
				for(int column=0;column<alignment_matrix[0].length;column++) {	
					//Fill local matrix of the current window combination from global matrix
					//Note it is cost matrix with cosine distance. I.e, not similarity. 
					for(int i=0;i<this.k;i++) {
						for(int j=0;j<this.k;j++) {
							cost_matrix[i][j] = global_cost_matrix_buffer[line+i][column+j];
						}
					}
					//That's the important line
					double cost = this.solver.solve(cost_matrix, threshold);
					
					//Now the tests 
					double cost_HAP = HAP.solve(cost_matrix, threshold); 
					double cost_WIKI = H_WIKI.solve(cost_matrix, threshold); 
					double cost_HKS = HKS.solve(cost_matrix, threshold);					
					
					if(!is_equal(cost_HAP,cost)) {
						System.err.println("cost_HAP!=cost\t"+cost_HAP+"\t"+cost);
					}
					if(!is_equal(cost_WIKI,cost)){
						System.err.println("cost_WIKI!=cost\t"+cost_WIKI+"\t"+cost);
					}
					if(!is_equal(cost_HKS,cost)){
						System.err.println("cost_HKS!=cost\t"+cost_HKS+"\t"+cost);
					}
					
					//normalize costs: Before it was distance. Now it is similarity.
					double normalized_similarity = 1.0 - (cost / (double)k);
					if(normalized_similarity<threshold) {
						normalized_similarity = 0;
					}
					alignment_matrix_line[column] = normalized_similarity;
				}
			}
			stop = System.currentTimeMillis();
			run_times[p] = (stop-start);
			System.out.println("P="+p+"\t"+(stop-start)+"\tms");
			this.alignement_matrixes.add(alignment_matrix);
		}
		String experiment_name = "";//default experiment, has no special name
		print_results(experiment_name, run_times);
	}
	
	private boolean is_equal(double val_1, double val_2) {
		if(val_1-DOUBLE_PRECISION_BOUND<val_2 && val_1+DOUBLE_PRECISION_BOUND>val_2) {
			return true;
		}
		return false;
	}

	/**
	 * This should always use the best combination of all techniques.
	 * Currently this is
	 * (1) Normalized vectors to unit length
	 * (2) Hungarian implementation from Kevin Stern
	 * (3) Dense global cost matrix
	 * (4) min(row_min,column_min) bound on the local cost matrix: O(n³)
	 * @return 
	 */
	public double[] run_solution(){
		//Check config (1) Normalized vectors to unit length
		if(!MatchesWithEmbeddings.NORMALIZE_VECTORS) {
			System.err.println("run_solution(): MatchesWithEmbeddings.NORMALIZE_VECTORS=false");
		}
		//Ensure config (2) Hungarian implementation from Kevin Stern
		this.solver = new HungarianKevinStern(k);
		
		// (3) Dense global cost matrix - compute once
		if(dense_global_matrix_buffer==null) {//XXX Has extra runtime measurement inside method
			create_dense_matrix();
		}
		
		System.out.println("HungarianExperiment.run_solution() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] cost_matrix = new double[k][k];
		
		double[] run_times = new double[num_paragraphs];
			
        long count_survived_pruning = 0;
        long count_computed_cells   = 0;
		
		double stop,start;
		for(int p=0;p<num_paragraphs;p++) {
			start = System.currentTimeMillis();
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = this.alignement_matrixes.get(p);//get the pre-allocated buffer. Done in Constructor
			final int[][] k_windows_p1 = this.k_with_windows_b1.get(p);
			final int[][] k_windows_p2 = this.k_with_windows_b2.get(p);	
			
			//For each pair of windows
			for(int line=0;line<alignment_matrix.length;line++) {
				//get the line to get rid of 2D array resolution
				final double[] alignment_matrix_line = alignment_matrix[line];
								
				for(int column=0;column<alignment_matrix[0].length;column++) {	
					//Fill local matrix of the current window combination from global matrix
					//Note it is cost matrix with cosine distance. I.e, not similarity. 
					for(int i=0;i<this.k;i++) {
						final int set_id_window_p1 = k_windows_p1[line][i];
						for(int j=0;j<this.k;j++) {
							final int set_id_window_p2 = k_windows_p2[column][j];
							double dist = dense_global_matrix_buffer[set_id_window_p1][set_id_window_p2];
							cost_matrix[i][j] = dist;
						}
					}
					
					// (4) compute the bound
					final double lb_cost = get_column_row_sum(cost_matrix);
					final double up_normalized_similarity = 1.0 - (lb_cost / (double)k);
					if(up_normalized_similarity+DOUBLE_PRECISION_BOUND>this.threshold) {
						count_survived_pruning++;
						//That's the important line
						double cost = this.solver.solve(cost_matrix, threshold);
						//normalize costs: Before it was distance. Now it is similarity.
						double normalized_similarity = 1.0 - (cost / (double)k);
						if(normalized_similarity>=threshold) {
							count_computed_cells++;
							alignment_matrix_line[column] = normalized_similarity;
						}//else keep it zero
					}
					if(SAFE_MODE) {//if false removed by compiler
						double cost = this.solver.solve(cost_matrix, threshold);
						//normalize costs: Before it was distance. Now it is similarity.
						double normalized_similarity = 1.0 - (cost / (double)k);
						if(normalized_similarity>up_normalized_similarity+DOUBLE_PRECISION_BOUND) {
							System.err.println("normalized_similarity>normalized_estimate_similarity "+up_normalized_similarity+" "+normalized_similarity);
						}else {
							//System.out.println("normalized_similarity<=normalized_estimate_similarity "+normalized_estimate_similarity+" "+normalized_similarity);
						}
					}
				}
			}
			stop = System.currentTimeMillis();
			run_times[p] = (stop-start);
			int size = alignment_matrix.length*alignment_matrix[0].length;
			double check_sum = sum(alignment_matrix);
			System.out.println("P="+p+"\t"+(stop-start)+"\t"+check_sum+"\t"+size+"\t"+count_survived_pruning+"\t"+count_computed_cells);
			this.alignement_matrixes.add(alignment_matrix);
		}
		String experiment_name = "";//default experiment, has no special name
		if(VERBOSE)
			print_results(experiment_name, run_times);
		return run_times;
	}
	
	void run_baseline_zick_zack(){
		if(this.solver==null) {
			System.err.println("Solver is null: Using StupidSolver");
			this.solver = new StupidSolver(k);
		}
		System.out.println("HungarianExperiment.run_baseline_zick_zack() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] cost_matrix = new double[k][k];
		
		double[] run_times = new double[num_paragraphs];
			
		double stop,start;
		for(int p=0;p<num_paragraphs;p++) {
			start = System.currentTimeMillis();
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = this.alignement_matrixes.get(p);//get the pre-allocated buffer. Done in Constructor
			final double[][] global_cost_matrix_buffer = fill_cost_matrix(p);
			
			//For each pair of windows
			for(int line=0;line<alignment_matrix.length;line++) {
				//get the line to get rid of 2D array resolution
				final double[] alignment_matrix_line = alignment_matrix[line];
				int column=0;
				{	//Initially we really fill the entire cost matrix
					//Fill local matrix of the current window combination from global matrix
					//Note it is cost matrix with cosine distance. I.e, not similarity. 
					for(int i=0;i<this.k;i++) {
						for(int j=0;j<this.k;j++) {
							cost_matrix[i][j] = global_cost_matrix_buffer[line+i][column+j];
						}
					}
					//That's the important line
					double cost = this.solver.solve(cost_matrix, threshold);
					//normalize costs: Before it was distance. Now it is similarity.
					double normalized_similarity = 1.0 - (cost / (double)k);
					if(normalized_similarity>=threshold) {
						alignment_matrix_line[column] = normalized_similarity;
					}//else keep it zero
					column++;
				}
				
				//the idea is that we now change only one k-vector, but not re-fill the entire matrix again
				for(;column<alignment_matrix[0].length;column++) {	
					final int replace_position = (column-1)%k;
					for(int i=0;i<this.k;i++) {
						cost_matrix[i][replace_position] = global_cost_matrix_buffer[line+i][column+k-1];
					}
					//That's the important line
					double cost = this.solver.solve(cost_matrix, threshold);
					//normalize costs: Before it was distance. Now it is similarity.
					double normalized_similarity = 1.0 - (cost / (double)k);
					if(normalized_similarity<threshold) {
						normalized_similarity = 0;
					}
					alignment_matrix_line[column] = normalized_similarity;
				}
			}
			stop = System.currentTimeMillis();
			run_times[p] = (stop-start);
			System.out.println("P="+p+"\t"+(stop-start)+"\tms");
			this.alignement_matrixes.add(alignment_matrix);
		}
		String experiment_name = "";//default experiment, has no special name
		//if(VERBOSE)
			print_results(experiment_name, run_times);
	}
	/*
	public double[] run_zick_zack_deep(){
		//Check config (1) Normalized vectors to unit length
		if(!MatchesWithEmbeddings.NORMALIZE_VECTORS) {
			System.err.println("run_solution(): MatchesWithEmbeddings.NORMALIZE_VECTORS=false");
		}
		//Ensure config (2) Hungarian implementation from Kevin Stern
		HungarianDeep solver = new HungarianDeep(k);//XXX I am overwriting this.solver - not so readable I know
		this.solver = solver;
		
		// (3) Dense global cost matrix - compute once
		if(dense_global_matrix_buffer==null) {//XXX Has extra runtime measurement inside method
			create_dense_matrix();
		}
		
		USE_GLOBAL_MATRIX = true;
		
		System.out.println("HungarianExperiment.run_zick_zack() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold+" "+solver.get_name());
		double[] run_times = new double[num_paragraphs];
		double lb_cost;
		
		double[][] current_window = new double[k][];

		double stop,start;
		for(int p=0;p<num_paragraphs;p++) {
			start = System.currentTimeMillis();
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = this.alignement_matrixes.get(p);//get the pre-allocated buffer. Done in Constructor
			final double[][] global_cost_matrix_buffer = fill_cost_matrix(p);	
			
			//For each pair of windows
			for(int line=0;line<alignment_matrix.length;line++) {
				//get the line to get rid of 2D array resolution
				final double[] alignment_matrix_line = alignment_matrix[line];
				for(int i=0;i<k;i++) {
					current_window[i] = global_cost_matrix_buffer[line+i];
				}
				
				int column=0;
				{	//Initially we really fill the entire cost matrix
					lb_cost = get_column_row_sum(current_window, column);
					
					if(SAFE_MODE) {
						final int[][] k_windows_p1 = this.k_with_windows_b1.get(p);
						final int[][] k_windows_p2 = this.k_with_windows_b2.get(p);
						double[][] cost_matrix_copy = new double[k][k];
						
						for(int i=0;i<this.k;i++) {
							final int set_id_window_p1 = k_windows_p1[line][i];
							for(int j=0;j<this.k;j++) {
								final int set_id_window_p2 = k_windows_p2[column][j];
								double dist = dense_global_matrix_buffer[set_id_window_p1][set_id_window_p2];
								cost_matrix_copy[i][j] = dist;
							}
						}
						
						double lb_cost_safe = get_column_row_sum_safe(cost_matrix_copy);
						if(lb_cost!=lb_cost_safe) {
							System.err.println("lb_cost!=lb_cost_safe "+lb_cost+" "+lb_cost_safe);
						}
						double[][] cost_matrix = solver.get_matrix_copy(current_window, column);
						if(!is_equal(cost_matrix, cost_matrix_copy)) {
							System.err.println("!is_equal(cost_matrix, cost_matrix_copy) for line="+line+" col="+column);
						}
						
						double cost_copy = solver_baseline.solve(cost_matrix_copy, threshold);
						double cost = solver.solve(current_window, column, col_minima, row_minima);
						if(!is_equal(cost, cost_copy)) {
							System.err.println("cost!=cost_copy "+cost+" "+cost_copy);
						}
					}
					
					final double up_normalized_similarity = 1.0 - (lb_cost / (double)k);
					if(up_normalized_similarity+DOUBLE_PRECISION_BOUND>this.threshold) {
						//That's the important line
						double cost = solver.solve(current_window, column, col_minima, row_minima);
						//normalize costs: Before it was distance. Now it is similarity.
						double normalized_similarity = 1.0 - (cost / (double)k);
						if(normalized_similarity>=threshold) {
							alignment_matrix_line[column] = normalized_similarity;
						}//else keep it zero
					}
					
					column++;
				}
				
				//the idea is that we now change only one k-vector, but not re-fill the entire matrix again
				for(;column<alignment_matrix[0].length;column++) {
					//Update the cost matrix exploiting the rolling window. I.e., the cost matrix is ring buffer.
					final int replace_position = (column-1)%k;
					
					lb_cost = get_row_sum(current_window, column);
					min_col(current_window, replace_position, column); 
					lb_cost = Math.max(lb_cost, this.col_sum);
					
					if(SAFE_MODE){
						final int[][] k_windows_p1 = this.k_with_windows_b1.get(p);
						final int[][] k_windows_p2 = this.k_with_windows_b2.get(p);
						double[][] cost_matrix_copy = new double[k][k];
						
						for(int i=0;i<this.k;i++) {
							final int set_id_window_p1 = k_windows_p1[line][i];
							for(int j=0;j<this.k;j++) {
								final int set_id_window_p2 = k_windows_p2[column][j];
								double dist = dense_global_matrix_buffer[set_id_window_p1][set_id_window_p2];
								cost_matrix_copy[i][j] = dist;
							}
						}
						
						double lb_cost_safe = get_column_row_sum_safe(cost_matrix_copy);
						if(!is_equal(lb_cost,lb_cost_safe)) {
							System.err.println("lb_cost!=lb_cost_safe "+lb_cost+" "+lb_cost_safe);
						}
						double[][] cost_matrix = solver.get_matrix_copy(current_window, column);
						if(!is_equal(cost_matrix, cost_matrix_copy)) {
							System.err.println("!is_equal(cost_matrix, cost_matrix_copy) for line="+line+" col="+column);
						}
						
						double cost_copy = solver_baseline.solve(cost_matrix_copy, threshold);
						double cost = solver.solve(current_window, column, col_minima, row_minima);
						if(column==1030) {
							HungarianKevinSternAlmpified HKSA = new HungarianKevinSternAlmpified(k);
							HKSA.solve(cost_matrix_copy, threshold, col_minima);
						}
						if(!is_equal(cost, cost_copy)) {
							System.err.println("cost!=cost_copy "+cost+" "+cost_copy+" at l="+line+" col="+column);
						}
					}
					
					//lb_cost = get_column_row_sum(cost_matrix);
					final double up_normalized_similarity = 1.0 - (lb_cost / (double)k);
					if(up_normalized_similarity+DOUBLE_PRECISION_BOUND>this.threshold) {
						//That's the important line
						double cost = solver.solve(current_window, column, col_minima, row_minima);
						//normalize costs: Before it was distance. Now it is similarity.
						double normalized_similarity = 1.0 - (cost / (double)k);
						if(normalized_similarity>=threshold) {
							alignment_matrix_line[column] = normalized_similarity;
						}//else keep it zero
					}
				}
			}
			stop = System.currentTimeMillis();
			run_times[p] = (stop-start);
			System.out.println("P="+p+"\t"+(stop-start)+"\tms");
			this.alignement_matrixes.add(alignment_matrix);
		}
		String experiment_name = "";//default experiment, has no special name
		//if(VERBOSE)
			print_results(experiment_name, run_times);
		return run_times;
	}*/
	
	
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

	public double[] run_zick_zack(){
		//Check config (1) Normalized vectors to unit length
		if(!MatchesWithEmbeddings.NORMALIZE_VECTORS) {
			System.err.println("run_solution(): MatchesWithEmbeddings.NORMALIZE_VECTORS=false");
		}
		//Ensure config (2) Hungarian implementation from Kevin Stern
		HungarianKevinSternAlmpified solver = new HungarianKevinSternAlmpified(k);//XXX I am overwriting this.solver - not so readable I know
		this.solver = solver;
		
		// (3) Dense global cost matrix - compute once
		if(dense_global_matrix_buffer==null) {//XXX Has extra runtime measurement inside method
			create_dense_matrix();
		}
		
		System.out.println("HungarianExperiment.run_zick_zack() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] cost_matrix = new double[k][k];
		
		double[] run_times = new double[num_paragraphs];
		double lb_cost;
			
		double stop,start;
		for(int p=0;p<num_paragraphs;p++) {
			start = System.currentTimeMillis();
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = this.alignement_matrixes.get(p);//get the pre-allocated buffer. Done in Constructor
			final int[][] k_windows_p1 = this.k_with_windows_b1.get(p);
			final int[][] k_windows_p2 = this.k_with_windows_b2.get(p);	
			
			//For each pair of windows
			for(int line=0;line<alignment_matrix.length;line++) {
				//get the line to get rid of 2D array resolution
				final double[] alignment_matrix_line = alignment_matrix[line];
				int column=0;
				{	//Initially we really fill the entire cost matrix
					//Fill local matrix of the current window combination from global matrix
					//Note it is cost matrix with cosine distance. I.e, not similarity. 
					for(int i=0;i<this.k;i++) {
						final int set_id_window_p1 = k_windows_p1[line][i];
						for(int j=0;j<this.k;j++) {
							final int set_id_window_p2 = k_windows_p2[column][j];
							double dist = dense_global_matrix_buffer[set_id_window_p1][set_id_window_p2];
							cost_matrix[i][j] = dist;
						}
					}
					
					lb_cost = get_column_row_sum(cost_matrix);
					
					if(SAFE_MODE) {
						double lb_cost_safe = get_column_row_sum(cost_matrix);
						if(lb_cost!=lb_cost_safe) {
							System.err.println("lb_cost!=lb_cost_safe "+lb_cost+" "+lb_cost_safe);
						}
					}
					
					final double up_normalized_similarity = 1.0 - (lb_cost / (double)k);
					if(up_normalized_similarity+DOUBLE_PRECISION_BOUND>this.threshold) {
						//That's the important line
						double cost = solver.solve(cost_matrix, threshold, col_minima);
						//normalize costs: Before it was distance. Now it is similarity.
						double normalized_similarity = 1.0 - (cost / (double)k);
						if(normalized_similarity>=threshold) {
							alignment_matrix_line[column] = normalized_similarity;
						}//else keep it zero
					}
					
					column++;
				}
				
				//the idea is that we now change only one k-vector, but not re-fill the entire matrix again
				for(;column<alignment_matrix[0].length;column++) {
					//Update the cost matrix exploiting the rolling window. I.e., the cost matrix is ring buffer.
					final int replace_position = (column-1)%k;
					
					for(int i=0;i<this.k;i++) {
						final int set_id_window_p1 = k_windows_p1[line][i];
						final int set_id_window_p2 = k_windows_p2[column][k-1];//Always the new one
						final double cost_l_c = dense_global_matrix_buffer[set_id_window_p1][set_id_window_p2];
						cost_matrix[i][replace_position] = cost_l_c;
					}
					
					if(SAFE_MODE){
						double[][] cost_matrix_copy = new double[k][k];
						for(int i=0;i<this.k;i++) {
							final int set_id_window_p1 = k_windows_p1[line][i];
							for(int j=0;j<this.k;j++) {
								final int set_id_window_p2 = k_windows_p2[column][j];
								double dist = dense_global_matrix_buffer[set_id_window_p1][set_id_window_p2];
								cost_matrix_copy[i][j] = dist;
							}
						}

						double cost_copy = solver_baseline.solve(cost_matrix_copy, threshold);
						double cost = solver_baseline.solve(cost_matrix, threshold);
						if(!is_equal(cost, cost_copy)) {
							System.err.println("cost!=cost_copy "+cost+" "+cost_copy);
						}
					}
					
					lb_cost = get_row_sum(cost_matrix);
					min_col(cost_matrix, replace_position);
					lb_cost = Math.max(lb_cost, this.col_sum);//XXX max()
					
					//lb_cost = get_column_row_sum(cost_matrix);
					final double up_normalized_similarity = 1.0 - (lb_cost / (double)k);
					if(up_normalized_similarity+DOUBLE_PRECISION_BOUND>this.threshold) {
						//That's the important line
						double cost = solver.solve(cost_matrix, threshold, col_minima);
						//normalize costs: Before it was distance. Now it is similarity.
						double normalized_similarity = 1.0 - (cost / (double)k);
						if(normalized_similarity>=threshold) {
							alignment_matrix_line[column] = normalized_similarity;
						}//else keep it zero
					}
				}
			}
			stop = System.currentTimeMillis();
			run_times[p] = (stop-start);
			System.out.println("P="+p+"\t"+(stop-start)+"\tms");
			this.alignement_matrixes.add(alignment_matrix);
		}
		String experiment_name = "";//default experiment, has no special name
		//if(VERBOSE)
			print_results(experiment_name, run_times);
		System.out.println(solver.get_statistics());
		return run_times;
	}

	private void min_col(final double[][] current_window, final int replace_position, final int offset) {
		this.col_sum-=this.col_minima[replace_position];
		
		final int position = offset+k-1;
		int row = 0;
		double min = current_window[row][position];
		row++;
		
		for(;row<k;row++) {
			double val = current_window[row][position];
			if(val < min) {
				min = val;
			}
		}
		
		this.col_minima[replace_position] = min;
		this.col_sum+=min;
	}
	
	private void min_col(final double[][] cost_matrix, final int replace_position) {
		this.col_sum-=this.col_minima[replace_position];
		
		int row = 0;
		double min = cost_matrix[row][replace_position];
		row++;
		
		for(;row<k;row++) {
			double val = cost_matrix[row][replace_position];
			if(val < min) {
				min = val;
			}
		}
		
		this.col_minima[replace_position] = min;
		this.col_sum+=min;
	}

	final void fill_local_similarity_matrix(final int[] k_window_p1, final int[] k_window_p2, final double[][] local_similarity_matrix){
		for(int i=0;i<k;i++) {
			final int token_id_1 = k_window_p1[i];
			for(int j=0;j<this.k;j++) {
				final int token_id_2 = k_window_p2[j];
				double sim = sim_cached(token_id_1, token_id_2);
				local_similarity_matrix[i][j] = -sim;//Note the minus-trick for the Hungarian
			}
		}
	}
	
	final void fill_local_similarity_matrix_incrementally(final int[] k_window_p1, final int[] k_window_p2, final double[][] local_similarity_matrix){
		final int copy_length = k-1;
		
		col_sum-=col_minima[0];
		System.arraycopy(col_minima, 1, col_minima, 0, copy_length);
		col_minima[copy_length] = Double.MAX_VALUE;
		
		for(int i=0;i<k;i++) {
			System.arraycopy(local_similarity_matrix[i], 1, local_similarity_matrix[i], 0, copy_length);
			final int token_id_1 = k_window_p1[i];
			final int token_id_2 = k_window_p2[copy_length];
			double sim = sim_cached(token_id_1, token_id_2);
			local_similarity_matrix[i][copy_length] = -sim;//Note the minus-trick for the Hungarian
			if(-sim<col_minima[copy_length]) {
				col_minima[copy_length]=-sim;
			}
			
		}
		col_sum+=col_minima[copy_length];
	}
	
	final double sim_cached(final int token_id_1, final int token_id_2) {
		return (token_id_1==token_id_2) ? 1 : 1-dense_global_matrix_buffer[token_id_1][token_id_2]; 
	}
	
	public double[] run_best_full_scan(){
		HungarianKevinStern solver = new HungarianKevinStern(k);
		System.out.println("HungarianExperiment.run_best_full_scan() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] local_similarity_matrix = new double[k][k];
		
		double[] run_times = new double[num_paragraphs];
		
		// (3) Dense global cost matrix - compute once
		if(dense_global_matrix_buffer==null) {
			create_dense_matrix();
		}
		
		final double MAX_SIM_ADDITION_NEW_NODE = 1.0/k;
		
		double stop,start;
		for(int p=0;p<num_paragraphs;p++) {
			start = System.currentTimeMillis();
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = this.alignement_matrixes.get(p);//get the pre-allocated buffer. Done in Constructor
			final int[][] k_windows_p1 = this.k_with_windows_b1.get(p);
			final int[][] k_windows_p2 = this.k_with_windows_b2.get(p);	
			
			double prior_cell_similarity;
			double prev_min_value;
			
			int count_survived_pruning = 0;
			int count_survived_second_pruning = 0;
			int count_survived_third_pruning = 0;
			int count_cells_exceeding_threshold = 0;
			
			double ub_sum;
			double sim;
			
			//For each pair of windows
			for(int line=0;line<alignment_matrix.length;line++) {
				count_survived_pruning++;
				count_survived_second_pruning++;
				//get the line to get rid of 2D array resolution
				final double[] alignment_matrix_line = alignment_matrix[line];
				
				int column=0;			
				{//Here we have no bound
					fill_local_similarity_matrix(k_windows_p1[line], k_windows_p2[column], local_similarity_matrix); 
					/*if(line==9) {
						System.err.println("line==9");
					}*/
					ub_sum = sum_bound_similarity(local_similarity_matrix)/(double)k;
					
					if(ub_sum+DOUBLE_PRECISION_BOUND>=threshold) {
						sim = -solver.solve_inj(local_similarity_matrix, threshold, col_minima);//Note the minus-trick for the Hungarian
						sim /= k;
						if(sim>=threshold) {
							count_cells_exceeding_threshold++;
							alignment_matrix_line[column] = sim;
						}//else keep it zero
						prior_cell_similarity = sim;
						if(SAFE_MODE) {
							double sim_save = -solver_baseline.solve(local_similarity_matrix, threshold);//Note the minus-trick for the Hungarian
							sim_save /= k;
							if(!is_equal(sim_save, sim)) {
								System.err.println("!is_equal(sim_save, sim)");
							}
						}
					}else{
						prior_cell_similarity = ub_sum;
					}
					
					prev_min_value = max(local_similarity_matrix);
					
					if(SAFE_MODE) {
						double sim_save = -solver_baseline.solve(local_similarity_matrix, threshold);//Note the minus-trick for the Hungarian
						//normalize 
						sim_save /= k;
						if(prior_cell_similarity+DOUBLE_PRECISION_BOUND<sim_save) {//not the prior cell
							System.err.println("sim_prior_cell<sim");
						}
						if(ub_sum+DOUBLE_PRECISION_BOUND<sim_save) {//not the prior cell
							System.err.println("upper_bound_sim<sim");
						}
	
						prior_cell_similarity=sim_save;
						prev_min_value = max(local_similarity_matrix);
					}
				}
				
				//For all other columns: Here we have a bound
				for(column=1;column<alignment_matrix[0].length;column++) {		
					double upper_bound_sim = prior_cell_similarity + MAX_SIM_ADDITION_NEW_NODE;
					upper_bound_sim-= (prev_min_value / k);
					
					//we always update the matrix
					fill_local_similarity_matrix_incrementally(k_windows_p1[line], k_windows_p2[column], local_similarity_matrix);
					prev_min_value = max(local_similarity_matrix);
					
					if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
						count_survived_pruning++;
						double max_sim_new_node = -col_minima[k-1];
						upper_bound_sim-=MAX_SIM_ADDITION_NEW_NODE;
						upper_bound_sim+=(max_sim_new_node/k);
						 
						if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
							count_survived_second_pruning++;
							
							ub_sum = sum_bound_similarity(local_similarity_matrix)/k;
							upper_bound_sim = (ub_sum<upper_bound_sim) ? ub_sum : upper_bound_sim;//The some bound is not necessarily tighter
							
							if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {	
								count_survived_third_pruning++;
								//That's the important line
								sim = -solver.solve_inj(local_similarity_matrix, threshold, col_minima);//Note the minus-trick for the Hungarian
								//normalize 
								sim /= k;
								
								if(sim>=threshold) {
									count_cells_exceeding_threshold++;
									alignment_matrix_line[column] = sim;
								}//else keep it zero
								prior_cell_similarity = sim;
								
								if(SAFE_MODE) {
									double sim_save = -solver_baseline.solve(local_similarity_matrix, threshold);//Note the minus-trick for the Hungarian
									sim_save /= k;
									if(!is_equal(sim_save, sim)) {
										System.err.println("!is_equal(sim_save, sim)");
									}
								}
							}else{
								prior_cell_similarity = upper_bound_sim;
							}
						}else{
							prior_cell_similarity = upper_bound_sim;
						}
					}else{
						prior_cell_similarity = upper_bound_sim;
					}
					
					if(SAFE_MODE) {
						//Fill local matrix of the current window combination from global matrix
						double[][] local_similarity_matrix_copy = new double[k][k];
						fill_local_similarity_matrix(k_windows_p1[line], k_windows_p2[column], local_similarity_matrix_copy); 
						if(!is_equal(local_similarity_matrix, local_similarity_matrix_copy)){
							System.err.println("!is_equal(local_similarity_matrix, local_similarity_matrix_copy)");
						}
						
						double sim_save = -solver_baseline.solve(local_similarity_matrix_copy, threshold);//Note the minus-trick for the Hungarian
						//normalize 
						sim_save /= k;
						if(prior_cell_similarity+DOUBLE_PRECISION_BOUND<sim_save) {//not the prior cell
							System.err.println("sim_prior_cell<sim");
						}
						if(upper_bound_sim+DOUBLE_PRECISION_BOUND<sim_save) {//not the prior cell
							System.err.println("upper_bound_sim<sim");
						}
						prior_cell_similarity=sim_save;//XXX evil trick to check whether the bounds are correct.
					}
				}
			}
			stop = System.currentTimeMillis();
			run_times[p] = (stop-start);
			int size = size(alignment_matrix);
			double check_sum = sum(alignment_matrix);
			System.out.println("k="+p+"\t"+(stop-start)+"\tms\t"+check_sum+"\t"+size+"\t"+count_survived_pruning+"\t"+count_survived_second_pruning+"\t"+count_survived_third_pruning+"\t"+count_cells_exceeding_threshold);
			this.alignement_matrixes.add(alignment_matrix);
		}
		return run_times;
	}
	
	public double[] run_incremental_cell_pruning_pranay(){
		this.solver = new HungarianKevinStern(k);
		System.out.println("HungarianExperiment.run_incremental_cell_pruning_pranay() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] local_similarity_matrix = new double[k][k];
		
		double[] run_times = new double[num_paragraphs];
		
		// (3) Dense global cost matrix - compute once
		if(dense_global_matrix_buffer==null) {
			create_dense_matrix();
		}
		
		final double MAX_SIM_ADDITION_NEW_NODE = 1.0/k;
		
		double stop,start;
		for(int p=0;p<num_paragraphs;p++) {
			start = System.currentTimeMillis();
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = this.alignement_matrixes.get(p);//get the pre-allocated buffer. Done in Constructor
			final int[][] k_windows_p1 = this.k_with_windows_b1.get(p);
			final int[][] k_windows_p2 = this.k_with_windows_b2.get(p);	
			
			double prior_cell_similarity;
			boolean prior_cell_updated_matrix;
			double prev_min_value;
			
			int count_survived_pruning = 0;
			int count_survived_second_pruning = 0;
			int count_cells_exceeding_threshold = 0;
			
			//For each pair of windows
			for(int line=0;line<alignment_matrix.length;line++) {
				count_survived_pruning++;
				count_survived_second_pruning++;
				//get the line to get rid of 2D array resolution
				final double[] alignment_matrix_line = alignment_matrix[line];
				
				int column=0;			
				{//Here we have no bound
					fill_local_similarity_matrix(k_windows_p1[line], k_windows_p2[column], local_similarity_matrix); 
					double sim = -this.solver.solve(local_similarity_matrix, threshold);//Note the minus-trick for the Hungarian
					sim /= k;
					
					if(sim>=threshold) {
						count_cells_exceeding_threshold++;
						alignment_matrix_line[column] = sim;
					}//else keep it zero
					
					prior_cell_similarity = sim;
					prior_cell_updated_matrix = true;
					prev_min_value = max(local_similarity_matrix);
				}
				
				//For all other columns: Here we have a bound
				for(column=1;column<alignment_matrix[0].length;column++) {		
					/*if(column==25) {
						System.err.println("column==25");
					}*/
					double upper_bound_sim = prior_cell_similarity + MAX_SIM_ADDITION_NEW_NODE;
					//double min_sim_deleted_node = min(k_windows_p1[line], k_windows_p2[column-1][0]);
					//upper_bound_sim-=min_cost_deleted_node;
					if(prior_cell_updated_matrix) {
						upper_bound_sim-= (prev_min_value / k);
					}
					
					if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
						count_survived_pruning++;
						//Fill local matrix of the current window combination from global matrix
						fill_local_similarity_matrix(k_windows_p1[line], k_windows_p2[column], local_similarity_matrix); 
					
						double max_sim_new_node = min(local_similarity_matrix);
						upper_bound_sim-=MAX_SIM_ADDITION_NEW_NODE;
						upper_bound_sim+=(max_sim_new_node/k);
						
						if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
							count_survived_second_pruning++;
							//That's the important line
							double sim = -this.solver.solve(local_similarity_matrix, threshold);//Note the minus-trick for the Hungarian
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
						prev_min_value = max(local_similarity_matrix);
						prior_cell_updated_matrix = true;
					}else{
						prior_cell_updated_matrix = false;
						prior_cell_similarity = upper_bound_sim;
					}
					
					if(SAFE_MODE) {
						//Fill local matrix of the current window combination from global matrix
						fill_local_similarity_matrix(k_windows_p1[line], k_windows_p2[column], local_similarity_matrix); 
						double sim = -solver_baseline.solve(local_similarity_matrix, threshold);//Note the minus-trick for the Hungarian
						//normalize 
						sim /= k;
						if(prior_cell_similarity+DOUBLE_PRECISION_BOUND<sim) {//not the prior cell
							System.err.println("sim_prior_cell<sim");
						}
						if(upper_bound_sim+DOUBLE_PRECISION_BOUND<sim) {//not the prior cell
							System.err.println("upper_bound_sim<sim");
						}
						prior_cell_similarity=sim;
						prior_cell_updated_matrix = true;
						prev_min_value = max(local_similarity_matrix);
					}
				}
			}
			stop = System.currentTimeMillis();
			run_times[p] = (stop-start);
			int size = size(alignment_matrix);
			double check_sum = sum(alignment_matrix);
			System.out.println("P="+p+"\t"+(stop-start)+"\tms\t"+check_sum+"\t"+size+"\t"+count_survived_pruning+"\t"+count_survived_second_pruning+"\t"+count_cells_exceeding_threshold);
			this.alignement_matrixes.add(alignment_matrix);
		}
		return run_times;
	}
	
	private double max(double[][] local_similarity_matrix) {
		double max = Double.NEGATIVE_INFINITY;
		for(double[] line : local_similarity_matrix) {
			if(max<line[0]) {//similarity of the deleted token
				max=line[0];
			}
		}
		return -max;
	}

	private double min(final int[] window, final int deleted_token) {
		double min = sim_cached(deleted_token,window[0]);
		for(int i=1;i<window.length;i++) {
			double sim = sim_cached(deleted_token,window[i]);
			if(sim<min) {
				min=sim;
			}
		}
		return min;
	}
	
	private double min(final double[][] window) {
		double min = window[0][k-1];
		for(int line=1;line<window.length;line++) {
			if(min>window[line][k-1]) {
				min=window[line][k-1];
			}
		}
		return -min;
	}

	public double[] run_incremental_cell_pruning(){
		this.solver = new HungarianKevinStern(k);
		System.out.println("HungarianExperiment.run_incremental_cell_pruning() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] local_similarity_matrix = new double[k][k];
		
		double[] run_times = new double[num_paragraphs];
		
		// (3) Dense global cost matrix - compute once
		if(dense_global_matrix_buffer==null) {
			create_dense_matrix();
		}
		
		final double MAX_SIM_ADDITION_NEW_NODE = 1.0/k;
		
		double stop,start;
		for(int p=0;p<num_paragraphs;p++) {
			start = System.currentTimeMillis();
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = this.alignement_matrixes.get(p);//get the pre-allocated buffer. Done in Constructor
			final int[][] k_windows_p1 = this.k_with_windows_b1.get(p);
			final int[][] k_windows_p2 = this.k_with_windows_b2.get(p);	
			
			/**
			 * Either the correct value or some upper bound. Use only for column>0 (there it is firstly initialized)
			 */
			double prior_cell_similarity;
			boolean prior_cell_exact_similarity;
			
			int count_survived_pruning = 0;
			int count_cells_exceeding_threshold = 0;
			
			//For each pair of windows
			for(int line=0;line<alignment_matrix.length;line++) {
				count_survived_pruning++;
				//get the line to get rid of 2D array resolution
				final double[] alignment_matrix_line = alignment_matrix[line];
				
				int column=0;			
				{//Here we have no bound
					fill_local_similarity_matrix(k_windows_p1[line], k_windows_p2[column], local_similarity_matrix); 
					
					//That's the important line
					double sim = -this.solver.solve(local_similarity_matrix, threshold);//Note the minus-trick for the Hungarian
					//normalize 
					sim /= k;
					
					if(sim>=threshold) {
						alignment_matrix_line[column] = sim;
					}//else keep it zero
					
					prior_cell_similarity = sim;
					prior_cell_exact_similarity = true;
					column++;
				}
				
				//For all other columns: Here we have a bound
				for(;column<alignment_matrix[0].length;column++) {		
					/*if(column==25) {
						System.err.println("column==25");
					}*/
					double upper_bound_sim = prior_cell_similarity + MAX_SIM_ADDITION_NEW_NODE;
					if(prior_cell_exact_similarity) {
						int deleted_node_assigment = solver.get_deleted_node_assigment();
						double sim_deleted_node = local_similarity_matrix[deleted_node_assigment][0];
						upper_bound_sim += (sim_deleted_node/k);
					}
					
					if(upper_bound_sim+DOUBLE_PRECISION_BOUND>=threshold) {
						count_survived_pruning++;
						//Fill local matrix of the current window combination from global matrix
						fill_local_similarity_matrix(k_windows_p1[line], k_windows_p2[column], local_similarity_matrix); 
						//That's the important line
						double sim = -this.solver.solve(local_similarity_matrix, threshold);//Note the minus-trick for the Hungarian
						//normalize 
						sim /= k;
						
						if(sim>=threshold) {
							count_cells_exceeding_threshold++;
							alignment_matrix_line[column] = sim;
						}//else keep it zero
						prior_cell_similarity = sim;
						prior_cell_exact_similarity = true;
					}else{
						prior_cell_similarity = upper_bound_sim;
						prior_cell_exact_similarity = false;
					}
					
					if(SAFE_MODE) {
						//Fill local matrix of the current window combination from global matrix
						fill_local_similarity_matrix(k_windows_p1[line], k_windows_p2[column], local_similarity_matrix); 
						double sim = -this.solver.solve(local_similarity_matrix, threshold);//Note the minus-trick for the Hungarian
						//normalize 
						sim /= k;
						if(prior_cell_similarity+DOUBLE_PRECISION_BOUND<sim) {//not the prior cell
							System.err.println("sim_prior_cell<sim");
						}
					}
					
					/*if(SAFE_MODE) {
						for(int i=0;i<this.k;i++) {
							final int set_id_window_p1 = k_windows_p1[line][i];
							for(int j=0;j<this.k;j++) {
								final int set_id_window_p2 = k_windows_p2[column][j];
								double cost = dense_global_matrix_buffer[set_id_window_p1][set_id_window_p2];
								local_similarity_matrix[i][j] = cost;//Here it is a cost matrix
							}
						}
						double cost = this.solver.solve(local_similarity_matrix, threshold);
						cost /=k;
						double cost_sim = 1 -cost;
						if(!is_equal(sim, cost_sim)) {
							System.err.println("sim!=1-cost");
						}
					}*/
				}
			}
			stop = System.currentTimeMillis();
			run_times[p] = (stop-start);
			int size = size(alignment_matrix);
			double check_sum = sum(alignment_matrix);
			System.out.println("P="+p+"\t"+(stop-start)+"\tms\t"+check_sum+"\t"+size+"\t"+count_survived_pruning+"\t"+count_cells_exceeding_threshold);
			this.alignement_matrixes.add(alignment_matrix);
		}
		return run_times;
	}
	
	private int size(double[][] alignment_matrix) {
		return alignment_matrix.length*alignment_matrix[0].length;
	}

	public double[] run_check_node_deletion(){
		HungarianKevinSternAlmpified HKS = new HungarianKevinSternAlmpified(k);
		System.out.println("HungarianExperiment.run_check_node_deletion() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] cost_matrix = new double[k][k];
		
		// (3) Dense global cost matrix - compute once
		if(dense_global_matrix_buffer==null) {//XXX Has extra runtime measurement inside method
			create_dense_matrix();
		}
		
		double[] run_times = new double[num_paragraphs];
			
		double bound_next_cell = Double.MAX_VALUE;
		final double MAX_SIM_ADDITION_NEW_NODE = 1.0/k;
		
		double stop,start;
		for(int p=0;p<num_paragraphs;p++) {
			start = System.currentTimeMillis();
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = this.alignement_matrixes.get(p);//get the pre-allocated buffer. Done in Constructor
			
			//For each pair of windows
			for(int line=0;line<alignment_matrix.length;line++) {
				//get the line to get rid of 2D array resolution
				final double[] alignment_matrix_line = alignment_matrix[line];
				final int[][] k_windows_p1 = this.k_with_windows_b1.get(p);
				final int[][] k_windows_p2 = this.k_with_windows_b2.get(p);					
				
				for(int column=0;column<alignment_matrix[0].length;column++) {
					if(column==0) {
						bound_next_cell = Double.MAX_VALUE;
					}
					//Fill local matrix of the current window combination from global matrix
					//Note it is cost matrix with cosine distance. I.e, not similarity. 
					//Fill local matrix of the current window combination from global matrix
					//Note it is cost matrix with cosine distance. I.e, not similarity. 
					for(int i=0;i<this.k;i++) {
						final int set_id_window_p1 = k_windows_p1[line][i];
						for(int j=0;j<this.k;j++) {
							final int set_id_window_p2 = k_windows_p2[column][j];
							double dist = dense_global_matrix_buffer[set_id_window_p1][set_id_window_p2];
							cost_matrix[i][j] = dist;
						}
					}
					
					double cost_injected = -1;
					
					if(column>0) {//There is a prior bound
						cost_injected = HKS.solve_injected(cost_matrix, threshold);
					}
					
					//That's the important line
					double cost = HKS.solve(cost_matrix, threshold);
					if(column>0) {//There is a prior bound
						if(bound_next_cell+DOUBLE_PRECISION_BOUND<cost) {
							System.err.println("bound_next_cell+DOUBLE_PRECISION_BOUND<cost");
						}
						if(cost_injected!=cost) {
							System.err.println("cost_injected!=cost");
						}
					}
					
					HKS.delete_node(cost_matrix, 0);
					double non_optimal_cost = HKS.get_cost(cost_matrix);
					if(cost>non_optimal_cost+cost+DOUBLE_PRECISION_BOUND) {
						System.err.println("cost+DOUBLE_PRECISION_BOUND>non_optimal_cost");
					}
					
					bound_next_cell = non_optimal_cost+1;
					int deleted_node_assigment = HKS.get_deleted_node_assigment();
					double sim_deleted_node = cost_matrix[deleted_node_assigment][0];
					bound_next_cell -= sim_deleted_node;
					
					//normalize costs: Before it was distance. Now it is similarity.
					double normalized_similarity = 1.0 - (cost / (double)k);
					if(normalized_similarity>=threshold) {
						alignment_matrix_line[column] = normalized_similarity;
					}//else keep it zero
				}
			}
			stop = System.currentTimeMillis();
			run_times[p] = (stop-start);
			System.out.println("P="+p+"\t"+(stop-start)+"\tms");
			this.alignement_matrixes.add(alignment_matrix);
		}
		String experiment_name = "run_check_node_deletion";//default experiment, has no special name
		//if(VERBOSE)
			print_results(experiment_name, run_times);
		return run_times;
	}
	
	public double[] run_baseline(){
		if(this.solver==null) {
			System.err.println("Solver is null: Using StupidSolver");
			this.solver = new StupidSolver(k);
		}
		System.out.println("HungarianExperiment.run_baseline() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] cost_matrix = new double[k][k];
		USE_GLOBAL_MATRIX = false;
		
		double[] run_times = new double[num_paragraphs];
			
		double stop,start;
		for(int p=0;p<num_paragraphs;p++) {
			start = System.currentTimeMillis();
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = this.alignement_matrixes.get(p);//get the pre-allocated buffer. Done in Constructor
			final double[][] global_cost_matrix_buffer = fill_cost_matrix(p);
			
			//For each pair of windows
			for(int line=0;line<alignment_matrix.length;line++) {
				//get the line to get rid of 2D array resolution
				final double[] alignment_matrix_line = alignment_matrix[line];
								
				for(int column=0;column<alignment_matrix[0].length;column++) {	
					//Fill local matrix of the current window combination from global matrix
					//Note it is cost matrix with cosine distance. I.e, not similarity. 
					for(int i=0;i<this.k;i++) {
						for(int j=0;j<this.k;j++) {
							cost_matrix[i][j] = global_cost_matrix_buffer[line+i][column+j];
						}
					}
					//That's the important line
					double cost = this.solver.solve(cost_matrix, threshold);
					//normalize costs: Before it was distance. Now it is similarity.
					double normalized_similarity = 1.0 - (cost / (double)k);
					if(normalized_similarity>=threshold) {
						alignment_matrix_line[column] = normalized_similarity;
					}//else keep it zero
				}
			}
			stop = System.currentTimeMillis();
			run_times[p] = (stop-start);
			System.out.println("P="+p+"\t"+(stop-start)+"\tms");
			this.alignement_matrixes.add(alignment_matrix);
		}
		String experiment_name = "";//default experiment, has no special name
		//if(VERBOSE)
			print_results(experiment_name, run_times);
		return run_times;
	}
	
	public double[] run_baseline_deep(){
		HungarianDeep solver = new HungarianDeep(k);
		System.out.println("HungarianExperiment.run_baseline_deep() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] cost_matrix = new double[k][k];
		USE_GLOBAL_MATRIX = false;
		
		double[] run_times = new double[num_paragraphs];
			
		double stop,start;
		for(int p=0;p<num_paragraphs;p++) {
			start = System.currentTimeMillis();
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = this.alignement_matrixes.get(p);//get the pre-allocated buffer. Done in Constructor
			final double[][] paragraph_cost_matrix = fill_cost_matrix(p);
			final double[][] current_lines = new double[k][];
			
			//For each pair of windows
			for(int line=0;line<alignment_matrix.length;line++) {
				//get the line to get rid of 2D array resolution
				final double[] alignment_matrix_line = alignment_matrix[line];
								
				for(int i=0;i<k;i++) {
					current_lines[i] = paragraph_cost_matrix[line+i];
				}
				solver.set_matrix(current_lines);
				
				for(int column=0;column<alignment_matrix[0].length;column++) {	
					//That's the important line
					double cost = solver.solve(column);
					//normalize costs: Before it was distance. Now it is similarity.
					double normalized_similarity = 1.0 - (cost / (double)k);
					if(normalized_similarity>=threshold) {
						alignment_matrix_line[column] = normalized_similarity;
					}//else keep it zero
					if(SAFE_MODE) {
						fill_cost_matrix_from_paragraph_matrix(cost_matrix, paragraph_cost_matrix, line, column);
						double cost_safe = solver_baseline.solve_safe(cost_matrix, threshold);
						double[][] cost_matrix_copy = solver.get_matrix_copy(current_lines, column);
						
						if(!is_equal(cost_matrix, cost_matrix_copy)) {
							System.err.println("!is_equal(cost_matrix, cost_matrix_copy) at line="+line+" col="+column);
							out("cost_matrix",cost_matrix);
							out("cost_matrix_copy",cost_matrix_copy);
						}
						if(!is_equal(cost_safe,cost)) {
							System.err.println("cost_safe!=cost at line="+line+" col="+column);
							System.err.println("HungarianDeep Assignement\t"+cost);
							out(solver.matchJobByWorker);
							System.err.println("Baseline Assignement\t"+cost_safe);
							out(solver_baseline.matchJobByWorker);
							out("cost_matrix",cost_matrix_copy);
						}

						double[] col_minima = new double[k];
						for(int c=0;c<k;c++) {
							double cst = cost_matrix[0][c];
							col_minima[c] = cst;
						}
						for(int l=1;l<k;l++) {
							for(int c=0;c<k;c++) {
								double cst = cost_matrix[l][c];
								if(cst<col_minima[c]) {
									col_minima[c] = cst;
								}
							}
						}
						solver_baseline.solve_inj(cost_matrix, threshold, col_minima);
						if(!is_equal(solver_baseline.matchJobByWorker,solver.matchJobByWorker)) {
							System.err.println("HungarianDeep Assignement\t"+cost);
							out(solver.matchJobByWorker);
							System.err.println("Baseline Assignement\t"+cost_safe);
							out(solver_baseline.matchJobByWorker);
							out("cost_matrix",cost_matrix_copy);
							solver.solve(current_lines, column);
							solver_baseline.solve_inj(cost_matrix, threshold, col_minima);
						}
					}
				}
			}
			stop = System.currentTimeMillis();
			run_times[p] = (stop-start);
			System.out.println("P="+p+"\t"+(stop-start)+"\tms");
			this.alignement_matrixes.add(alignment_matrix);
			
			int size = size(alignment_matrix);
			double check_sum = sum(alignment_matrix);
			System.out.println("P="+p+"\t"+(stop-start)+"\tms\t"+check_sum+"\t"+size);
			this.alignement_matrixes.add(alignment_matrix);
		}

		return run_times;
	}
	
	private boolean is_equal(int[] arr_1, int[] arr_2) {
		if(arr_1==arr_2) {
			System.err.println("Same array provided");
		}
		if(arr_1.length!=arr_2.length) {
			return false;
		}
		for(int i=0;i<arr_1.length;i++) {
			if(arr_1[i]!=arr_2[i]) {
				return false;
			}
		}
		return true;
	}

	void out(int[] arr) {
		for(int d : arr) {
			System.out.print(d+"\t");
		}
		System.out.println();
	}
	
	void out(double[] arr) {
		for(double d : arr) {
			System.out.println(d+"\t");
		}
	}
	
	void fill_cost_matrix_from_paragraph_matrix(final double[][] cost_matrix, final double[][] paragraph_cost_matrix, final int line, final int column) {
		for(int i=0;i<k;i++) {
			for(int j=0;j<k;j++) {
				cost_matrix[i][j] = paragraph_cost_matrix[line+i][column+j];
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

	public double[] run_check_hungo_heuristics(){
		System.out.println("HungarianExperiment.run_check_hungo_heuristics() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] cost_matrix = new double[k][k];
		USE_GLOBAL_MATRIX = true;
		
		//Everybody gets its own solver
		HungarianKevinStern solver = new HungarianKevinStern(k);
		HungarianKevinStern solver_safe 	= new HungarianKevinStern(k);
		HungarianKevinStern solver_inverted = new HungarianKevinStern(k);
		HungarianKevinStern solver_row_only = new HungarianKevinStern(k);
		HungarianKevinStern solver_col_only = new HungarianKevinStern(k);
		HungarianKevinStern solver_inj      = new HungarianKevinStern(k);
		
		
		double[] run_times = new double[num_paragraphs];
			
		double stop,start;
		for(int p=0;p<num_paragraphs;p++) {
			start = System.currentTimeMillis();
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = this.alignement_matrixes.get(p);//get the pre-allocated buffer. Done in Constructor
			final double[][] global_cost_matrix_buffer = fill_cost_matrix(p);
			
			//For each pair of windows
			for(int line=0;line<alignment_matrix.length;line++) {
				//get the line to get rid of 2D array resolution
				final double[] alignment_matrix_line = alignment_matrix[line];
								
				for(int column=0;column<alignment_matrix[0].length;column++) {	
					//Fill local matrix of the current window combination from global matrix
					//Note it is cost matrix with cosine distance. I.e, not similarity. 
					for(int i=0;i<this.k;i++) {
						for(int j=0;j<this.k;j++) {
							cost_matrix[i][j] = global_cost_matrix_buffer[line+i][column+j];
						}
					}
					
					//That's the important line
					double cost = solver.solve(cost_matrix, threshold);
					double cost_safe = solver_safe.solve_safe(cost_matrix, threshold);
					//double cost_inverted = solver_inverted.solve_inverted_reduce(cost_matrix, threshold);
					//double cost_row_only = solver_row_only.solve_row_heuristic(cost_matrix, threshold);
					//double cost_col_only = solver_col_only.solve_column_heuristic(cost_matrix, threshold);
					
					get_column_row_sum(cost_matrix);
					double cost_inj = solver_inj.solve_inj(cost_matrix, threshold, col_minima); 
					
					if(!is_equal(cost_safe,cost)) {
						System.err.println("cost_safe!=cost:\t"+cost_safe+"\t"+cost+" l="+line+" col="+column);
					}
					/*if(!is_equal(cost_safe,cost_inverted)) {
						System.err.println("cost_safe!=cost_inverted:\t"+cost_safe+"\t"+cost_inverted+" l="+line+" col="+column);
					}
					if(!is_equal(cost_safe,cost_row_only)) {
						System.err.println("cost_safe!=cost_row_only:\t"+cost_safe+"\t"+cost_row_only+" l="+line+" col="+column);
					}
					if(!is_equal(cost_safe,cost_col_only)) {
						System.err.println("cost_safe!=cost_col_only:\t"+cost_safe+"\t"+cost_col_only+" l="+line+" col="+column);
					}*/
					if(!is_equal(cost_safe,cost_inj)) {
						System.err.println("cost_safe!=cost_inj:\t"+cost_safe+"\t"+cost_inj+" l="+line+" col="+column);
					}
					
					
					//normalize costs: Before it was distance. Now it is similarity.
					double normalized_similarity = 1.0 - (cost / (double)k);
					if(normalized_similarity>=threshold) {
						alignment_matrix_line[column] = normalized_similarity;
					}//else keep it zero
				}
			}
			stop = System.currentTimeMillis();
			run_times[p] = (stop-start);
			System.out.println("P="+p+"\t"+(stop-start)+"\tms");
			this.alignement_matrixes.add(alignment_matrix);
		}
		String experiment_name = "";//default experiment, has no special name
		//if(VERBOSE)
			print_results(experiment_name, run_times);
		return run_times;
	}
	
	void print_results(String experiment_name, double[] run_times) {
		//Print result matrixes
		
		if(TO_FILE) {
			String path = ".//results//"+System.currentTimeMillis()+"_sim()="+SIM_FUNCTION+"_k="+k+"_threshold="+threshold+"_"+this.solver.get_name()+".tsv";
			try {
				this.f = new FileWriter(path);
				output = new BufferedWriter(f);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		for(int p = 0;p<num_paragraphs;p++){
			double[][] alignment_matrix = alignement_matrixes.get(p);
			out_matrix(alignment_matrix, p);
			int num_cells = alignment_matrix.length*alignment_matrix[0].length;
			double check_sum = sum(alignment_matrix);
			System.out.println(p+"\t"+run_times[p]+"\t"+num_cells+"\t"+check_sum);
		}
		if(TO_FILE) {
			try {
				output.close();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
	}
	
	private void out_matrix(double[][] alignment_matrix, final int p_id) {
		//System.out.println("Next matrix");
		if(TO_FILE) {
			try {
				output.write("Next matrix "+p_id);
				output.newLine();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		for(double[] array : alignment_matrix) {
			String temp = null; 
			if(VERBOSE) {
				temp  = outTSV(array);
				System.out.println(temp);
			}
			if(TO_FILE) {
				if(temp==null) {temp  = outTSV(array);}
				try {
					output.write(temp);
					output.newLine();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
		}
	}

	private String outTSV(double[] array) {//TODO with String builder
		StringBuffer sb = new StringBuffer(1000);
		sb.append(array[0]);
		//String s = ""+array[0];
		for(int i=1;i<array.length;i++) {
			//s+="\t"+array[i];
			sb.append("\t");
			sb.append(array[i]);
		}
		return sb.toString();
	}
	
	static boolean USE_GLOBAL_MATRIX = false;
	private double[][] fill_cost_matrix(final int paragraph) {
		final int[] raw_paragraph_1 = this.raw_paragraphs_b1.get(paragraph);
		final int[] raw_paragraph_2 = this.raw_paragraphs_b2.get(paragraph);
		final double[][] global_cost_matrix = new double[raw_paragraph_1.length][raw_paragraph_2.length];
		
		if(USE_GLOBAL_MATRIX) {
			if(dense_global_matrix_buffer==null) {//XXX Has extra runtime measurement inside method
				create_dense_matrix();
			}
			for(int line=0;line<raw_paragraph_1.length;line++) {
				final int set_id_window_p1 = raw_paragraph_1[line];
				final double[] cost_matrix_line = dense_global_matrix_buffer[set_id_window_p1];
				for(int column=0;column<raw_paragraph_2.length;column++) {
					final int set_id_window_p2 = raw_paragraph_2[column];	
					final double dist = cost_matrix_line[set_id_window_p2];
					global_cost_matrix[line][column] = dist;
				}
			}
		}else{
			for(int line=0;line<raw_paragraph_1.length;line++) {
				final int set_id_window_p1 = raw_paragraph_1[line];
				final double[] vec_1 = this.embedding_vector_index.get(set_id_window_p1);
				for(int column=0;column<raw_paragraph_2.length;column++) {
					final int set_id_window_p2 = raw_paragraph_2[column];
					final double[] vec_2 = this.embedding_vector_index.get(set_id_window_p2);
					final double dist = dist(set_id_window_p1,set_id_window_p2,vec_1,vec_2);
					global_cost_matrix[line][column] = dist;
				}
			}
		}
		
		return global_cost_matrix;
	}

	/*private void fill_cost_matrix(final int[] k_window_p1, final int[] k_window_p2, final double[][] cost_matrix) {
		for(int line=0;line<this.k;line++) {
			final int set_id_window_p1 = k_window_p1[line];
			final double[] vec_1 = this.embedding_vector_index.get(set_id_window_p1);
			for(int column=0;column<this.k;column++) {
				final int set_id_window_p2 = k_window_p2[column];
				final double[] vec_2 = this.embedding_vector_index.get(set_id_window_p2);
				final double dist = dist(set_id_window_p1,set_id_window_p2,vec_1,vec_2);
				cost_matrix[line][column] = dist;
			}
		}
	}*/
	
	public static final double EQUAL 	= 0;
	public static final double MAX_DIST = 1.0;
	
	public static final int COSINE 			= 0;
	public static final int STRING_EDIT 	= 1;
	public static final int VANILLA_OVERLAP = 3;
	public static final int SIM_FUNCTION = COSINE;
	
	@SuppressWarnings("unused")
	public static double dist(final int set_id1, final int set_id2, final double[] vec_1, final double[] vec_2) {
		if(set_id1==set_id2) {
			return EQUAL;
		}else if(vec_1==null || vec_2==null){//may happen e.g., for stop words
			return MAX_DIST;
		}
		if(SIM_FUNCTION == COSINE) {
			return cosine_distance(vec_1, vec_2);	
		}else if(SIM_FUNCTION == STRING_EDIT){
			return edit_dist(set_id1, set_id2);
		}else if(SIM_FUNCTION == VANILLA_OVERLAP){
			return MAX_DIST;//only EQUAL or MAX_DIST
		}else{
			System.err.println("dist(): Unknown sim fnction");
			return cosine_distance(vec_1, vec_2);
		}
	}
	
	//TODO make Strings available
	static int edit_dist(final int set_id1, final int set_id2){
		System.err.println("edit_dist() not yet implemented");
		String s1=null;
		String s2=null;
		return edit_dist(s1, s2, s1.length(), s2.length());
	}
	
    static int edit_dist(String s1, String s2, int pos_s1, int pos_s2){
    	//End of recursion if one Token sequence has been entirely consumed
    	if (pos_s1 == 0) {
    		 return pos_s2;
    	}
        if(pos_s2 == 0) {
            return pos_s1;
        }
 
        //Here both sequences are identical. Move both positions to the token before.
        if (s1.charAt(pos_s1 - 1)==s2.charAt(pos_s2 - 1)) {
            return edit_dist(s1, s2, pos_s1 - 1, pos_s2 - 1);
        }
 
        return 1
            + min(edit_dist(s1, s2, pos_s1, pos_s2 - 1)  		// Insert 
            		, edit_dist(s1, s2, pos_s1 - 1, pos_s2)		// Remove
            		, edit_dist(s1, s2, pos_s1 - 1,pos_s2 - 1)	// Replace
              );
    }
    
	static final int min(final int x, final int y, final int z){
        if (x <= y && x <= z)
            return x;
        if (y <= x && y <= z)
            return y;
        else
            return z;
    }

	static double cosine_distance(final double[] vectorA, final double[] vectorB) {
		if(MatchesWithEmbeddings.NORMALIZE_VECTORS) {
			double dotProduct = 0.0;
		    for (int i = 0; i < vectorA.length; i++) {
		        dotProduct += vectorA[i] * vectorB[i];
		    }
		    double dist = 1-dotProduct;
		    dist = (dist < 0) ? 0 : dist;
		    dist = (dist > 1) ? 1 : dist;
		    return dist;
		}else{
			double dotProduct = 0.0;
		    double normA = 0.0;
		    double normB = 0.0;
		    for (int i = 0; i < vectorA.length; i++) {
		        dotProduct += vectorA[i] * vectorB[i];
		        normA += Math.pow(vectorA[i], 2);
		        normB += Math.pow(vectorB[i], 2);
		    }
		    double dist = 1-(dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)));
		    dist = (dist < 0) ? 0 : dist;
		    dist = (dist > 1) ? 1 : dist;
		    return dist;
		}
	    
	}
	
	public double[] run_candidates_min_matrix_3() {
		if(k_with_windows_b1.size()!=1) {
			System.err.println("Expecting book granularity");
		}
		//Check config (1) Normalized vectors to unit length
		if(!MatchesWithEmbeddings.NORMALIZE_VECTORS) {
			System.err.println("run_candidates_min_matrix_3(): MatchesWithEmbeddings.NORMALIZE_VECTORS=false");
		}
		//Ensure config (2) Hungarian implementation from Kevin Stern
		this.solver = new HungarianKevinStern(k);
		
		// (3) Dense global cost matrix - compute once
		if(dense_global_matrix_buffer==null) {//XXX Has extra runtime measurement inside method
			create_dense_matrix();
		}
		
		System.out.println("HungarianExperiment.run_candidates_min_matrix() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] cost_matrix = new double[k][k];
		
		double[] run_times = new double[num_paragraphs];
		
		/**
		 * inverted_index.get(i)[my_token_id] -> ordered list of token_id with sim(my_token_id, token_id) >= threshold 
		 */
		final ArrayList<int[]> neighborhood_index = create_neihborhood_index(dense_global_matrix_buffer);
		/**
		 * inverted_window_index.get(my_token_id).get(paragraph_id) -> ordered list of cells containing some other token, s.t.  sim(my_token_id, token_id) >= threshold. I.e., this is a candidate. 
		 */
		final ArrayList<ArrayList<int[]>> inverted_window_index = create_inverted_window_index(this.k_with_windows_b2, neighborhood_index);
		/**
		 * inverted_token_index.get(paragraph_id).get(my_token_id) -> all occurrences of my_token_id in this paragraph
		 */
		final HashMap<Integer, ArrayList<Integer>> inverted_token_index = create_inverted_token_index_book_granularity(this.k_with_windows_b1);
		
		double stop,start;
		
		//Allocate space for the alignment matrix
		final double[][] alignment_matrix = this.alignement_matrixes.get(0);//get the pre-allocated buffer. Done in Constructor
		final int[][] k_windows_p1 = this.k_with_windows_b1.get(0);
		final int[][] k_windows_p2 = this.k_with_windows_b2.get(0);	
		final boolean[][] candidates = new boolean[alignment_matrix.length][alignment_matrix[0].length];
		
		int count_survived_pruning = 0;
		int count_cells_exceeding_threshold = 0;
		
		start = System.currentTimeMillis();
		
		//(1) Determine candidates
		double start_candidates = System.currentTimeMillis();
		for(Entry<Integer, ArrayList<Integer>> e : inverted_token_index.entrySet()) {
			/** This is the token itself */
			final int token_id = e.getKey();
			/** This contains all windows of book_1 where token_id is in, i.e., this refers to lines */
			final ArrayList<Integer> token_id_occurences = e.getValue();
			/** This contains all windows with token_id with sim(token_id, token_id) >= threshold, i.e., this refers columns*/  
			final int[] index = inverted_window_index.get(token_id).get(0);
			
			for(int line : token_id_occurences) {
				final boolean[] candidates_line = candidates[line];
				for(int pos : index) {
					candidates_line[pos] = true;
				}
			}
		}
		double stop_candidates = System.currentTimeMillis();
					
		
		//(2) Validate candidates
		for(int line=0;line<alignment_matrix.length;line++) {	
			final double[] alignment_matrix_line = alignment_matrix[line];
			final boolean[] candidates_line = candidates[line];
			
			for(int column=0;column<alignment_matrix[0].length;column++) {
				boolean is_candidate = candidates_line[column];
				if(is_candidate){
					count_survived_pruning++;
					//get local cost matrix
					for(int i=0;i<this.k;i++) {
						final int set_id_window_p1 = k_windows_p1[line][i];
						for(int j=0;j<this.k;j++) {
							final int set_id_window_p2 = k_windows_p2[column][j];
							double dist = dense_global_matrix_buffer[set_id_window_p1][set_id_window_p2];
							cost_matrix[i][j] = dist;
						}
					}
					
					// (4) compute the bound
					final double lb_cost = get_column_row_sum(cost_matrix);
					final double up_normalized_similarity = 1.0 - (lb_cost / (double)k);
					
					if(up_normalized_similarity+DOUBLE_PRECISION_BOUND>this.threshold) {
						//That's the important line
						double cost = this.solver.solve(cost_matrix, threshold);
						//normalize costs: Before it was distance. Now it is similarity.
						double normalized_similarity = 1.0 - (cost / (double)k);
						if(normalized_similarity>=threshold) {
							count_cells_exceeding_threshold++;
							alignment_matrix_line[column] = normalized_similarity;
						}//else keep it zero
					}
				}//else safe mode
				if(SAFE_MODE) {
					safe_mode_run_candidates(k_windows_p1, k_windows_p2, line, column, cost_matrix ,is_candidate);
				}
			}
			
		}
		stop = System.currentTimeMillis();
		run_times[0] = stop-start;
		
		int size = size(alignment_matrix);
		double check_sum = sum(alignment_matrix);
		System.out.println("P=0"+"\t"+(stop-start)+"\tms\t"+check_sum+"\t"+size+"\t"+count_survived_pruning+"\t"+count_cells_exceeding_threshold+"\t"+(stop_candidates-start_candidates));
		
		return run_times;
	}
	
	private HashMap<Integer,ArrayList<Integer>> create_inverted_token_index_book_granularity(ArrayList<int[][]> k_with_windows) {
		System.out.println("create_inverted_token_index() START");
		if(k_with_windows.size()!=1) {
			System.err.println("Expecting book granularity");
		}
		final double start = System.currentTimeMillis();
		HashMap<Integer,ArrayList<Integer>> indexes = new HashMap<Integer,ArrayList<Integer>>(max_id+1);
				
		final int[][] windows_in_paragraph = k_with_windows.get(0);
			
		for(int window_id=0;window_id<windows_in_paragraph.length;window_id++) {
			final int[] window = windows_in_paragraph[window_id];
			for(int token_id : window) {
				ArrayList<Integer> index = indexes.get(token_id);
				if(index==null) {
					index = new ArrayList<Integer>();	
					indexes.put(token_id, index);
				}
				index.add(window_id);
			}
		}
		
		System.out.println("create_inverted_token_index() STOP "+(System.currentTimeMillis()-start));
		return indexes;
	}

	private int[] to_primitive_array(final ArrayList<Integer> list) {
		final int[] arr = new int[list.size()];
		for(int i=0;i<list.size();i++) {
			arr[i] = list.get(i);
		}
		return arr;
	}

	public double[] run_candidates_min_matrix() {
		//Check config (1) Normalized vectors to unit length
		if(!MatchesWithEmbeddings.NORMALIZE_VECTORS) {
			System.err.println("run_solution(): MatchesWithEmbeddings.NORMALIZE_VECTORS=false");
		}
		//Ensure config (2) Hungarian implementation from Kevin Stern
		this.solver = new HungarianKevinStern(k);
		
		// (3) Dense global cost matrix - compute once
		if(dense_global_matrix_buffer==null) {//XXX Has extra runtime measurement inside method
			create_dense_matrix();
		}
		
		System.out.println("HungarianExperiment.run_candidates_min_matrix() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] cost_matrix = new double[k][k];
		
		double[] run_times = new double[num_paragraphs];
		
		/**
		 * inverted_index.get(i)[my_token_id] -> ordered list of token_id with sim(my_token_id, token_id) >= threshold 
		 */
		final ArrayList<int[]> neighborhood_index = create_neihborhood_index(dense_global_matrix_buffer);
		/**
		 * inverted_window_index.get(my_token_id).get(paragraph_id) -> ordered list of cells containing some other token, s.t.  sim(my_token_id, token_id) >= threshold. I.e., this is a candidate. 
		 */
		final ArrayList<ArrayList<int[]>> inverted_window_index = create_inverted_window_index(this.k_with_windows_b2, neighborhood_index);
		
		double stop,start;
		for(int p=0;p<num_paragraphs;p++) {
			start = System.currentTimeMillis();
			
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = this.alignement_matrixes.get(p);//get the pre-allocated buffer. Done in Constructor
			final int[][] k_windows_p1 = this.k_with_windows_b1.get(p);
			final int[][] k_windows_p2 = this.k_with_windows_b2.get(p);	
						
			final boolean[] candidates = new boolean[alignment_matrix[0].length];
			
			for(int line=0;line<alignment_matrix.length;line++) {
				Arrays.fill(candidates, false);//no candidate so far
				//get the line to get rid of 2D array resolution
				final double[] alignment_matrix_line = alignment_matrix[line];
				
				//Create candidates: the window of p2 is fixed
				final int[] window_p1 = k_windows_p1[line];
				for(int id : window_p1) {
					final ArrayList<int[]> token_index = inverted_window_index.get(id);
					final int[] index = token_index.get(p);
					for(int pos : index) {
						candidates[pos] = true;
					}
				}
						
				//Validate candidates
				for(int column=0;column<candidates.length;column++) {
					boolean is_candidate = candidates[column];
					if(is_candidate){
						//get local cost matrix
						for(int i=0;i<this.k;i++) {
							final int set_id_window_p1 = k_windows_p1[line][i];
							for(int j=0;j<this.k;j++) {
								final int set_id_window_p2 = k_windows_p2[column][j];
								double dist = dense_global_matrix_buffer[set_id_window_p1][set_id_window_p2];
								cost_matrix[i][j] = dist;
							}
						}
						
						// (4) compute the bound
						final double lb_cost = get_column_row_sum(cost_matrix);
						final double up_normalized_similarity = 1.0 - (lb_cost / (double)k);
						if(up_normalized_similarity+DOUBLE_PRECISION_BOUND>this.threshold) {
							//That's the important line
							double cost = this.solver.solve(cost_matrix, threshold);
							//normalize costs: Before it was distance. Now it is similarity.
							double normalized_similarity = 1.0 - (cost / (double)k);
							if(normalized_similarity>=threshold) {
								alignment_matrix_line[column] = normalized_similarity;
							}//else keep it zero
						}
					}//else safe mode
					if(SAFE_MODE) {
						safe_mode_run_candidates(k_windows_p1, k_windows_p2, line, column, cost_matrix ,is_candidate);
					}
				}
				
			}
			stop = System.currentTimeMillis();
			run_times[p] = stop-start;
		}
		
		
		String experiment_name = "Candidates min(local_matrix)";//default experiment, has no special name
		//if(VERBOSE)
			print_results(experiment_name, run_times);
		
		return run_times;
	}
	
	public double[] run_candidates_min_matrix_2() {
		//XXX Assume book granularity for now
		//Check config (1) Normalized vectors to unit length
		if(!MatchesWithEmbeddings.NORMALIZE_VECTORS) {
			System.err.println("run_solution(): MatchesWithEmbeddings.NORMALIZE_VECTORS=false");
		}
		//Ensure config (2) Hungarian implementation from Kevin Stern
		this.solver = new HungarianKevinStern(k);
		
		// (3) Dense global cost matrix - compute once
		if(dense_global_matrix_buffer==null) {//XXX Has extra runtime measurement inside method
			create_dense_matrix();
		}
		
		System.out.println("HungarianExperiment.run_candidates_min_matrix() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold+" "+solver.get_name());
		final double[][] cost_matrix = new double[k][k];
		
		double[] run_times = new double[num_paragraphs];
		
		/**
		 * inverted_index.get(i) -> index for paragraph token_id
		 * inverted_index.get(i)[token_id] -> some other_token_id with sim(token_id, other_token_id) > threshold 
		 */
		final ArrayList<int[]> neighborhood_index = create_neihborhood_index(dense_global_matrix_buffer);
		final ArrayList<ArrayList<int[]>> inverted_window_index = create_inverted_window_index(this.k_with_windows_b2, neighborhood_index);
		
		double stop,start;
		for(int p=0;p<num_paragraphs;p++) {
			start = System.currentTimeMillis();
			
			//Allocate space for the alignment matrix
			final double[][] alignment_matrix = this.alignement_matrixes.get(p);//get the pre-allocated buffer. Done in Constructor
			final int[][] k_windows_p1 = this.k_with_windows_b1.get(p);
			final int[][] k_windows_p2 = this.k_with_windows_b2.get(p);	
								
			for(int line=0;line<alignment_matrix.length;line++) {
				//get the line to get rid of 2D array resolution
				final double[] alignment_matrix_line = alignment_matrix[line];
				final boolean[] already_checked = new boolean[alignment_matrix[0].length];
				
				//Create candidates: the window of p2 is fixed
				final int[] window_p1 = k_windows_p1[line];
				for(int id : window_p1) {
					final ArrayList<int[]> token_index = inverted_window_index.get(id);
					final int[] index = token_index.get(p);
					for(int column : index) {
						if(!already_checked[column]) {//hopefully not computed before
							already_checked[column] = true;
							//get local cost matrix
							for(int i=0;i<this.k;i++) {
								final int set_id_window_p1 = k_windows_p1[line][i];
								for(int j=0;j<this.k;j++) {
									final int set_id_window_p2 = k_windows_p2[column][j];
									double dist = dense_global_matrix_buffer[set_id_window_p1][set_id_window_p2];
									cost_matrix[i][j] = dist;
								}
							}
							// (4) compute the bound
							final double lb_cost = get_column_row_sum(cost_matrix);
							final double up_normalized_similarity = 1.0 - (lb_cost / (double)k);
							if(up_normalized_similarity+DOUBLE_PRECISION_BOUND>this.threshold) {
								//That's the important line
								double cost = this.solver.solve(cost_matrix, threshold);
								//normalize costs: Before it was distance. Now it is similarity.
								double normalized_similarity = 1.0 - (cost / (double)k);
								if(normalized_similarity>=threshold) {
									alignment_matrix_line[column] = normalized_similarity;
								}//else keep it zero
							}
						}
					}
				}
			}
			stop = System.currentTimeMillis();
			run_times[p] = stop-start;
		}
		
		
		String experiment_name = "Candidates min(local_matrix)";//default experiment, has no special name
		//if(VERBOSE)
			print_results(experiment_name, run_times);
		
		return run_times;
	}
	
	void safe_mode_run_candidates(int[][] k_windows_p1, int[][] k_windows_p2, int line, int column, double[][] cost_matrix, boolean is_candidate) {
		//get local cost matrix
		for(int i=0;i<this.k;i++) {
			final int set_id_window_p1 = k_windows_p1[line][i];
			for(int j=0;j<this.k;j++) {
				final int set_id_window_p2 = k_windows_p2[column][j];
				double dist = dense_global_matrix_buffer[set_id_window_p1][set_id_window_p2];
				cost_matrix[i][j] = dist;
			}
		}
		double cost_safe = solver.solve(cost_matrix);
		double sim_safe = 1-cost_safe;
		if(!is_candidate && sim_safe>=threshold) {
			System.err.println("!is_candidate && sim_safe>=threshold");
		}
		if(sim_safe>=threshold) {
			if(!is_candidate) {
				System.err.println("sim_safe>=threshold, but !is_candidate at w_1="+Arrays.toString(k_windows_p1[line])+" and w_2="+Arrays.toString(k_windows_p2[column])+"(line="+line+",column="+column+")");
			}
		}
		/*if(sim_safe<threshold) {
			if(is_candidate) {
				System.err.println("Loose bound");
			}
		}*/
	}

	/**
	 * indexes.get(token_id)[paragraph]-> int[] of position with token having sim(token_id, some token) > threshold
	 * @param k_with_windows_b12
	 * @return
	 */
	private ArrayList<ArrayList<int[]>> create_inverted_window_index(final ArrayList<int[][]> k_with_windows, ArrayList<int[]> neihborhood_indexes) {
		System.out.println("ArrayList<ArrayList<int[]>> create_inverted_window_index() BEGIN");
		double start = System.currentTimeMillis();
		ArrayList<ArrayList<int[]>> indexes = new ArrayList<ArrayList<int[]>>();
		//For each token
		for(int token_id = 0;token_id<neihborhood_indexes.size();token_id++) {
			//Create the list of occurrences for token: token_id
			final int[] neihborhood_index = neihborhood_indexes.get(token_id);
			ArrayList<int[]> occurences_per_paragraph = new ArrayList<int[]>(k_with_windows.size());
			
			//For each windowed paragraph: Inspect whether one of the tokens in neihborhood_indexes is in the windows
			for(int paragraph=0;paragraph<k_with_windows.size();paragraph++) {
				int[][] windows = k_with_windows.get(paragraph);
				ArrayList<Integer> index_this_paragraph = new ArrayList<Integer>();
				for(int pos=0;pos<windows.length;pos++) {
					int[] curr_window = windows[pos];
					if(is_in(neihborhood_index, curr_window)) {
						index_this_paragraph.add(pos);
					}
				}
				int[] temp = new int[index_this_paragraph.size()];
				for(int i=0;i<index_this_paragraph.size();i++) {
					temp[i] = index_this_paragraph.get(i);
				}
				occurences_per_paragraph.add(temp);
			}
			indexes.add(occurences_per_paragraph);
		}
		System.out.println("ArrayList<ArrayList<int[]>> create_inverted_window_index() END in\t"+(System.currentTimeMillis()-start));
		return indexes;
	}

	//TODO exploit running window property: Maybe order ids by frequency
	private boolean is_in(int[] neihborhood_index, int[] curr_window) {
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
	 * inverted_index.get(i) -> index for paragraph token_id
	 * inverted_index.get(i)[token_id] -> some other_token_id with sim(token_id, other_token_id) > threshold 
	 */
	private ArrayList<int[]> create_neihborhood_index(final double[][] matrix) {
		System.out.println("create_neihborhood_index() BEGIN");
		double start = System.currentTimeMillis();
		
		ArrayList<int[]> indexes = new ArrayList<int[]>(matrix.length);
		for(final double[] line : matrix) {
			ArrayList<Integer> index = new ArrayList<Integer>(line.length);//TODO remove double effort?
			for(int id=0;id<line.length;id++) {
				final double dist = line[id];
				final double sim = 1 - dist;
				if(sim>=threshold){
					index.add(id);
				}
			}
			int[] index_arr = new int[index.size()];
			for(int i=0;i<index_arr.length;i++) {
				index_arr[i] = index.get(i);
			}
			indexes.add(index_arr);
		}
		System.out.println("create_neihborhood_index() END in\t"+(System.currentTimeMillis()-start));
		if(LOGGING_MODE) {
			System.out.println("Matrix size: "+matrix.length);
			ArrayList<Integer> counts = new ArrayList<Integer>(matrix.length);
			for(int[] index : indexes) {
				counts.add(index.length);
				//System.out.println(index.length);
			}
			Collections.sort(counts);
			for(int count : counts) {
				System.out.println(count);
			}
		}
		return indexes;
	}

	static void out(ArrayList<int[]> set_ids, ArrayList<int[][]> k_with_window_set_ids) {
		for(int paragraph=0;paragraph<set_ids.size();paragraph++) {
			int[] 	set_ids_paragraph = set_ids.get(paragraph);
			int[][] set_ids_windowed_paragraph = k_with_window_set_ids.get(paragraph);
			System.out.println(Arrays.toString(set_ids_paragraph));
			for(int[] window : set_ids_windowed_paragraph){
				System.out.print(Arrays.toString(window)+"\t");	
			}
			System.out.println();
		}
	}
}
