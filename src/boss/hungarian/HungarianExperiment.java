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
import boss.util.Histogram;

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
	
	final double[] k_buffer;
	
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
		
		this.k_buffer = new double[k];
		
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
	public void run_idea_nikolaus() {
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
	
	private double get_column_row_sum(final double[][] cost_matrix) {
		double row_sum = 0;
		Arrays.fill(this.k_buffer, Double.MAX_VALUE);
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
	public void run_baseline_global_matrix_dense(){
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
	public void run_baseline_global_matrix_sparse(){
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
		this.dense_global_matrix_buffer = new double[max_id+1][max_id+1];//This is big....
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
		System.out.println("create_dense_matrix()\t"+(stop-start));
	}

	private int get_key(int set_id_window_p1, int set_id_window_p2) {
		if(set_id_window_p1>set_id_window_p2) {//exploit symmetry
			return set_id_window_p1 |= (set_id_window_p2 << 16);
		}else{
			return set_id_window_p2 |= (set_id_window_p1 << 16);
		}
	}

	
	public void test_hungarian_implementations(){
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
						//That's the important line
						double cost = this.solver.solve(cost_matrix, threshold);
						//normalize costs: Before it was distance. Now it is similarity.
						double normalized_similarity = 1.0 - (cost / (double)k);
						if(normalized_similarity>=threshold) {
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
			System.out.println("P="+p+"\t"+(stop-start)+"\tms");
			this.alignement_matrixes.add(alignment_matrix);
		}
		String experiment_name = "";//default experiment, has no special name
		if(VERBOSE)
			print_results(experiment_name, run_times);
		return run_times;
	}
	
	public void run_baseline_zick_zack(){
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
		//final double[] min_cost_lines  = new double[k];
		final double[] min_cost_columns= new double[k];
		double sum_min_lines, sum_min_columns, lb_cost;
		
			
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
					
					//Compute the bounds: In the fist cell we have to do it entirely
					/*get_column_row_minima(cost_matrix, min_cost_lines, min_cost_columns);
					sum_min_lines   = sum(min_cost_lines);
					sum_min_columns = sum(min_cost_columns);
					lb_cost         = Math.max(sum_min_columns, sum_min_lines);*/
					
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
						double cost = solver.solve(cost_matrix, threshold, k_buffer, true);
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
					//double old_min_cost_col = min_cost_columns[replace_position];
					min_cost_columns[replace_position] = Double.MAX_VALUE;
					for(int i=0;i<this.k;i++) {
						final int set_id_window_p1 = k_windows_p1[line][i];
						final int set_id_window_p2 = k_windows_p2[column][k-1];//Always the new one
						double cost_l_c = dense_global_matrix_buffer[set_id_window_p1][set_id_window_p2];
						//double old_cost_l_c = cost_matrix[i][replace_position];
						cost_matrix[i][replace_position] = cost_l_c;
						
						/*
						//Step 1: Line costs
						if(cost_l_c<min_cost_lines[i]) {//becomes the new line minimum
							min_cost_lines[i] = cost_l_c;
							sum_min_lines-=old_cost_l_c;
							sum_min_lines+=cost_l_c;
						}else if(old_cost_l_c==min_cost_lines[i]) {//the old cost was the minimum
							//re-compute the minimum
							//TODO
						}else{//the new cost are not smaller than the minimum and the old cost was not the minimum
							//Best case nothing todo
						}
						//Update column costs	
						if(min_cost_columns[replace_position]>cost_l_c){
							min_cost_columns[replace_position] = cost_l_c;
						}*/
					}
					//sum_min_columns-=old_min_cost_col;
					//sum_min_columns+=min_cost_columns[replace_position];
					
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

						double cost_copy = this.solver.solve(cost_matrix_copy, threshold);
						double cost = solver.solve(cost_matrix, threshold);
						if(!is_equal(cost, cost_copy)) {
							System.err.println("cost!=cost_copy "+cost+" "+cost_copy);
						}
					}
					
					lb_cost = get_column_row_sum(cost_matrix);
					final double up_normalized_similarity = 1.0 - (lb_cost / (double)k);
					if(up_normalized_similarity+DOUBLE_PRECISION_BOUND>this.threshold) {
						//That's the important line
						double cost = solver.solve(cost_matrix, threshold, k_buffer, true);//FIXME must be false, but algo does not work correctly with this.
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

	public double[] run_baseline(){
		if(this.solver==null) {
			System.err.println("Solver is null: Using StupidSolver");
			this.solver = new StupidSolver(k);
		}
		System.out.println("HungarianExperiment.run_baseline() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold+" "+solver.get_name());
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
	
	private double[][] fill_cost_matrix(final int paragraph) {
		final int[] raw_paragraph_1 = this.raw_paragraphs_b1.get(paragraph);
		final int[] raw_paragraph_2 = this.raw_paragraphs_b2.get(paragraph);
		final double[][] global_cost_matrix = new double[raw_paragraph_1.length][raw_paragraph_2.length];
		
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
	public
	final static double dist(final int set_id1, final int set_id2, final double[] vec_1, final double[] vec_2) {
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
		    return 1-(dotProduct);
		}else{
			double dotProduct = 0.0;
		    double normA = 0.0;
		    double normB = 0.0;
		    for (int i = 0; i < vectorA.length; i++) {
		        dotProduct += vectorA[i] * vectorB[i];
		        normA += Math.pow(vectorA[i], 2);
		        normB += Math.pow(vectorB[i], 2);
		    }   
		    return 1-(dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)));
		}
	    
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
