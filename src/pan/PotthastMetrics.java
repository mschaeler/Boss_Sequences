package pan;

import java.util.ArrayList;
import java.util.List;

import boss.test.SemanticTest;

/**
 * Based on An Evaluation Framework for Plagiarism Detection https://aclanthology.org/C10-2115.pdf
 * Note, we assume that matrices used here refer to <bold>r</bold> and <bold>s</bold> (Page 3 Lines 18 ff.), i.e., not to <emph>r</emph> and <emph>s</emph> (Page 3 Lines 18 ff.) 
 */
public class PotthastMetrics {
	static boolean VERBOSE = false;
	
	static final double IS_CORE_CELL = 1.0d;
	static final double IS_REACHABLE_CELL = 0.5d;
	
	public static double CORE_CELL_THRESHOLD = 0.9d;
	public static double REACHABILITY_CELL_THRESHOLD = 0.5d;
	
	static double[][] mark_cluster_seeds(double[][] matrix, final int k){
		return mark_cluster_seeds(matrix, k, CORE_CELL_THRESHOLD);
	}
	
	static double[][] mark_cluster_seeds(double[][] matrix, final int k, final double core_threshold){
		final int num_lines = matrix.length;
		final int num_columns = matrix[0].length;
		double[][] cluster_seeds = new double[num_lines][num_columns];
		
		for(int line=0;line<num_lines;line++) {
			for(int column=0;column<num_columns;column++) {
				boolean b = is_core_cell(matrix, line, column, core_threshold);
				if(b) {
					cluster_seeds[line][column] = IS_CORE_CELL; 
				}
			}
		}
		return cluster_seeds;
	}
	//is core cell
	//TODO initial guess
	private static boolean is_core_cell(double[][] matrix, int line, int column, final double threshold) {
		double cell_similarity = matrix[line][column];
		if(cell_similarity>threshold) {
			return true;
		}else{
			return false;
		}
	}
	static double[][] expand_cluster_seeds(double[][] matrix, double[][] cluster_seeds){
		final int num_lines = matrix.length;
		final int num_columns = matrix[0].length;
		for(int line=0;line<num_lines;line++) {
			for(int column=0;column<num_columns;column++) {
				if(cluster_seeds[line][column]==IS_CORE_CELL) {
					expand_cluster_seed_cell(matrix, cluster_seeds, line, column);	
				}
			}
		}
		return cluster_seeds;
	}
	static void expand_cluster_seed_cell(final double[][] matrix, final double[][] cluster_seeds, final int line, final int column) {
		final int num_lines = matrix.length;
		final int num_columns = matrix[0].length;
		
		for(int l=Math.max(line-1, 0);l<Math.min(line+1,num_lines-1);l++) {
			for(int c=Math.max(column-1, 0);c<Math.min(column+1,num_columns-1);c++) {
				double similarity = matrix[l][c];
				double current_label = cluster_seeds[l][c];
				if(similarity>=REACHABILITY_CELL_THRESHOLD && current_label!=IS_CORE_CELL && current_label!=IS_REACHABLE_CELL) {
					cluster_seeds[l][c] = IS_REACHABLE_CELL;
					expand_cluster_seed_cell(matrix, cluster_seeds, l, c);
				}
			}
		}
	}
	

	static final double[] core_thresholds = {1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1};
	static ArrayList<Double>[][] recall_results = null;
	static ArrayList<Double>[][] gran_results = null;
	public static void run() {
		/**
		 * all_matrices.(pair_id).(k-3)
		 */
		ArrayList<ArrayList<double[][]>> all_matrices;
		all_matrices = MatrixLoader.load_all_excerpt_matrices();
			
		recall_results = new ArrayList[all_matrices.get(0).size()][core_thresholds.length];
		gran_results = new ArrayList[all_matrices.get(0).size()][core_thresholds.length];
		for(int line=0;line<recall_results.length;line++) {
			for(int column=0;column<recall_results[line].length;column++) {
				recall_results[line][column] = new ArrayList<Double>();
				gran_results[line][column] = new ArrayList<Double>();
			}
		}
		
		for(int pair_id=0;pair_id<all_matrices.size();pair_id++) {
			run_single_pair(all_matrices.get(pair_id));
		}
		System.out.println("Average Recall");
		System.out.println();

		System.out.print("k\t");
		for(double d : core_thresholds) {
			System.out.print("recall "+d+"\t");
		}
		System.out.println();
		for(int k_minus_3=0;k_minus_3<recall_results.length;k_minus_3++) {
			ArrayList<Double>[] line = recall_results[k_minus_3];
			System.out.print(k_minus_3+3+"\t");
			for(int i=0;i<line.length;i++) {
				System.out.print(avg(line[i])+"\t");	
			}
			System.out.println();
		}
		
		System.out.println("Average Granularity");
		System.out.println();

		System.out.print("k\t");
		for(double d : core_thresholds) {
			System.out.print("gran "+d+"\t");
		}
		System.out.println();
		for(int k_minus_3=0;k_minus_3<gran_results.length;k_minus_3++) {
			ArrayList<Double>[] line = gran_results[k_minus_3];
			System.out.print(k_minus_3+3+"\t");
			for(int i=0;i<line.length;i++) {
				System.out.print(avg(line[i])+"\t");	
			}
			System.out.println();
		}
	}
	
	private static double avg(ArrayList<Double> arrayList) {
		double sum = 0.0d;
		for(double d : arrayList) {
			sum += d;
		}
		return sum/(double)arrayList.size();
	}

	/**
	 * Collects all results.
	 */
	static ArrayList<Double>[][] precsion_results;
	static double[][] precsion_results_true_positives = null;
	static double[][] precsion_results_all_positves = null;
	
	public static void run_full_documents() {		
		/**
		 * all_matrices.(pair_id).(k-3)
		 */
		List<String> l = MatrixLoader.get_all_excerpt_directories();
		
		for(int i=0;i<l.size();i++) {
			String dir = l.get(i);
			
			ArrayList<double[][]> my_matrices = MatrixLoader.load_all_matrices_of_pair(MatrixLoader.get_org_document_dir(dir));
			ArrayList<double[][]> my_excerpt_matrices = MatrixLoader.load_all_matrices_of_pair(dir);
			if(precsion_results_true_positives==null) {
				precsion_results_true_positives = new double[my_matrices.size()][core_thresholds.length];
			}
			if(precsion_results_all_positves==null) {
				precsion_results_all_positves = new double[my_matrices.size()][core_thresholds.length];
			}
			if(precsion_results==null) {
				precsion_results = new ArrayList[my_matrices.size()][core_thresholds.length];
				for(int line=0;line<precsion_results.length;line++) {
					for(int column=0;column<precsion_results[line].length;column++) {
						precsion_results[line][column] = new ArrayList<Double>();
					}
				}
			}
			System.out.println("Computing precison for "+MatrixLoader.get_org_document_dir(dir)+" and "+dir);
			run_single_pair_for_precision(my_matrices, my_excerpt_matrices);
		}
		System.out.println("True positives");
		System.out.println("k\tprecision .9\tprecision .8\tprecision .7\tprecision .6");
		for(double[] line : precsion_results_true_positives) {
			System.out.println("k\t"+SemanticTest.outTSV(line));
		}
		
		System.out.println("All positives");
		System.out.println("k\tprecision .9\tprecision .8\tprecision .7\tprecision .6");
		for(double[] line : precsion_results_all_positves) {
			System.out.println("k\t"+SemanticTest.outTSV(line));
		}
		System.out.println("Precision");
		System.out.println("k\tprecision .9\tprecision .8\tprecision .7\tprecision .6");
		for(int line=0;line<precsion_results_all_positves.length;line++) {
			System.out.print(line+3+"\t");
			for(int colum=0;colum<precsion_results_all_positves[0].length;colum++) {
				System.out.print(precsion_results_true_positives[line][colum] / precsion_results_all_positves[line][colum] +"\t");  
			}
			System.out.println();
		}
		System.out.println("Response-time only cluster extraction");
		System.out.println("k\tt .9\tt .8\tt .7\tt .6");
		for(int line=0;line<run_time_plagiarism_extraction.length;line++) {
			System.out.print(line+3+"\t");//k
			for(double time : run_time_plagiarism_extraction[line]) {
				System.out.print(time+"\t");
			}
			System.out.println();
		}
	}
	
	static void run_single_pair(ArrayList<double[][]> pair_matrices) {
		boolean header_written = false;
		for(int k_minus_3 = 0;k_minus_3<pair_matrices.size();k_minus_3++) {
			double[][] matrix = pair_matrices.get(k_minus_3);
			ArrayList<double[][]> matrices = new ArrayList<double[][]>();
			matrices.add(matrix);
			
			for(double core_threshold : core_thresholds) {
				matrices.add(expand_cluster_seeds(matrix,mark_cluster_seeds(matrix, k_minus_3+3, core_threshold)));	
			}			
			
			if(VERBOSE) {
				out_tsv(matrices);
				System.out.println();
				System.out.println();
			}else {
				if(!header_written) {
					System.out.print("k\t");
					for(int i=1;i<matrices.size();i++) {
						System.out.print("recall\tgran\t");
					}
					header_written=true;
					System.out.println();
				}
				System.out.print(k_minus_3+3+"\t");
				for(int i=1;i<matrices.size();i++) {	
					double recall = recall(matrices.get(i));
					double gran = gran(matrices.get(i));
					double[] res = {recall, gran};
					System.out.print(SemanticTest.outTSV(res));
					recall_results[k_minus_3][i-1].add(recall);
					gran_results[k_minus_3][i-1].add(gran);
				}
				System.out.println();
			}
		}
	}
	
	static double[][] run_time_plagiarism_extraction = null;
	static void run_single_pair_for_precision(ArrayList<double[][]> pair_matrices, ArrayList<double[][]> pair_excerpt_matrices) {
		boolean header_written = false; 
		if(run_time_plagiarism_extraction == null) {run_time_plagiarism_extraction = new double[pair_excerpt_matrices.size()][core_thresholds.length];}
		
		for(int k_minus_3 = 0;k_minus_3<pair_matrices.size();k_minus_3++) {
			double[][] matrix = pair_matrices.get(k_minus_3);
			ArrayList<double[][]> matrices = new ArrayList<double[][]>();
				
			matrices.add(matrix);
			for(double core_threshold : core_thresholds) {
				matrices.add(expand_cluster_seeds(matrix, mark_cluster_seeds(matrix, k_minus_3+3, core_threshold)));	
			}
						
			double[][] matrix_excerpt = pair_excerpt_matrices.get(k_minus_3);
			ArrayList<double[][]> matrices_excerpt = new ArrayList<double[][]>();
			matrices_excerpt.add(matrix_excerpt);
			for(double core_threshold : core_thresholds) {
				double start = System.currentTimeMillis();
				matrices_excerpt.add(expand_cluster_seeds(matrix_excerpt, mark_cluster_seeds(matrix_excerpt, k_minus_3+3,core_threshold)));
				double stop = System.currentTimeMillis();
				run_time_plagiarism_extraction[k_minus_3][get_index(core_threshold, core_thresholds)] += (stop-start);
			}			
			
			if(VERBOSE) {
				out_tsv(matrices);
				System.out.println();
				System.out.println();
			}else {
				if(!header_written) {
					System.out.print("k\t");
					for(int i=1;i<matrices.size();i++) {
						System.out.print("precision\t");
					}
					header_written=true;
					System.out.println();
				}
				System.out.print(k_minus_3+3+"\t");
				for(int i=1;i<matrices.size();i++) {	
					double precision = precision(matrices.get(i), matrices_excerpt.get(i), k_minus_3, i);
					System.out.print(precision+"\t");
				}
				System.out.println();
			}
		}
	}
	
	private static int get_index(double core_threshold, double[] coreThresholds) {
		for(int index = 0;index<core_thresholds.length;index++) {
			if(coreThresholds[index]==core_threshold) {
				return index;
			}
		}
		System.err.println("get_index() index not found retuning -1");
		return -1;
	}

	private static double precision(double[][] marked_cells, double[][] marked_cells_excerpt, final int k_minus_3, final int i) {
		boolean[] marked_lines = get_marked_lines(marked_cells);
		double count_all_positives = count_greater_zero(marked_lines); 
		
		boolean[] marked_lines_excerpt = get_marked_lines(marked_cells_excerpt);
		double count_true_positives = count_greater_zero(marked_lines_excerpt);
		
		double precision = count_true_positives / count_all_positives;
		
		precsion_results_all_positves[k_minus_3][i-1] += count_all_positives;
		precsion_results_true_positives[k_minus_3][i-1] += count_true_positives;
		if(!Double.isNaN(precision) && !Double.isInfinite(precision)) {
			precsion_results[k_minus_3][i-1].add(precision);
		}		
		return precision;
	}

	private static double gran(double[][] marked_cells) {
		boolean[] marked_lines = get_marked_lines(marked_cells);
		
		double count = 1.0d;
		boolean was_found = false;
		
		for(boolean b : marked_lines) {
			if(was_found) {
				if(b) {
					//still in the same found fragment
				}else {
					was_found = false;
				}
			}else{
				if(b) {//hit a new found fragment
					count++;
					was_found = true;
				}//else still not in a found fragment
			}
		}
		
		return count;
	}

	static double recall(double[][] marked_cells) {
		boolean[] marked_lines = get_marked_lines(marked_cells);
		double count = count_greater_zero(marked_lines); 
		
		return count / (double)marked_lines.length;
	}
	
	boolean[] get_marked_columns(double[][] marked_cells) {
		boolean[] marked_columns = new boolean[marked_cells[0].length];
		for(double[] line : marked_cells) {
			for(int i=0;i<line.length;i++) {
				if(line[i]>=IS_REACHABLE_CELL) {
					marked_columns[i] = true;
				}
			}
		}
		return marked_columns;
	}
	static boolean[] get_marked_lines(double[][] marked_cells) {
		boolean[] marked_lines = new boolean[marked_cells.length];
		for(int l=0;l<marked_cells.length;l++) {
			double[] line = marked_cells[l];
			for(int c=0;c<line.length;c++) {
				if(line[c]>=IS_REACHABLE_CELL) {
					marked_lines[l] = true;
				}
			}
		}
		return marked_lines;
	}
	
	double count_greater_zero(double[] array) {
		double count = 0.0d;
		for(double d : array) {
			if(d>0.0d) {
				count++;	
			}
		}
		return count;
	}
	
	static double count_greater_zero(boolean[] array) {
		double count = 0.0d;
		for(boolean b : array) {
			if(b) {
				count++;	
			}
		}
		return count;
	}
	
	static void out_tsv(double[][] matrix, double[][] clusters) {
		for(int line=0;line<matrix.length;line++) {
			String out = SemanticTest.outTSV(matrix[line])+"\t\t\t\t"+SemanticTest.outTSV(clusters[line]);
			System.out.println(out);
		}
		
	}
	static void out_tsv(ArrayList<double[][]> arrays) {
		for(int line=0;line<arrays.get(0).length;line++) {
			ArrayList<String> out_strings = new ArrayList<String>(arrays.size());
			for(double[][] matrix : arrays) {
				out_strings.add(SemanticTest.outTSV(matrix[line]));
			}
			
			for(String s : out_strings) {
				System.out.print(s);
				System.out.print("\t\t\t\t");
			}
			System.out.println();
		}
		
	}

	public static void out_aggregated_result() {
		System.out.println("PotthastMetrics.out_aggregated_result()");
		double[][] recall = null;
		if(recall_results!=null) {
			recall = new double[recall_results.length][recall_results[0].length];
			System.out.println();
			System.out.println("Average Recall");
			System.out.print("k\t");
			for(double d : core_thresholds) {
				System.out.print("recall "+d+"\t");
			}
			System.out.println();
			for(int k_minus_3=0;k_minus_3<recall_results.length;k_minus_3++) {
				ArrayList<Double>[] line = recall_results[k_minus_3];
				System.out.print(k_minus_3+3+"\t");
				for(int i=0;i<line.length;i++) {
					double temp = avg(line[i]);
					recall[k_minus_3][i] = temp;
					System.out.print(temp+"\t");	
				}
				System.out.println();
			}
		}
		double[][] gran = null;
		if(gran_results!=null) {
			gran = new double[gran_results.length][gran_results[0].length];
			System.out.println();
			System.out.println("Average Granularity");
			System.out.print("k\t");
			for(double d : core_thresholds) {
				System.out.print("gran "+d+"\t");
			}
			System.out.println();
			for(int k_minus_3=0;k_minus_3<gran_results.length;k_minus_3++) {
				ArrayList<Double>[] line = gran_results[k_minus_3];
				System.out.print(k_minus_3+3+"\t");
				for(int i=0;i<line.length;i++) {
					double temp = avg(line[i]);
					gran[k_minus_3][i] = temp;
					System.out.print(temp+"\t");	
				}
				System.out.println();
			}
		}
		double[][] weigted_gran = null;
		if(gran!=null) {
			weigted_gran = new double[gran.length][gran[0].length];
			System.out.println();
			System.out.println("Weigthed Granularity");
			System.out.print("k\t");
			for(double d : core_thresholds) {
				System.out.print("weigted_gran "+d+"\t");
			}
			System.out.println();
			for(int k_minus_3=0;k_minus_3<gran.length;k_minus_3++) {
				double[] line = gran[k_minus_3];
				System.out.print(k_minus_3+3+"\t");
				for(int i=0;i<line.length;i++) {
					double temp = log_2(1+line[i]);
					weigted_gran[k_minus_3][i] = temp;
					System.out.print(temp+"\t");	
				}
				System.out.println();
			}
		}
		double[][] precison = null;
		if(precsion_results_all_positves!=null && precsion_results_true_positives!=null) {
			precison = new double[precsion_results_all_positves.length][precsion_results_all_positves[0].length];
			System.out.println();
			System.out.println("Precision");
			System.out.print("k\t");
			for(double d : core_thresholds) {
				System.out.print(d+"\t");
			}
			System.out.println();
			for(int line=0;line<precsion_results_all_positves.length;line++) {
				System.out.print(line+3+"\t");
				for(int colum=0;colum<precsion_results_all_positves[0].length;colum++) {
					double temp = precsion_results_true_positives[line][colum] / precsion_results_all_positves[line][colum];
					precison[line][colum] = temp;
					System.out.print(temp +"\t");  
				}
				System.out.println();
			}
		}
		//Alternative document-wise precision
		if(precsion_results!=null) {
			System.out.println();
			System.out.println("Precision (alternative)");
			System.out.print("k\t");
			for(double d : core_thresholds) {
				System.out.print(d+"\t");
			}
			System.out.println();
			for(int line=0;line<precsion_results.length;line++) {
				System.out.print(line+3+"\t");
				for(int colum=0;colum<precsion_results[0].length;colum++) {
					double temp = avg(precsion_results[line][colum]);
					if(temp>precison[line][colum]) {precison[line][colum]=temp;}//replace it, though I do not like this way of computing precision
					System.out.print(temp +"\t");  
				}
				System.out.println();
			}
		}
		double[][] harmonic_mean = null;
		if(precison!=null && recall!=null) {
			harmonic_mean = new double[precison.length][precison[0].length];
			System.out.println();
			System.out.println("Harmonic mean");
			System.out.print("k\t");
			for(double d : core_thresholds) {
				System.out.print(d+"\t");
			}
			System.out.println();
			for(int line=0;line<precison.length;line++) {
				System.out.print(line+3+"\t");
				for(int colum=0;colum<precison[0].length;colum++) {
					double hm = 2.0d / ((1.0d/ precison[line][colum]) + (1.0d / recall[line][colum]));
					harmonic_mean[line][colum] = hm;
					System.out.print(hm +"\t");  
				}
				System.out.println();
			}
		}
		if(harmonic_mean!=null && weigted_gran!=null) {
			System.out.println();
			System.out.println("Overall score");
			System.out.print("k\t");
			for(double d : core_thresholds) {
				System.out.print(d+"\t");
			}
			System.out.println();
			for(int line=0;line<precison.length;line++) {
				System.out.print(line+3+"\t");
				for(int colum=0;colum<precison[0].length;colum++) {
					double temp = harmonic_mean[line][colum] / weigted_gran[line][colum];
					System.out.print(temp +"\t");  
				}
				System.out.println();
			}
		}
	}
	private static double log_2(double d) {
		return Math.log(d) / Math.log(2.0d);
	}
}
