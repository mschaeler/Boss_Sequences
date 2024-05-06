package pan;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;

import boss.test.SemanticTest;
import boss.util.Config;

public class PanMetrics {//TODO micro average
	static final double[] core_thresholds = {1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1};
	static final double IS_CORE_CELL = 1.0d;
	static final double IS_REACHABLE_CELL = 0.5d;
	public static double CORE_CELL_THRESHOLD = 0.9d;
	public static double REACHABILITY_CELL_THRESHOLD = 0.5d;
	
	
	public static void run_seda() {
		PanResult.clear();
		MatrixLoader.path_to_matrices = MatrixLoader.path_to_pan_matrices;
		//SemanticTest.print_seda_texts();//to get the ground truth
		new PanMetrics().run_seda_();
	}
	
	public static void run_jaccard() {
		PanResult.clear();
		MatrixLoader.path_to_matrices = MatrixLoader.path_to_jaccard_matrices;
		SemanticTest.print_jaccard_texts();//to get the ground truth
		new PanMetrics().run();
	}
	
	public static void out_results() {
		System.out.println();
		System.out.println("Aggregated Results of PAN 11");
		ArrayList<Integer> k_s = PanResult.get_all_k_values();
		ArrayList<Double> thresholds = PanResult.get_all_thresholds();
		System.out.print("Precision macro");
		for(double t : thresholds) {
			System.out.print("\t"+t);
		}
		System.out.print("\t\tRecall macro");
		for(double t : thresholds) {
			System.out.print("\t"+t);
		}
		System.out.print("\t\tGranularity");
		for(double t : thresholds) {
			System.out.print("\t"+t);
		}
		System.out.println();
		ArrayList<String> micro_metric = new ArrayList<String>();
		for(int k : k_s) {
			//macro
			String precision="k="+k;
			String recall="k="+k;
			String granulartiy="k="+k;
			//micro
			String micro_precision="k="+k;
			String micro_recall="k="+k;
			for(double t : thresholds) {
				ArrayList<PanResult> r = PanResult.get_results(k,t);
				/*if(k==15) {
					System.out.println("Raw results for k="+k+" theta="+t);
					for(PanResult pr : r) {
						System.out.println(pr);
					}
					System.out.println();
				}*/
				double avg_recall    = 0;
				double avg_precision = 0;
				double avg_gran		 = 0;
				double sum_all_true_postives   = 0.0d;
				double sum_found_true_postives = 0.0d;
				double sum_retrieved		   = 0.0d;
				
				for(PanResult pr : r) {
					//System.out.println(pr);
					avg_recall 	  += pr.recall;
					avg_precision += pr.precision;
					avg_gran 	  += pr.granularity;
					sum_all_true_postives   += pr.all_true_positives;
					sum_found_true_postives += pr.found_true_positives;
					sum_retrieved		    += pr.retrieved_elements;
				}
				//macro metrics
				avg_recall /= (double) r.size();
				recall+="\t"+avg_recall;
				avg_precision /= (double) r.size();
				precision+="\t"+avg_precision;
				avg_gran /= (double) r.size();
				granulartiy+="\t"+avg_gran;
				//micro metrics
				double m_recall = sum_found_true_postives / sum_all_true_postives;
				micro_recall+="\t"+m_recall;
				double m_precision = sum_found_true_postives / sum_retrieved;
				micro_precision+="\t"+m_precision;
			
			}
			System.out.println(precision+"\t\t"+recall+"\t\t"+granulartiy);
			micro_metric.add(micro_precision+"\t\t"+micro_recall);
		}
		System.out.println();
		System.out.print("Precision micro");
		for(double t : thresholds) {
			System.out.print("\t"+t);
		}
		System.out.print("\t\tRecall micro");
		for(double t : thresholds) {
			System.out.print("\t"+t);
		}
		System.out.println();
		for(String s : micro_metric) {
			System.out.println(s);
		}
	}
	
	String get_excerpts(String name, List<String> ex) {
		String[] tokens = name.split("_");// susp_00228_src_05889 -> [susp, 00228 , src, 05889]
		for(String e : ex) {
			if(e.contains(tokens[1]) && e.contains("_"+tokens[3])) {
				return e;
			}
		}
		System.err.println("get_excerpts(String, List<String>) did not find excerpt of "+name);
		return null;
	}
	
	void run_seda_() {
		List<String> l = MatrixLoader.get_all_susp_src_directories();
		List<String> ex = MatrixLoader.get_all_excerpt_directories();
		
		for(int i=0;i<l.size();i++) {//For each data set pair
			//String dir  = l.get(i);
			String name = l.get(i);
			String excerpts = get_excerpts(name, ex);
			
			HashMap<Integer,double[][]> my_matrices = MatrixLoader.load_all_matrices_of_pair_hashed(name);
			HashMap<Integer,double[][]> my_excerpts_matrices = MatrixLoader.load_all_matrices_of_pair_hashed(excerpts);
			
			for(Entry<Integer, double[][]> e : my_matrices.entrySet()) {//For each window size k
				final int k = e.getKey();
				double[][] matrix = e.getValue();
				double[][] e_matrix = my_excerpts_matrices.get(k);
				final double real_count_true_positives = e_matrix.length+e_matrix[0].length;//This is the matrix of the part being the plagiarism cse
				
				for(double core_threshold : core_thresholds) {// For each core_threshold
					double start = System.currentTimeMillis();
					double[][] marked_cells = (expand_cluster_seeds(matrix, mark_cluster_seeds(matrix, k, core_threshold)));//TODO measure time
					double stop = System.currentTimeMillis();
					final boolean[] marked_susp = new boolean[marked_cells.length+k-1]; 
					final boolean[] marked_src  = new boolean[marked_cells[0].length+k-1];
					get_marked_lines(marked_cells, marked_susp, marked_src, k);
					
					double[][] e_marked_cells = (expand_cluster_seeds(e_matrix, mark_cluster_seeds(e_matrix, k, core_threshold)));
					final boolean[] e_marked_susp = new boolean[marked_cells.length+k-1]; 
					final boolean[] e_marked_src  = new boolean[marked_cells[0].length+k-1];
					get_marked_lines(e_marked_cells, e_marked_susp, e_marked_src, k);
					
					double true_positives_2 = get_retrieved_elements(e_marked_susp, e_marked_src);
					double retrieved_elements = get_retrieved_elements(marked_susp, marked_src);
					
					double recall = true_positives_2 / real_count_true_positives;
					double precision = (retrieved_elements>0) ? true_positives_2 / retrieved_elements : 0;
					
					double granularity 	  = gran(e_marked_susp);
					PanResult pr = new PanResult(name, k, core_threshold, precision, recall, granularity, real_count_true_positives, true_positives_2, retrieved_elements);
					System.out.println(pr);
				}
			}
		}
		out_results();
	}
	
	
	
	void run() {
		List<String> l = MatrixLoader.get_all_susp_src_directories();
		
		for(int i=0;i<l.size();i++) {//For each data set pair
			//String dir  = l.get(i);
			String name = l.get(i);
			int[] base_line_in_tokens = get_base_line(name);
			
			
			
			HashMap<Integer,double[][]> my_matrices = MatrixLoader.load_all_matrices_of_pair_hashed(name);
			for(Entry<Integer, double[][]> e : my_matrices.entrySet()) {//For each window size k
				final int k = e.getKey();
				final int[] base_line = get_baseline_in_windows(base_line_in_tokens, k);
				final double real_count_true_positives = (base_line[1]-base_line[0]+1)+(base_line[3]-base_line[2]+1);//inclusively
				double[][] matrix = e.getValue();
				for(double core_threshold : core_thresholds) {// For each core_threshold
					double start = System.currentTimeMillis();
					double[][] marked_cells = (expand_cluster_seeds(matrix, mark_cluster_seeds(matrix, k, core_threshold)));//TODO measure time
					double stop = System.currentTimeMillis();
					
					final boolean[] marked_susp = new boolean[marked_cells.length+k-1]; 
					final boolean[] marked_src  = new boolean[marked_cells[0].length+k-1];
					get_marked_lines(marked_cells, marked_susp, marked_src, k);
					
					//double true_positives = get_true_positives(ground_truth_susp, ground_truth_src, marked_susp, marked_src);
					double true_positives_2 = get_true_positives(base_line, marked_cells, k);
					/*if(true_positives == true_positives_2) {
						System.out.println("true_positives == true_positives_2");
					}else if(true_positives > true_positives_2) {
						System.out.println("true_positives > true_positives_2");
					}else {
						System.err.println("true_positives < true_positives_2");
					}*/
					
					double retrieved_elements = get_retrieved_elements(marked_susp, marked_src);
					
					double recall = true_positives_2 / real_count_true_positives;
					double precision = (retrieved_elements>0) ? true_positives_2 / retrieved_elements : 0;
					
					double granularity 	  = gran(marked_susp, base_line[0], base_line[1]);
					PanResult pr = new PanResult(name, k, core_threshold, precision, recall, granularity, real_count_true_positives, true_positives_2, retrieved_elements);
					System.out.println(pr);
				}
			}
		}
		out_results();
	}
	
	private int[] get_baseline_in_windows(int[] base_line_in_tokens, int k) {
		//if(k==15)
		//	System.out.println(k);
		int[] base_line = new int[base_line_in_tokens.length];
		System.arraycopy(base_line_in_tokens, 0, base_line, 0, base_line_in_tokens.length);//it remains as it is
		if(Config.USE_FULL_PLAGIAT_CELLS) {
			//modify the end of the plagiarism run.
			base_line[1] -=k+1;
			if(base_line[1]<base_line[0]) {
				base_line[1]=base_line[0];
			}
			base_line[3] -=k+1;
			if(base_line[3]<base_line[2]) {
				base_line[3]=base_line[2];
			}
		}
		return base_line;
	}

	/**
	 * For excerpts
	 * @param marked_cells
	 * @param k
	 * @return
	 */
	double get_true_positives(double[][] marked_cells, final int k) {
		boolean[] marked_susp = new boolean[marked_cells.length];
		boolean[] marked_src = new boolean[marked_cells[0].length];
		
		for(int l=0;l<marked_cells.length;l++) {
			double[] line = marked_cells[l];
			for(int c=0;c<line.length;c++) {
				if(line[c]>=IS_REACHABLE_CELL) {
					marked_susp[l] = true;
					marked_src[c]  = true;
				}
			}
		}
		double found_true_positives = get_retrieved_elements(marked_susp, marked_src);
		return found_true_positives;
	}
	
	double get_true_positives(int[] base_line, double[][] marked_cells, final int k) {
		boolean[] marked_susp = new boolean[marked_cells.length];
		boolean[] marked_src = new boolean[marked_cells[0].length];
		
		//Scan only in border of real plagiarism case
		for(int l=base_line[0];l<=base_line[1]-k+1;l++) {
			double[] line = marked_cells[l];
			for(int c=base_line[2];c<=base_line[3]-k+1;c++) {
				if(line[c]>=IS_REACHABLE_CELL) {
					marked_susp[l] = true;
					marked_src[c]  = true;
				}
			}
		}
		double found_true_positives = get_retrieved_elements(marked_susp, marked_src);
		return found_true_positives;
	}

	double get_retrieved_elements(boolean[] marked_susp, boolean[] marked_src) {
		double retrieved_elements = 0.0d;
		for(boolean b : marked_susp) {
			if(b) {
				retrieved_elements++;
			}
		}
		for(boolean b : marked_src) {
			if(b) {
				retrieved_elements++;
			}
		}
		return retrieved_elements;
	}

	double get_true_positives(BitSet ground_truth_susp, BitSet ground_truth_src, boolean[] marked_susp, boolean[] marked_src) {
		double true_positives = 0.0d;
		for (int i = ground_truth_susp.nextSetBit(0); i != -1; i = ground_truth_susp.nextSetBit(i + 1)) {
		    if(marked_susp[i]) {//we also found it
		    	true_positives++;
		    }
		}
		for (int i = ground_truth_src.nextSetBit(0); i != -1; i = ground_truth_src.nextSetBit(i + 1)) {
		    if(marked_src[i]) {//we also found it
		    	true_positives++;
		    }
		}
		return true_positives;
	}

	void get_marked_lines(double[][] marked_cells, boolean[] marked_susp, boolean[] marked_src, int k) {
		for(int l=0;l<marked_cells.length;l++) {
			double[] line = marked_cells[l];
			for(int c=0;c<line.length;c++) {
				if(line[c]>=IS_REACHABLE_CELL) {
					marked_susp[l] = true;
					marked_src[c]  = true;
				}
			}
		}
		
	}

	BitSet get_marked_lines(double[][] marked_cells, final int k) {
		BitSet marked_lines = new BitSet(marked_cells.length);
		for(int l=0;l<marked_cells.length;l++) {
			double[] line = marked_cells[l];
			for(int c=0;c<line.length;c++) {
				if(line[c]>=IS_REACHABLE_CELL) {
					marked_lines.set(l);
					//break;//XXX
				}
			}
		}
		//TODO jetzt nur auf Zellen Basis -> Token Basis
		return marked_lines;
	}
	
	BitSet get_marked_columns(double[][] marked_cells, final int k) {
		final int num_lines = marked_cells.length;
		final int num_columns = marked_cells[0].length;
		
		BitSet marked_columns = new BitSet(num_columns);
		for(int c=0;c<num_columns;c++) {
			for(int l=0;l<num_lines;l++) {
				if(marked_cells[l][c] >=IS_REACHABLE_CELL) {
					marked_columns.set(c);
					//break;//XXX
				} 
			}
		}
		//TODO jetzt nur auf Zellen Basis -> Token Basis
		return marked_columns;
	}
	
	double[][] mark_cluster_seeds(double[][] matrix, final int k, final double core_threshold){
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
	
	boolean is_core_cell(double[][] matrix, int line, int column, final double threshold) {
		double cell_similarity = matrix[line][column];
		if(cell_similarity>threshold) {
			return true;
		}else{
			return false;
		}
	}

	double[][] expand_cluster_seeds(double[][] matrix, double[][] cluster_seeds){
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
	
	void expand_cluster_seed_cell(final double[][] matrix, final double[][] cluster_seeds, final int line, final int column) {
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
	
	double count_all_positive_tokens(boolean[] marked_cells) {
		double count = 0;
		for(int pos = 0;pos<marked_cells.length;pos++) {//exclusively
			if(marked_cells[pos]) {
				count++;
			}
		}
		return count;
	}
	
	double count_false_positive_tokens(boolean[] marked_cells, int start, int stop) {
		double count = 0;
		for(int pos = 0;pos<start;pos++) {//exclusively
			if(marked_cells[pos]) {
				count++;
			}
		}
		for(int pos = stop+1;pos<=marked_cells.length;pos++) {//exclusively
			if(marked_cells[pos]) {
				count++;
			}
		}
		return count;
	}

	double count_true_positives(boolean[] marked_cells, int start, int stop) {
		double count = 0;
		for(int pos = start;pos<=stop;pos++) {//inclusively
			if(marked_cells[pos]) {
				count++;
			}
		}
		return count;
	}
	
	double gran(boolean[] marked_lines) {		
		double count = 0.0d;
		boolean was_found = false;
		
		for(int i=0;i<marked_lines.length;i++) {
			boolean b = marked_lines[i];
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
	
	double gran(boolean[] marked_lines, int susp_ground_truth_from, int susp_ground_truth_to) {		
		double count = 0.0d;
		boolean was_found = false;
		
		for(int i=susp_ground_truth_from;i<=susp_ground_truth_to;i++) {
			boolean b = marked_lines[i];
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

	int[] get_base_line(String name) {
		int[] base_line = Jaccard.materialized_ground_truth.get(name);
		if(base_line==null) {
			System.err.println("PanMetrics.get_base_line(String name) no such baseline for "+name);
		}
		return base_line;
	}
}
