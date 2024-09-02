package pan;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;
import boss.util.Config;

public class PanMetrics {//TODO micro average
	//static final double[] core_thresholds = {1.0,0.9,0.8,0.7,0.6,0.5,0.4,0.3,0.2,0.1};
	static final double[] core_thresholds = {0.69,0.68,0.67,0.66,0.65,0.64,0.63,0.62,0.61,0.6};
	//static final double[] core_thresholds = {0.8,0.79,0.78,0.77,0.76,0.75,0.74,0.73,0.72,0.71,0.70,0.69,0.68};
	static final double IS_CORE_CELL = 1.0d;
	static final double IS_REACHABLE_CELL = 0.5d;
	public static double REACHABILITY_CELL_THRESHOLD = 0.5d;
	
	String name;
	PanMetrics(String name){
		this.name = name;
		print_config();
	}
	
	private void print_config() {
		System.out.println("PanMetrics("+name+") Config.PLAGIAT_GRANUALRITY="+Config.PLAGIAT_GRANUALRITY+ " USE_CONNECTIVITY_THRESHOLD="+Config.USE_CONNECTIVITY_THRESHOLD+" Config.REMOVE_STOP_WORDS="+Config.REMOVE_STOP_WORDS+" IS_REACHABLE_CELL="+IS_REACHABLE_CELL+" Config.USE_TXT_ALIGN_PREPROCESSING="+Config.USE_TXT_ALIGN_PREPROCESSING);
	}
	
	public static void run_seda() {
		PanResult.clear();
		MatrixLoader.path_to_matrices = MatrixLoader.path_to_pan_matrices;
		//SemanticTest.print_seda_texts();//to get the ground truth
		new PanMetrics("SeDA").run_seda_();
	}
	
	public static void run_jaccard() {
		PanResult.clear();
		MatrixLoader.path_to_matrices = MatrixLoader.path_to_jaccard_matrices;
		//SemanticTest.print_jaccard_texts();//to get the ground truth
		new PanMetrics("Jaccard").run_seda_();
	}
	
	public void out_results() {
		System.out.println();
		System.out.print("Aggregated Results of PAN 11\t");
		print_config();
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
	
	public static String get_excerpts(String name, List<String> ex) {
		String[] tokens = name.split("_");// susp_00228_src_05889 -> [susp, 00228 , src, 05889]
		for(String e : ex) {
			if(e.contains(tokens[1]) && e.contains("_"+tokens[3])) {
				return e;
			}
		}
		System.err.println("get_excerpts(String, List<String>) did not find excerpt of "+name);
		return null;
	}
	
	/**
	 * 
	 * @param ground_truths
	 * @param detections
	 */
	public static void run_xml_detections(ArrayList<Detection> ground_truths, ArrayList<Detection> detections) {
		HashSet<String> all_susp_documents = new HashSet<String>();
		HashSet<Integer> all_ks = new HashSet<Integer>(); 
		HashSet<Double> all_thetas = new HashSet<Double>();
		
		for(Detection gt : ground_truths) {
			 all_susp_documents.add(gt.susp);
		}
		for(Detection d : detections) {
			all_ks.add(d.k);
			all_thetas.add(d.theta);
		}
		
		for(String f : all_susp_documents) {
			System.out.println(f);
		}
		for(Integer k : all_ks) {
			System.out.println(k);
		}
		for(Double theta : all_thetas) {
			System.out.println(theta);
		}
		
		for(String susp : all_susp_documents) {
			//get all susp detections
			ArrayList<Detection> src_documents = get_all_detections_by_name(susp, ground_truths);
			for(Detection plagiat : src_documents) {
				ArrayList<Detection> corresponding_detections = new ArrayList<Detection>();
				for(Detection d : detections) {
					if(d.susp.equals(plagiat.susp) && d.src.equals(plagiat.src)) {
						corresponding_detections.add(d);
					}
				}
				//for each k
				for(int k : all_ks) {
					//for each threshold
					for(double theta : all_thetas) {
						ArrayList<Detection> my_detections = get_detections(k, theta, corresponding_detections);
						compute(plagiat, my_detections , susp, k, theta);
					}
				}
				//get all the other detections having d.susp = susp but d.src not in src_documents
				ArrayList<Detection> falsely_retrieved_documents = new ArrayList<Detection>();
				for(Detection d : detections) {
					if(d.susp.equals(susp) && !src_documents.contains(d.src)) {
						falsely_retrieved_documents.add(d);
					}
				}
			}
		}
		new PanMetrics("Org TxtAlign").out_results();
	}
	static void compute(Detection plagiat, ArrayList<Detection> my_detections, String name, int k, double theta) {
		int max_position_susp = plagiat.this_offset+plagiat.this_length;
		int max_position_src  = plagiat.source_offset+plagiat.source_length;
		for(Detection my_d : my_detections) {
			if(my_d.this_offset+my_d.this_length > max_position_susp) {
				max_position_susp = my_d.this_offset+my_d.this_length;
			}
			if(my_d.source_offset+my_d.source_length > max_position_src) {
				max_position_src = my_d.source_offset+my_d.source_length;
			}
		}
		boolean[] found_susp = new boolean[max_position_susp];
		boolean[] found_src  = new boolean[max_position_src];
		for(Detection my_d : my_detections) {
			mark_pos(found_susp, my_d.this_offset, my_d.this_offset+my_d.this_length);
			mark_pos(found_src, my_d.source_offset, my_d.source_offset+my_d.source_length);
		}
		double all_true_positives = (plagiat.this_offset+plagiat.this_length)+(plagiat.source_offset+plagiat.source_length); 
		double retrieved_elements = count(found_susp);
		retrieved_elements += count(found_src);
		
		double found_true_positives = count(found_susp, plagiat.this_offset, plagiat.this_offset+plagiat.this_length);
		found_true_positives += count(found_src, plagiat.source_offset, plagiat.source_offset+plagiat.source_length);
		
		double recall = found_true_positives / all_true_positives;
		if(recall>1) {
			System.err.println("recall>1");
		}
		double precision = (retrieved_elements>0) ? found_true_positives / retrieved_elements : 0;
		PanResult pr = new PanResult(name, k, theta, precision, recall, my_detections.size(), all_true_positives, found_true_positives, retrieved_elements);
		System.out.println(pr);
	}
	
	private static double count(boolean[] arr, int from, int to) {
		double count=0;
		for(int pos=from;pos<to;pos++) {
			if(arr[pos]) {
				count++;
			}
		}
		return count;
	}

	private static double count(boolean[] arr) {
		double count=0;
		for(boolean b : arr) {
			if(b) {
				count++;
			}
		}
		return count;
	}

	private static void mark_pos(boolean[] arr, int from, int to) {
		for(int pos=from;pos<to;pos++) {
			arr[pos] = true;
		}
	}

	private static ArrayList<Detection> get_detections(int k, double theta, ArrayList<Detection> corresponding_detections) {
		ArrayList<Detection> res = new ArrayList<Detection>();
		for(Detection d : corresponding_detections) {
			if(d.k == k && d.theta == theta) {
				res.add(d);
			}
		}
		return res;
	}

	/**
	 * get all susp detections from ground truth
	 */
	static ArrayList<Detection> get_all_detections_by_name(String name, ArrayList<Detection> detections ){
		ArrayList<Detection> src_documents = new ArrayList<Detection>();
		for(Detection gt : detections) {
			 if(gt.susp.equals(name)) {
				 System.out.println(gt);
				 src_documents.add(gt);
			 }
		}
		return src_documents;
	}
	
	
	
	public void run_seda_() {
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
				final int[] baseline = find_baseline(matrix, e_matrix, k);//different for each k
				
				if(Config.PLAGIAT_GRANUALRITY==Config.PLAGIAT_GRANUALRITY_TOKEN) {
					if(baseline[1]-baseline[0]!=e_matrix.length){
						System.err.println("susp: "+e_matrix.length+" vs."+(baseline[1]-baseline[0])+" "+Arrays.toString(baseline)+" "+k);
					}
					if(baseline[3]-baseline[2]!=e_matrix[0].length){
						System.err.println("src: "+e_matrix[0].length+" vs."+(baseline[3]-baseline[2])+" "+Arrays.toString(baseline)+" "+k);
					}
				}
				
				for(double core_threshold : core_thresholds) {// For each core_threshold
					double start = System.currentTimeMillis();
					double[][] marked_cells;
					if(Config.USE_CONNECTIVITY_THRESHOLD) {
						marked_cells = (expand_cluster_seeds(matrix, mark_cluster_seeds(matrix, k, core_threshold)));//TODO measure time
					}else{
						marked_cells = (mark_cluster_seeds(matrix, k, core_threshold));
					}
					 
					double stop = System.currentTimeMillis();
					
					boolean[] marked_susp = null; 
					boolean[] marked_src = null;
					if(Config.PLAGIAT_GRANUALRITY==Config.PLAGIAT_GRANUALRITY_CELL) {
						marked_susp = new boolean[marked_cells.length+k-1]; 
						marked_src  = new boolean[marked_cells[0].length+k-1];
					}else if(Config.PLAGIAT_GRANUALRITY==Config.PLAGIAT_GRANUALRITY_TOKEN) {
						marked_susp = new boolean[marked_cells.length+k-1]; 
						marked_src  = new boolean[marked_cells[0].length+k-1];
					}else {
						System.err.println("init() not supported Config.PLAGIAT_GRANUALRITY "+Config.PLAGIAT_GRANUALRITY);
					}
					get_marked_tokens(marked_cells, marked_susp, marked_src, k);
					
					double[][] ex_marked_cells = (expand_cluster_seeds(e_matrix, mark_cluster_seeds(e_matrix, k, core_threshold)));//TODO measure time
					if(Config.USE_CONNECTIVITY_THRESHOLD) {
						ex_marked_cells = (expand_cluster_seeds(e_matrix, mark_cluster_seeds(e_matrix, k, core_threshold)));//TODO measure time
					}else{
						ex_marked_cells = (mark_cluster_seeds(e_matrix, k, core_threshold));//TODO measure time
					}
					
					boolean[] ex_marked_susp = null; 
					boolean[] ex_marked_src = null;
					if(Config.PLAGIAT_GRANUALRITY==Config.PLAGIAT_GRANUALRITY_CELL) {
						ex_marked_susp = new boolean[ex_marked_cells.length+k-1]; 
						ex_marked_src  = new boolean[ex_marked_cells[0].length+k-1];
					}else if(Config.PLAGIAT_GRANUALRITY==Config.PLAGIAT_GRANUALRITY_TOKEN) {
						ex_marked_susp = new boolean[ex_marked_cells.length+k-1]; 
						ex_marked_src  = new boolean[ex_marked_cells[0].length+k-1];
					}else {
						System.err.println("init() not supported Config.PLAGIAT_GRANUALRITY "+Config.PLAGIAT_GRANUALRITY);
					}
					get_marked_tokens(ex_marked_cells, ex_marked_susp, ex_marked_src, k);
					
					double true_positives = get_retrieved_elements(ex_marked_susp, ex_marked_src);
					double retrieved_elements = get_retrieved_elements(marked_susp, marked_src);
					final double real_count_true_positives = ex_marked_susp.length+ex_marked_src.length;
					
					double recall = true_positives / real_count_true_positives;
					if(recall>1) {
						System.err.println("recall>1");
					}
					double precision = (retrieved_elements>0) ? true_positives / retrieved_elements : 0;
					
					double granularity 	  = gran(marked_susp, baseline[0], baseline[1]);
					PanResult pr = new PanResult(name, k, core_threshold, precision, recall, granularity, real_count_true_positives, true_positives, retrieved_elements);
					System.out.println(pr);
				}
			}
		}
		out_results();
	}

	private double get_real_count_true_positives(final int[] baseline) {
		return (baseline[1]-baseline[0])+(baseline[3]-baseline[2]);
	}
	
	
	private double get_true_positives(int[] baseline, boolean[] marked_susp, boolean[] marked_src) {
		double count = 0;
		for(int i=baseline[0];i<baseline[1];i++) {
			if(marked_susp[i]) {
				count++;
			}
		}
		for(int i=baseline[2];i<baseline[3];i++) {
			if(marked_src[i]) {
				count++;
			}
		}
		return count;
	}

	private int[] find_baseline(final double[][] matrix, final double[][] excerpt_matrix, final int k) {
		for(int line=0;line<matrix.length;line++) {
			for(int column=0;column<matrix[0].length;column++) {
				if(matrix[line][column]==excerpt_matrix[0][0]) {//Found possible start
					if(found(matrix,excerpt_matrix,line,column)) {
						if(Config.PLAGIAT_GRANUALRITY==Config.PLAGIAT_GRANUALRITY_CELL) {
							int[] baseline = {line, line+excerpt_matrix.length+k-1,column,column+excerpt_matrix[0].length+k-1};
							return baseline;	
						}else if(Config.PLAGIAT_GRANUALRITY==Config.PLAGIAT_GRANUALRITY_TOKEN) {
							int[] baseline = {line, line+excerpt_matrix.length,column,column+excerpt_matrix[0].length};
							return baseline;
						}else {
							System.err.println("find_baseline() mnot supported Config.PLAGIAT_GRNAUALRITY "+Config.PLAGIAT_GRANUALRITY);
							return null;
						}
					}
				}
			}
		}
		System.err.println("find_baseline() : did not find ther excerpt returnin null");
		return null;
	}

	private boolean found(double[][] matrix, double[][] excerpt_matrix, final int start_line, final int start_column) {
		for(int l=start_line;l<excerpt_matrix.length;l++) {
			for(int c=start_column;c<excerpt_matrix[0].length;c++) {
				if(matrix[start_line+l][start_column+c]!=excerpt_matrix[l][c]) {
					return false;//found first difference
				}
			}
		}
		return true;
	}

	private void run() {
		List<String> l = MatrixLoader.get_all_susp_src_directories();
		
		for(int i=0;i<l.size();i++) {//For each data set pair
			//String dir  = l.get(i);
			String name = l.get(i);
			int[] base_line_in_tokens = get_base_line(name);
			
			HashMap<Integer,double[][]> my_matrices = MatrixLoader.load_all_matrices_of_pair_hashed(name);
			for(Entry<Integer, double[][]> e : my_matrices.entrySet()) {//For each window size k
				final int k = e.getKey();
				final int[] base_line = get_baseline_based_on_plag_genaularity(base_line_in_tokens, k);
				final double real_count_true_positives = get_real_count_true_positives(base_line);
				double[][] matrix = e.getValue();
				for(double core_threshold : core_thresholds) {// For each core_threshold
					double start = System.currentTimeMillis();
					double[][] marked_cells = (expand_cluster_seeds(matrix, mark_cluster_seeds(matrix, k, core_threshold)));//TODO measure time
					double stop = System.currentTimeMillis();
					
					final boolean[] marked_susp = new boolean[marked_cells.length+k-1]; 
					final boolean[] marked_src  = new boolean[marked_cells[0].length+k-1];
					get_marked_tokens(marked_cells, marked_susp, marked_src, k);
					
					double true_positives_2 = get_true_positives(base_line, marked_susp, marked_src);
					
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
	
	private int[] get_baseline_based_on_plag_genaularity(int[] base_line_in_tokens, int k) {
		//if(k==15)
		//	System.out.println(k);
		int[] base_line = new int[base_line_in_tokens.length];
		System.arraycopy(base_line_in_tokens, 0, base_line, 0, base_line_in_tokens.length);//it remains as it is
		if(Config.PLAGIAT_GRANUALRITY == Config.PLAGIAT_GRANUALRITY_CELL) {
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
	private double get_true_positives(double[][] marked_cells, final int k) {
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

	private double get_retrieved_elements(boolean[] marked_susp, boolean[] marked_src) {
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

	private void get_marked_tokens(double[][] marked_cells, boolean[] marked_susp, boolean[] marked_src, int k) {
		if(Config.PLAGIAT_GRANUALRITY==Config.PLAGIAT_GRANUALRITY_CELL) {
			for(int l=0;l<marked_cells.length;l++) {
				double[] line = marked_cells[l];
				for(int c=0;c<line.length;c++) {
					if(line[c]>=IS_REACHABLE_CELL) {
						marked_susp[l] = true;//this cell only
						marked_src[c]  = true;	
					}
				}
			}
		}else if(Config.PLAGIAT_GRANUALRITY==Config.PLAGIAT_GRANUALRITY_TOKEN) {
			for(int l=0;l<marked_cells.length;l++) {
				double[] line = marked_cells[l];
				for(int c=0;c<line.length;c++) {
					if(line[c]>=IS_REACHABLE_CELL) {
						mark(marked_susp, l, k);//mark k tokens
						mark(marked_src, c, k);	
					}
				}
			}
		}else {
			System.err.println("init() not supported Config.PLAGIAT_GRANUALRITY "+Config.PLAGIAT_GRANUALRITY);
		}
	}

	private final void mark(final boolean[] marked_token, final int start, final int k) {
		for(int offset=start;offset<=start+k-1;offset++) {
			marked_token[offset] = true;
		}
	}
	
	private double[][] mark_cluster_seeds(double[][] matrix, final int k, final double core_threshold){
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
	
	private boolean is_core_cell(double[][] matrix, int line, int column, final double threshold) {
		double cell_similarity = matrix[line][column];
		if(cell_similarity>threshold) {
			return true;
		}else{
			return false;
		}
	}

	private double[][] expand_cluster_seeds(double[][] matrix, double[][] cluster_seeds){
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
	
	private void expand_cluster_seed_cell(final double[][] matrix, final double[][] cluster_seeds, final int line, final int column) {
		final int num_lines = matrix.length;
		final int num_columns = matrix[0].length;
		
		for(int l=Math.max(line-1, 0);l<Math.min(line+1,num_lines-1);l++) {
			for(int c=Math.max(column-1, 0);c<Math.min(column+1,num_columns-1);c++) {
				double similarity = matrix[l][c];
				double current_label = cluster_seeds[l][c];
				if(similarity>=REACHABILITY_CELL_THRESHOLD && current_label!=IS_CORE_CELL && current_label!=IS_REACHABLE_CELL) {
					cluster_seeds[l][c] = IS_REACHABLE_CELL;
					//expand_cluster_seed_cell(matrix, cluster_seeds, l, c);
				}
			}
		}
	}	
	
	private double gran(boolean[] marked_lines, int susp_ground_truth_from, int susp_ground_truth_to) {		
		double count = 0.0d;
		boolean was_found = false;
		
		for(int i=susp_ground_truth_from;i<susp_ground_truth_to;i++) {
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

	private int[] get_base_line(String name) {
		int[] base_line = Jaccard.materialized_ground_truth.get(name);
		if(base_line==null) {
			System.err.println("PanMetrics.get_base_line(String name) no such baseline for "+name);
		}
		return base_line;
	}
}
