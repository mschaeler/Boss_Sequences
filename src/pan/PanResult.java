package pan;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

import boss.hungarian.Solutions;

public class PanResult {
	@SuppressWarnings("unchecked")
	/**
	 * all_results[k]
	 */
	public static ArrayList<PanResult>[] all_results = new ArrayList[17];
	static {
		for(int i=0;i<all_results.length;i++) {
			all_results[i] = new ArrayList<PanResult>(); 
		}
	}
	
	//final Solutions s;
	
	final double sum;
	final double num_cells;
	final double avg_cell_similarity;
	final double num_rows;
	final double num_cols;
	
	final int[] tokens_t1; 
	final int[] tokens_t2; 
	final int k; 
	final int[][] k_with_windows_b1; 
	final int[][] k_with_windows_b2;
	
	HashMap<Double, Double> num_colums_marked_as_similar = new HashMap<Double, Double>();
	HashMap<Double, Double> num_rows_marked_as_similar = new HashMap<Double, Double>();
	HashMap<Double, Double> num_cluster_rows = new HashMap<Double, Double>();
	/**
	 * Number of clusters. Ideally, it is one.
	 */
	HashMap<Double, Double> granularity = new HashMap<Double, Double>();
	
	public static double connectivity_threshold = 0.5;
	
	public PanResult(Solutions s) {
		this(s, s.alignement_matrix);
	}
	
	public PanResult(Solutions s, double[][] matrix) {
		//this.s = s;
		all_results[s.k].add(this);
		
		this.sum = sum(matrix);
		this.num_cells = num_cells(matrix);
		this.avg_cell_similarity = avg_cell_similarity(matrix);
		this.num_rows = matrix.length;
		this.num_cols = matrix[0].length;
		
		this.tokens_t1 = s.tokens_b1.clone();
		this.tokens_t2 = s.tokens_b2.clone();
		this.k = s.k;
		this.k_with_windows_b1 = s.k_with_windows_b1.clone();
		this.k_with_windows_b2 = s.k_with_windows_b2.clone();
		
		double[] thresholds = {0.5,0.6,0.7,0.8,0.9};
		for(double t : thresholds) {
			num_rows_marked_as_similar.put(t, num_rows_marked_as_similar(t, matrix));
			//XXX num_colums_marked_as_similar.put(t, num_colums_marked_as_similar(t, s.alignement_matrix));
			num_cluster_rows.put(t, num_cluster_rows(matrix, t, connectivity_threshold));
		}
	}

	public static double[][] jaccard_windows(int[][] k_with_windows_b1, int[][] k_with_windows_b2){
		double[][] matrix = new double[k_with_windows_b1.length][k_with_windows_b2.length];
		for(int row=0;row<matrix.length;row++) {
			int[] w_r = k_with_windows_b1[row];
			for(int colum=0;colum<matrix[0].length;colum++) {
				int[] w_c = k_with_windows_b2[colum];
				double jaccard_sim = jaccard(w_r, w_c);
				if(jaccard_sim>0.2) {//XXX the threshold works here differently
					matrix[row][colum] = jaccard_sim;
				}
			}
		}
		return matrix;
	}
	
	double[][] jaccard_windows(){
		return jaccard_windows(this.k_with_windows_b1, this.k_with_windows_b2);
	}
	
	double jaccard_all_text() {
		double jaccard_sim = jaccard(tokens_t1, tokens_t2);
		return jaccard_sim;
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
		
		return jaccard_sim;
	}
	
	double sum(double[][] matrix) {
		double sum = 0;
		for(double[] array : matrix) {
			for(double d : array) {
				sum+=d;
			}
		}
		return sum;
	}
	
	double sum() {
		return this.sum;
	}
	
	double num_cells(double[][] matrix) {
		return matrix.length*matrix[0].length;
	}
	
	double num_cells() {
		return this.num_cells;
	}
	
	double avg_cell_similarity() {
		double sim = this.sum / this.num_cells;
		return sim;
	}
	
	double avg_cell_similarity(double[][] matrix) {
		double sim = sum(matrix) / num_cells(matrix);
		return sim;
	}

	double num_colums_marked_as_similar(double threshold, double[][] matrix) {
		final int num_columns = matrix[0].length;
		int count_column_marked_as_similar = 0;
		
		for(int column=0;column<num_columns;column++) {
			for(double[] row : matrix) {
				double d = row[column];
				if(d>=threshold) {
					count_column_marked_as_similar++;
					break;//Count each row only once
				}
			}
		}
		return (double) count_column_marked_as_similar;
	}
	
	double num_colums_marked_as_similar(double threshold) {
		Double num = num_colums_marked_as_similar.get(threshold);
		if(num == null) {
			System.err.println("num_colums_marked_as_similar(double) non saved threshold "+threshold);
			return Double.NEGATIVE_INFINITY;
		}
		return num.doubleValue();
	}
	
	double fraction_colums_marked_as_similar(double threshold) {
		return num_colums_marked_as_similar(threshold) / (double) num_cols;
	}

	double num_rows_marked_as_similar(double threshold, double[][] matrix) {
		int count_rows_marked_as_similar = 0;
		for(double[] row : matrix) {
			for(double d : row) {
				if(d>=threshold) {
					count_rows_marked_as_similar++;
					break;//Count each row only once
				}
			}
		}
		if(count_rows_marked_as_similar>num_rows) {
			System.err.println(count_rows_marked_as_similar>num_rows);
		}
		return (double) count_rows_marked_as_similar;
	}
	
	double num_cluster_rows(double threshold) {
		Double num = num_cluster_rows.get(threshold);
		if(num == null) {
			System.err.println("num_cluster_rows(double) non saved threshold "+threshold);
			return Double.NEGATIVE_INFINITY;
		}
		return num.doubleValue();
	}
	
	double num_rows_marked_as_similar(double threshold) {
		Double num = num_rows_marked_as_similar.get(threshold);
		if(num == null) {
			System.err.println("num_rows_marked_as_similar(double) non saved threshold "+threshold);
			return Double.NEGATIVE_INFINITY;
		}
		return num.doubleValue();
	}
	
	double fraction_rows_marked_as_similar(double threshold, double[][] matrix) {
		return num_rows_marked_as_similar(threshold, matrix) / (double) matrix.length;
	}
	double fraction_rows_marked_as_similar(double threshold) {
		return num_rows_marked_as_similar(threshold) / (double) num_rows;
	}
	
	double precision() {
		return -1;//TODO
	}
	
	double recall() {
		return -1;//TODO
	}
	
	/*public static void clear_results() {
		all_results.clear();
	}*/
	
	public void out_matrix() {
		//TODO
	}
	
	public static void out_my_matrices(Solutions s) {
		System.out.println("*******************Semantic alignment vs. Jaccard alignment");
		double[][] jaccard_matrix = new PanResult(s).jaccard_windows();
		System.out.println("k="+s.k);
		for(int row=0;row<s.alignement_matrix.length;row++) {
			//
			for(int column=0;column<s.alignement_matrix[0].length;column++) {
				System.out.print(s.alignement_matrix[row][column]+"\t");
			}
			System.out.print("\t\t\t\t");//Some tabs to optically separate them
			for(int column=0;column<s.alignement_matrix[0].length;column++) {
				System.out.print(jaccard_matrix[row][column]+"\t");
			}
			System.out.println();
		}
	}
	
	public static String result_header() {
		return "k\tnum cells\tnum rows\tsum\tavg_cell_similarity\trecal_0.5\trecal_0.6\trecal_0.7\trecal_0.8\trecal_0.9\tJaccard text";
	}
	
	/*public String toString() {
		return s.k+"\t"+num_cells()+"\t"+sum()+"\t"+avg_cell_similarity()
			+"\t"+fraction_colums_marked_as_similar(0.5)+"\t"+fraction_rows_marked_as_similar(0.5)
			+"\t"+fraction_colums_marked_as_similar(0.6)+"\t"+fraction_rows_marked_as_similar(0.6)
			+"\t"+fraction_colums_marked_as_similar(0.7)+"\t"+fraction_rows_marked_as_similar(0.7)
			+"\t"+fraction_colums_marked_as_similar(0.8)+"\t"+fraction_rows_marked_as_similar(0.8)
			+"\t"+fraction_colums_marked_as_similar(0.9)+"\t"+fraction_rows_marked_as_similar(0.9)
			+"\t"+jaccard_all_text();
	}*/

	public String toString() {
		return k+"\t"+num_cells()+"\t"+num_rows()+"\t"+sum()+"\t"+avg_cell_similarity()
			+"\t"+fraction_rows_marked_as_similar(0.5)
			+"\t"+fraction_rows_marked_as_similar(0.6)
			+"\t"+fraction_rows_marked_as_similar(0.7)
			+"\t"+fraction_rows_marked_as_similar(0.8)
			+"\t"+fraction_rows_marked_as_similar(0.9)
			+"\t"+jaccard_all_text();
	}
	
	private double num_rows() {
		return this.num_rows;
	}

	public static void out() {
		for(ArrayList<PanResult> all_results_k : all_results) {
			if(!all_results_k.isEmpty()) {
				System.out.println("k\tnum cells\tnum rows\tsum\tavg_cell_similarity\trecal_0.5\trecal_0.6\trecal_0.7\trecal_0.8\trecal_0.9\tJaccard text\trecal_0.5\trecal_0.6\trecal_0.7\trecal_0.8\trecal_0.9");
				for(PanResult pr : all_results_k) {
					System.out.println(pr);
				}
				System.out.println();
			}
		}
		//all_results.clear();
	}
	
	public static void out_agg() {
		System.out.println("k\tnum cells\tnum rows\tnum cols\tsum\tavg_cell_similarity"
				+ "\trecal_0.5\trecal_0.6\trecal_0.7\trecal_0.8\trecal_0.9\tJaccard text"
				+ "\trecal_0.5\trecal_0.6\trecal_0.7\trecal_0.8\trecal_0.9"
				+ "\t#cluster_0.5\t#cluster_0.6\t#cluster_0.7\t#cluster_0.8\t#cluster_0.9"
				+ "\tcluster_0.5\tcluster_0.6\tcluster_0.7\tcluster_0.8\tcluster_0.9"
		);
		for(ArrayList<PanResult> all_results_k : all_results) {
			if(!all_results_k.isEmpty()) {
				double num_cells = 0;
				double num_rows = 0;
				double num_cols = 0;
				double sum = 0;
				double r_sim_05 = 0;
				double r_sim_06 = 0;
				double r_sim_07 = 0;
				double r_sim_08 = 0;
				double r_sim_09 = 0;
				double jaccard = 0;
				final double size = all_results_k.size();
				
				double r_cluster_05 = 0;
				double r_cluster_06 = 0;
				double r_cluster_07 = 0;
				double r_cluster_08 = 0;
				double r_cluster_09 = 0;
				
				
				for(PanResult pr : all_results_k) {
					num_cells += pr.num_cells;
					num_rows += pr.num_rows;
					num_cols += pr.num_cols;
					sum += pr.sum;
					r_sim_05 += pr.num_rows_marked_as_similar(0.5);
					r_sim_06 += pr.num_rows_marked_as_similar(0.6);
					r_sim_07 += pr.num_rows_marked_as_similar(0.7);
					r_sim_08 += pr.num_rows_marked_as_similar(0.8);
					r_sim_09 += pr.num_rows_marked_as_similar(0.9);
					jaccard += pr.jaccard_all_text();
					r_cluster_05 += pr.num_cluster_rows(0.5);
					r_cluster_06 += pr.num_cluster_rows(0.6);
					r_cluster_07 += pr.num_cluster_rows(0.7);
					r_cluster_08 += pr.num_cluster_rows(0.8);
					r_cluster_09 += pr.num_cluster_rows(0.9);
				}
				if(num_rows<r_sim_05) {
					System.err.println(num_rows<r_sim_05);
				}
				//normalize
				/*num_cells /= size;
				sum /= size;
				avg_cell_similarity /= size;
				c_sim_05 /= size;
				r_sim_05 /= size;
				c_sim_06 /= size;
				r_sim_06 /= size;
				c_sim_07 /= size;
				r_sim_07 /= size;
				c_sim_08 /= size;
				r_sim_08 /= size;
				c_sim_09 /= size;
				r_sim_09 /= size;*/
				jaccard /= size;
				
				System.out.println(all_results_k.get(0).k+"\t"+num_cells+"\t"+num_rows+"\t"+num_cols+"\t"+sum+"\t"+(sum/num_cells)
						+"\t"+r_sim_05
						+"\t"+r_sim_06
						+"\t"+r_sim_07
						+"\t"+r_sim_08
						+"\t"+r_sim_09
						+"\t"+jaccard
						+"\t"+r_sim_05/num_rows
						+"\t"+r_sim_06/num_rows
						+"\t"+r_sim_07/num_rows
						+"\t"+r_sim_08/num_rows
						+"\t"+r_sim_09/num_rows
						+"\t"+r_cluster_05
						+"\t"+r_cluster_06
						+"\t"+r_cluster_07
						+"\t"+r_cluster_08
						+"\t"+r_cluster_09
						+"\t"+r_cluster_05/num_rows
						+"\t"+r_cluster_06/num_rows
						+"\t"+r_cluster_07/num_rows
						+"\t"+r_cluster_08/num_rows
						+"\t"+r_cluster_09/num_rows
				);
			}
		}
		//all_results.clear();
	}
	
	public double num_cluster_rows(double[][] matrix, double core_threshold, double connectivity_threshold) {
		double[] max_row_sim = new double[matrix.length];
		boolean[] cluster_rows = new boolean[matrix.length];
		for(int row=0;row<matrix.length;row++) {
			max_row_sim[row] = max(matrix[row]);
		}
		double count_clusters = 0;//TODO muss man speichern
		
		for(int row=0;row<matrix.length;row++) {
			//find seeds
			if(max_row_sim[row]>=core_threshold) {
				count_clusters++;
				cluster_rows[row] = true;
				//extend them
				int i=1;
				while(row-i>=0 && cluster_rows[row-i] == false && max_row_sim[row-i]>=connectivity_threshold) {
					cluster_rows[row-i] = true;
					i++;
				}
				row++;
				while(row<matrix.length && max_row_sim[row]>=connectivity_threshold) {
					cluster_rows[row] = true;
					row++;
				}
			}
		}
		
		double count = 0;
		for(int row=0;row<matrix.length;row++) {
			if(cluster_rows[row]) count++;
		}
		
		this.granularity.put(core_threshold, count_clusters);
		return count;
	}

	private static double max(double[] arr) {
		double max = Double.NEGATIVE_INFINITY;
		for(double d : arr) {
			if(d>max) {
				max = d;
			}
		}
		return max;
	}

	public static void clear() {
		for( ArrayList<PanResult> a : all_results) {
			a.clear();
		}
	}
	
	//TODO originale Dokumente längen
	
	/**
	 * Metrics according to https://aclanthology.org/C10-2115.pdf
	 * @return
	 */
	/**
	 * Metrics according to https://aclanthology.org/C10-2115.pdf
	 * @param pan
	 * @param threshold
	 * @return
	 */
	public static double precision(PanResult pan, double threshold) {
		double nominator = pan.num_cluster_rows(threshold);
		return nominator/pan.num_rows;
	}
	public static double recall(PanResult pan, double threshold) {
		double nominator = pan.num_cluster_rows(threshold);
		return nominator/pan.num_cols;
	}
	/**
	 * Fragments detecting a plagiat - we want a 1:1 mapping 
	 * @param pan
	 * @return
	 */
	public static double gran(PanResult pan, double threshold) {
		Double gran = pan.granularity.get(threshold);
		if(gran==null) {
			System.err.println("gran(PanResult,"+threshold+") - no value for this threshold");
			return 0.0d;
		}
		return gran.doubleValue();
	}
	public static double plagdet(PanResult pan, double threshold) {
		return F(precision(pan, threshold), recall(pan, threshold)) / log_2(1+gran(pan, threshold));
	}

	/**
	 * Harmonic mean
	 * @return
	 */
	private static double F(double val_1, double val_2) {
		float sum = 0; 
	    sum += 1.0d / val_1;
	    sum += 1.0d / val_2;
	    return 2.0d / sum; 
	}

	private static double log_2(double d) {
		return Math.log(d) / Math.log(2.0d);
	}
}
