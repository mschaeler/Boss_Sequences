package pan;

import java.util.ArrayList;
import java.util.Arrays;
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
	
	final Solutions s;
	
	public PanResult(Solutions s) {
		this.s = s;
		all_results[s.k].add(this);
	}
	
	double[][] jaccard_windows(){
		double[][] matrix = new double[s.k_with_windows_b1.length][s.k_with_windows_b2.length];
		for(int row=0;row<matrix.length;row++) {
			int[] w_r = s.k_with_windows_b1[row];
			for(int colum=0;colum<matrix[0].length;colum++) {
				int[] w_c = s.k_with_windows_b2[colum];
				double jaccard_sim = jaccard(w_r, w_c);
				if(jaccard_sim>0.2) {//XXX the threshold works here differently
					matrix[row][colum] = jaccard_sim;
				}
			}
		}
		return matrix;
	}
	
	double jaccard_all_text() {
		int[] tokens_t1 = s.tokens_b1;//clone, if you change them
		int[] tokens_t2 = s.tokens_b2;
		
		double jaccard_sim = jaccard(tokens_t1, tokens_t2);
		
		return jaccard_sim;
	}
	
	double jaccard(int[] tokens_t1, int[] tokens_t2) {
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
	
	/**
	 * Computes Hamming distance of the two arguments exploiting that they are ordered. Thus, we use a merge algorithm.
	 * @param left - ordered set
	 * @param right - ordered set
	 * @return
	 */
	int hamming_distance(int[] left, int[] right) {
		int different_elements = 0;
		int pntr_left = 0;
		int pntr_right = 0;
		
		while(true) {
			if(left[pntr_left]==right[pntr_right]) {//same element
				pntr_left++;
				pntr_right++;
			}else if(left[pntr_left]<right[pntr_right]) {//element in left, but not in right
				pntr_left++;
				different_elements++;
			}else {//element in right, but not in left
				pntr_right++;
				different_elements++;
			}
			if(pntr_left == left.length && pntr_right == right.length) {
				//both ends reached, no remaining elements, simply break
				return different_elements;
			}else if(pntr_left == left.length) {
				// reached end of left, all remaining elements in right are different 
				different_elements += right.length-pntr_right;
				return different_elements;
			}else if(pntr_right == right.length) {
				// reached end of right, all remaining elements in left are different
				different_elements += left.length-pntr_left;
				return different_elements;
			}//else simply continue
		}
	}
	
	double sum() {
		return s.sum(s.alignement_matrix);
	}
	
	double num_cells() {
		return (double) s.alignement_matrix.length*s.alignement_matrix[0].length;
	}
	
	double avg_cell_similarity() {
		double sim = sum() / num_cells();
		return sim;
	}
	
	double num_colums_marked_as_similar(double threshold) {
		final int num_columns = s.alignement_matrix[0].length;
		int count_column_marked_as_similar = 0;
		
		for(int column=0;column<num_columns;column++) {
			for(double[] row : s.alignement_matrix) {
				double d = row[column];
				if(d>=threshold) {
					count_column_marked_as_similar++;
					break;//Count each row only once
				}
			}
		}
		return (double) count_column_marked_as_similar;
	}
	
	double fraction_colums_marked_as_similar(double threshold) {
		final int num_columns = s.alignement_matrix[0].length;
		return num_colums_marked_as_similar(threshold) / (double) num_columns;
	}

	double num_rows_marked_as_similar(double threshold) {
		int count_rows_marked_as_similar = 0;
		for(double[] row : s.alignement_matrix) {
			for(double d : row) {
				if(d>=threshold) {
					count_rows_marked_as_similar++;
					break;//Count each row only once
				}
			}
		}
		return (double) count_rows_marked_as_similar;
	}
	
	double fraction_rows_marked_as_similar(double threshold) {
		return num_rows_marked_as_similar(threshold) / (double) s.alignement_matrix.length;
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
	
	public void out_my_matrices() {
		System.out.println("*******************Semantic alignment vs. Jaccard alignment");
		double[][] jaccard_matrix = jaccard_windows();
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
		return s.k+"\t"+num_cells()+"\t"+num_rows()+"\t"+sum()+"\t"+avg_cell_similarity()
			+"\t"+fraction_rows_marked_as_similar(0.5)
			+"\t"+fraction_rows_marked_as_similar(0.6)
			+"\t"+fraction_rows_marked_as_similar(0.7)
			+"\t"+fraction_rows_marked_as_similar(0.8)
			+"\t"+fraction_rows_marked_as_similar(0.9)
			+"\t"+jaccard_all_text();
	}
	
	private int num_rows() {
		return s.alignement_matrix.length;
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
		System.out.println(result_header());
		for(ArrayList<PanResult> all_results_k : all_results) {
			if(!all_results_k.isEmpty()) {
				double num_cells = 0;
				double num_rows = 0;
				double sum = 0;
				double r_sim_05 = 0;
				double r_sim_06 = 0;
				double r_sim_07 = 0;
				double r_sim_08 = 0;
				double r_sim_09 = 0;
				double jaccard = 0;
				final double size = all_results_k.size();
				
				for(PanResult pr : all_results_k) {
					num_cells += pr.num_cells();
					num_rows += pr.num_rows();
					sum += pr.sum();
					r_sim_05 += pr.num_colums_marked_as_similar(0.5);
					r_sim_06 += pr.num_colums_marked_as_similar(0.6);
					r_sim_07 += pr.num_colums_marked_as_similar(0.7);
					r_sim_08 += pr.num_colums_marked_as_similar(0.8);
					r_sim_09 += pr.num_colums_marked_as_similar(0.9);
					jaccard += pr.jaccard_all_text();
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
				
				System.out.println(all_results_k.get(0).s.k+"\t"+num_cells+"\t"+num_rows+"\t"+sum+"\t"+(num_cells/sum/size)
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
				);
			}
		}
		//all_results.clear();
	}

	public static void clear() {
		for( ArrayList<PanResult> a : all_results) {
			a.clear();
		}
	}
}
