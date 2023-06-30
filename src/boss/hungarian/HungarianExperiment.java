package boss.hungarian;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;

public class HungarianExperiment {
	final int num_paragraphs;
	final int k;
	final double threshold;
	
	final ArrayList<int[][]> k_with_windows_b1;
	final ArrayList<int[][]> k_with_windows_b2;
	
	final ArrayList<int[]> raw_paragraphs_b1;
	final ArrayList<int[]> raw_paragraphs_b2;
	
	final ArrayList<double[][]> alignement_matrixes;
	
	final HashMap<Integer, double[]> embedding_vector_index;
	
	Solver solver = null;
	
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
		
		//out(raw_paragraphs_b1, k_with_windows_b1);
		//out(raw_paragraphs_b2, k_with_windows_b2);
		
		this.alignement_matrixes = new ArrayList<double[][]>(num_paragraphs);
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
	
	final boolean VERBOSE = true;
	public void run(){
		final double[][] cost_matrix = new double[k][k];
		if(this.solver==null) {
			System.err.println("Solver is null: Using StupidSolver");
			this.solver = new StupidSolver(k);
		}
			
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
					//Note it is cost matrix with cosine distance. I.e, not similarity. 
					fill_cost_matrix(current_window_p1,current_window_p2,cost_matrix);
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
			System.out.println("P="+p+"\t"+(stop-start)+"\tms");
			this.alignement_matrixes.add(alignment_matrix);
		}
		//Print result matrixes
		for(double[][] alignment_matrix : alignement_matrixes){
			out_matrix(alignment_matrix);
			int num_cells = alignment_matrix.length*alignment_matrix[0].length;
			System.out.println(num_cells);
		}
	}
	
	private void out_matrix(double[][] alignment_matrix) {
		System.out.println("Next matrix");
		for(double[] array : alignment_matrix) {
			String temp = outTSV(array);
			System.out.println(temp);
		}
	}

	private String outTSV(double[] array) {
		String s = ""+array[0];
		for(int i=1;i<array.length;i++) {
			s+="\t"+array[i];
		}
		return s;
	}

	private void fill_cost_matrix(final int[] k_window_p1, final int[] k_window_p2, final double[][] cost_matrix) {
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
	}
	
	public static final double EQUAL = 0;
	public static final double MAX_DIST = 1.0;
	final double dist(final int set_id1, final int set_id2, final double[] vec_1, final double[] vec_2) {
		if(set_id1==set_id2) {
			return EQUAL;
		}else if(vec_1==null || vec_2==null){//may happen e.g., for stop words
			return MAX_DIST;
		}
		return cosine_distance(vec_1, vec_2);
	}

	//Optional TODO - normalize all vectors to length = 1. Then, computation is much simpler.
	static double cosine_distance(final double[] vectorA, final double[] vectorB) {
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
