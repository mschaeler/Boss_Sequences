package boss.hungarian;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
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
	
	public boolean VERBOSE = false;
	public boolean TO_FILE = true;
	FileWriter f;
	BufferedWriter output;
	
	public void run(){
		System.out.println("HungarianExperiment.run() dist="+SIM_FUNCTION+" k="+k+" threshold="+threshold);
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
		int p = 0;
		if(TO_FILE) {
			String path = ".//results//"+System.currentTimeMillis()+"_sim()="+SIM_FUNCTION+"_k="+k+"_threshold="+threshold+"_"+this.solver.get_name()+".tsv";
			try {
				this.f = new FileWriter(path);
				output = new BufferedWriter(f);
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		for(double[][] alignment_matrix : alignement_matrixes){
			out_matrix(alignment_matrix, p++);
			int num_cells = alignment_matrix.length*alignment_matrix[0].length;
			System.out.println(num_cells);
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
		System.out.println("Next matrix");
		if(TO_FILE) {
			try {
				output.write("Next matrix "+p_id);
				output.newLine();
			} catch (IOException e) {
				e.printStackTrace();
			}
		}
		for(double[] array : alignment_matrix) {
			String temp = outTSV(array);
			if(VERBOSE) {
				System.out.println(temp);
			}
			if(TO_FILE) {
				try {
					output.write(temp);
					output.newLine();
				} catch (IOException e) {
					e.printStackTrace();
				}
			}
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
	
	public static final double EQUAL 	= 0;
	public static final double MAX_DIST = 1.0;
	
	public static final int COSINE 			= 0;
	public static final int STRING_EDIT 	= 1;
	public static final int VANILLA_OVERLAP = 3;
	public static final int SIM_FUNCTION = VANILLA_OVERLAP;
	
	@SuppressWarnings("unused")
	final double dist(final int set_id1, final int set_id2, final double[] vec_1, final double[] vec_2) {
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
	public static int edit_dist(final int set_id1, final int set_id2){
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
