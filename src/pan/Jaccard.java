package pan;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;

import boss.lexicographic.Tokenizer;
import boss.util.Config;
import boss.util.Util;
import plus.data.Book;

public class Jaccard {
	public final Book b_1;
	public final Book b_2;
	final int language = Book.LANGUAGE_ENGLISH;//XXX Assumption for PAN corpus only
	
	public static final int print_nothing 		= -1;
	public static final int print_final_token 	= 0;
	public static final int print_everything 	= 1;
	/**
	 * -1 print nothing
	 * 0 print final tokens
	 * 1 print everything
	 */
	public static int verbose_level = 1;
	/**
	 * Suspicious document as Text
	 */
	public final ArrayList<String> text_tokens_b_1;
	/**
	 * Source document as Text
	 */
	public final ArrayList<String> text_tokens_b_2;
	/**
	 * Suspicious document as integer code
	 */
	public final int[] int_tokens_b_1;
	/**
	 * Source document as integer code
	 */
	public final int[] int_tokens_b_2;
	
	
	public Jaccard(ArrayList<Book> books){
		this(books.get(0), books.get(1));
	}
	
	public Jaccard(ArrayList<Book> books, HashMap<String, Integer> token_ids){
		this(books.get(0), books.get(1), token_ids);
	}
	
	public Jaccard(Book b_1, Book b_2){
		this.b_1 = b_1;
		this.b_2 = b_2;
		
		text_tokens_b_1 = tokenize(b_1);
		//if(verbose_level>=print_final_token) {System.out.println("final B1\t\t"+Util.outTSV(text_tokens_b_1));}
		int_tokens_b_1 = new int[text_tokens_b_1.size()];
		text_tokens_b_2 = tokenize(b_2);
		//if(verbose_level>=print_final_token) {System.out.println("final B2\t\t"+Util.outTSV(text_tokens_b_2));}
		int_tokens_b_2 = new int[text_tokens_b_2.size()];
	}
	
	public Jaccard(Book b_1, Book b_2, HashMap<String, Integer> token_ids){
		this(b_1, b_2);
		encode(text_tokens_b_1, int_tokens_b_1, token_ids);
		encode(text_tokens_b_2, int_tokens_b_2, token_ids);
		
		if(verbose_level>=print_final_token) System.out.println("final susp\t\t"+Util.outTSV(text_tokens_b_1));
		if(verbose_level>=print_final_token) System.out.println("id's susp\t\t"+Util.outTSV(int_tokens_b_1));
		if(verbose_level>=print_final_token) System.out.println("final src\t\t"+Util.outTSV(text_tokens_b_2));
		if(verbose_level>=print_final_token) System.out.println("id's src\t\t"+Util.outTSV(int_tokens_b_2));
	}
	
	void encode(final ArrayList<String> text_tokens, final int[] int_tokens, HashMap<String, Integer> token_ids){
		for(int pos=0;pos<text_tokens.size();pos++) {
			String s = text_tokens.get(pos);
			Integer token_id = token_ids.get(s);
			if(token_id==null) {
				System.err.println("Jaccard.encode() token_id==null for "+s);
				int_tokens[pos] = -1;
			}else{
				int_tokens[pos] = token_id.intValue();
			}
		}
	}
	
	ArrayList<String> tokenize(Book b){
		String org = b.to_single_line_string();
		if(verbose_level>=print_everything) {System.out.println("org=\t\t\t"+org);}
		String temp = Tokenizer.replace_non_alphabetic_characters(org, language);
		if(verbose_level>=print_everything) {System.out.println("alpha numeric=\t\t"+temp);}
		temp = Tokenizer.remove_duplicate_whitespaces(temp);
		if(verbose_level>=print_everything) {System.out.println("whitespaces=\t\t"+temp);}
		String[] tokens = temp.split(" ");
		
		tokens = Tokenizer.to_lower_case(tokens);
		//if(verbose_level>=print_everything) System.out.println("to_lower_case\t\t"+Util.outTSV(tokens)); 
		//tokens = Tokenizer.stop_word_removal(tokens, language);//sometimes stop words are modified upon stemming. Thus, check it twice.
		if(verbose_level>=print_everything) System.out.println("stop_word_removal\t"+Util.outTSV(tokens));
		tokens = Tokenizer.stemming(tokens, language);
		if(verbose_level>=print_everything) System.out.println("stemming\t\t"+Util.outTSV(tokens));
		tokens = Tokenizer.stop_word_removal(tokens, language);
		if(verbose_level>=print_everything) System.out.println("stop_word_removal\t"+Util.outTSV(tokens));
		
		//TODO check for empty Strings, remove nulls
		ArrayList<String> ret = new ArrayList<String>();
		for(String s : tokens) {
			if(s != null) {
				if(s.length()!=0) {
					ret.add(s);	
				}
			}
		}
		
		return ret;
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
	
	/**
	 * 
	 * @param k window size
	 * @return alignment matrix A
	 */
	public double[][] jaccard_windows(final int k){
		System.out.println("jaccard_windows("+k+")");		
		
		final int[][] k_with_windows_b1 = create_windows(int_tokens_b_1, k);
		final int[][] k_with_windows_b2 = create_windows(int_tokens_b_2, k);
		
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
	
	/**
	 * 
	 * @param tokens_t1 window of the suspicious document
	 * @param tokens_t2 window of the src document
	 * @return exact jaccard similarity
	 */
	double jaccard(final int[] tokens_t1, final int[] tokens_t2) {
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
		
		/*if(jaccard_sim>=0.8) {
			System.out.println(Arrays.toString(tokens_t1));
			System.out.println(Arrays.toString(tokens_t2));
		}*/
		
		return jaccard_sim;
	}
	
	/**
	 * get_ground_truth[0] = start token of the plagiat in susp (i.e, Book 1)
	 * get_ground_truth[1] = inclusive stop token of the plagiat in susp (i.e, Book 1)
	 * get_ground_truth[2] = start token of the plagiat in src (i.e, Book 2)
	 * get_ground_truth[3] = inclusive stop token of the plagiat in src (i.e, Book 2) 
	 */
	public int[] ground_truth = null;
	/**
	 * 
	 * @param plaggiarism_excert excerpt the PAN Benchmark considers a plagiarism case (at character level)
	 * @return array with offsets where plagiarism starts at token level
	 */
	public int[] get_ground_truth(Jaccard plaggiarism_excert) {
		int[] offsets_b1 = find(this.int_tokens_b_1, plaggiarism_excert.int_tokens_b_1);
		int[] offsets_b2 = find(this.int_tokens_b_2, plaggiarism_excert.int_tokens_b_2);
		
		int[] ground_truth = {offsets_b1[0],offsets_b1[1],offsets_b2[0],offsets_b2[1]}; 
		this.ground_truth = ground_truth;
		
		if(verbose_level>=print_final_token) System.out.println("ground truth\t"+Util.outTSV(ground_truth));
		
		return ground_truth;
	}
	
	public static int[] get_ground_truth(int[] int_tokens_b_1, int[] int_tokens_b_2, int[] int_tokens_excerpt_1, int[] int_tokens_excerpt_2) {
		int[] offsets_b1 = find(int_tokens_b_1, int_tokens_excerpt_1);
		int[] offsets_b2 = find(int_tokens_b_2, int_tokens_excerpt_2);
		
		int[] ground_truth = {offsets_b1[0],offsets_b1[1],offsets_b2[0],offsets_b2[1]}; 
		
		return ground_truth;
	}

	private static int[] find(int[] array, int[] sub_array) {
		for(int i=0;i<array.length;i++) {//TODO - sub_array.length
			if(array[i]==sub_array[0]) { //found possible start
				if(find(array, sub_array, i)) {
					int[] temp = {i,i+sub_array.length-1};
					return temp;
				}
			}
		}
		System.err.println("Jaccard.find(int[],int[]) Did not find the excerpt");
		return null;
	}
	private static boolean find(int[] array, int[] sub_array, final int offset) {
		for(int i=0;i<sub_array.length;i++) {
			if(array[offset+i]!=sub_array[i]) {
				return false;
			}
		}
		return true;
	}
	
	/**
	 * 
	 */
	public static HashMap<String, int[]> materialized_ground_truth = null; 
}
