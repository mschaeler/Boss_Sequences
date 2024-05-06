package boss.util;

public class Config {
	/**
	 * TODO
	 */
	public static final int BYTES_PER_DOUBLE = 8;
	/**
	 * TODO
	 */
	public static boolean JACCARC_COLLAPSE_SET_IDS = true;
	
	public static boolean REMOVE_STOP_WORDS = true;
	
	public static boolean REMOVE_NUMBERS = true;
	
	public static boolean STEM_WORDS = true;
	
	/**
	 * German stop words
	 */
	public static final int DE_STOP_WORDS = 0;
	/**
	 * English stop words
	 */
	public static final int EN_STOP_WORDS = 1;
	/**
	 * Stop words of Dong Deng's 2023 SIGMOD Paper
	 */
	public static final int DD_STOP_WORDS = 2;
	/**
	 * ENUM for the used stop words list : DE_STOP_WORDS, EN_STOP_WORDS, or DD_STOP_WORDS from boss.lexicographic.StopWords class
	 */
	public static int USED_STOP_WORD_LIST = 2;
	
	/**
	 * Basically a cell may contain some tokens belonging to a plagiarism case and others that do not. 
	 * If this variable is set to true, a cell is deemed a true-positive palgiat, iff all tokens in it are part of the plagiat.
	 * If this variable is set to false, a cell is deemed a true-positive palgiat, iff at least one tokens in it is part of the plagiat. 
	 * Applies to Jaccard only. For SeDA it is always true.
	 */
	public static boolean USE_FULL_PLAGIAT_CELLS = true;
	
	/**
	 * Window sizes.
	 */
	//public static int[] k_s= {3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32};
	public static int[] k_s= {3,4,5,6,7,8,9,10,11,12,13,14,15,16};
	//public static int[] k_s= {3,4,5,6,7,8,9,10,11,12,13,14,15};
	//public static int[] k_s= {8};
}
