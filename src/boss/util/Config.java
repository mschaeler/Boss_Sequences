package boss.util;

public interface Config {
	/**
	 * TODO
	 */
	int BYTES_PER_DOUBLE = 8;
	/**
	 * TODO
	 */
	boolean JACCARC_COLLAPSE_SET_IDS = true;
	
	boolean REMOVE_STOP_WORDS = true;
	
	/**
	 * German stop words
	 */
	int DE_STOP_WORDS = 0;
	/**
	 * English stop words
	 */
	int EN_STOP_WORDS = 1;
	/**
	 * Stop words of Dong Deng's 2023 SIGMOD Paper
	 */
	int DD_STOP_WORDS = 2;
	/**
	 * ENUM for the used stop words list : DE_STOP_WORDS, EN_STOP_WORDS, or DD_STOP_WORDS from boss.lexicographic.StopWords class
	 */
	int USED_STOP_WORD_LIST = 2;
	
	/**
	 * Window sizes.
	 */
	int[] k_s= {3,4,5,6,7,8,9,10,11,12,13,14,15};
}
