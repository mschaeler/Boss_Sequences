package boss.token_mapping;

import java.util.ArrayList;
import java.util.HashSet;

import boss.lexicographic.StringToken;
import boss.semantic.SemanticToken;

/**
 * This class represents any dictionary that contains all entries found in a dictionary, word embedding or similar.
 * It is basically used to transform lexicographic Tokens into SemanticTokens.
 * The Lexicographical Token is named full form, the corresponding Semantic Entry baseform.
 * 
 * @author b1074672
 *
 */
public abstract class MyDictionary {
	/**
	 * Enum e.g., Book.LANGUAGE_OLD_GREEK
	 */
	int language;
	HashSet<String> index 		= null;	
	ArrayList<String> all_words = null;
	
	public String map_to_word(StringToken t) {
		String map_me = t.raw_token;
		String mapped_to = map_to_word_exact(map_me);
		if(mapped_to==null) {//We did not find an exact match. Go for the best match
			mapped_to = map_to_word_best_match(map_me);
		}
		//TODO maybe, if the best match is not good enough consider falling back to String edit distance.
		return mapped_to;
	}
	
	private String map_to_word_exact(String map_me) {
		if(index.contains(map_me)) {
			return map_me;//basefom == fullform
		}else{
			return null;  //No exact match. Probably need to go for best match
		}
	}
	
	/**
	 * Finds best match by longest common prefix in this.all_words
	 * @param map_me
	 * @return
	 */
	private String map_to_word_best_match(String map_me) {
		int max_similarity = Integer.MIN_VALUE;
		String best_match = null;
		
		for(String s : this.all_words) {
			int common_prefix = common_prefix(s, map_me);
			if(common_prefix > max_similarity) {
				max_similarity = common_prefix;
				best_match = s;
			}
		}
		return best_match;
	}
	
	/**
	 * Finds best match of the provided key in the dictionary or word embedding having 
	 * Longest common prefix. We use this method to match lexicographic tokens to their semantic
	 * 
	 * @param key
	 * @return
	 */
	protected final int common_prefix(final String s1, final String s2) {
		int length_prefix = 0;
		for(int i=0;i<Math.min(s1.length(), s2.length());i++) {
			if(s1.charAt(i)==s2.charAt(i)) {
				length_prefix++;
			}else {
				return length_prefix;
			}
		}
		return length_prefix;
	}
	
	public abstract SemanticToken map_to_semantic(StringToken t);//TODO if basefom != null and internal() function?
}
