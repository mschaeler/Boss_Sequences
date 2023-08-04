package boss.embedding;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;

public class Match implements Comparable<Match>{
	/**
	 * s1 is from embedding, s2 from the text
	 * s2 is mapped to exactly one s1, but multiple s2 may map to the same s1
	 */
	public final String string_in_embedding;
	public final String string_in_text;
	/**
	 * [0,1] - 0 no similarity, 1 identic.
	 */
	final double score;
	
	//double[] vector;
	/**
	 * 
	 * @param s1 is from embedding
	 * @param s2 s2 from the text
	 * @param score
	 */
	public Match(String string_in_embedding, String string_in_text, double score){
		this.string_in_embedding = string_in_embedding;
		this.string_in_text 	 = string_in_text;
		this.score = score;
	}
	@Override
	public int compareTo(Match arg0) {
		return string_in_embedding.compareTo(arg0.string_in_embedding);
	}
	
	public String toString(){
		return string_in_embedding+"\t"+string_in_text+"\t"+score;	
	}
	
	public static HashSet<String> all_required_entries(Match[] matches) {
		HashSet<String> set = new HashSet<String>(matches.length);
		for(Match m : matches) {
			String word_in_embedding = m.string_in_embedding;
			set.add(word_in_embedding);
		}
		return set;
	}
}
