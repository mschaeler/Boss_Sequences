package boss.embedding;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;

public class Match implements Comparable<Match>{
	/**
	 * s1 is from embedding, s2 from the text
	 * s2 is mapped to exactly one s1, but multiple s2 may map to the same s1
	 */
	final String s1, s2;
	/**
	 * [0,1] - 0 no similarity, 1 identic.
	 */
	final double score;
	
	double[] vector;
	public Match(String s1, String s2, double score){
		this.s1 = s1;
		this.s2 = s2;
		this.score = score;
	}
	@Override
	public int compareTo(Match arg0) {
		return s1.compareTo(arg0.s1);
	}
	
	public String toString(){
		return s1+"\t"+s2+"\t"+score;	
	}
	
	public static HashSet<String> all_required_entries(Match[] matches) {
		HashSet<String> set = new HashSet<String>(matches.length);
		for(Match m : matches) {
			String word_in_embedding = m.s1;
			set.add(word_in_embedding);
		}
		return set;
	}
}
