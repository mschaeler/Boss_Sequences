package boss.semantic;

import plus.lexicographic.Token;

public abstract class SemanticToken extends Token implements Comparable<SemanticToken>{
	/** The word as found in the text (after pre-processing)*/
	public String fullform;
	/** The word as found in a dictionary or word embedding*/
	public String baseform;
	
	@Override
	public boolean matches(Token other_token) {
		if(other_token instanceof SemanticToken) {
			return baseform.equals(((SemanticToken) other_token).baseform);
		}else{
			return false;
		}
	}
	
	@Override
	public int compareTo(SemanticToken arg0) {
		return baseform.compareTo(arg0.baseform);//By their lexicographical order
	}
}
