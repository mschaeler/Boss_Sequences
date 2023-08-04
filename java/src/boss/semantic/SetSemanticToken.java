package boss.semantic;

import java.util.ArrayList;
import java.util.Collections;

import boss.lexicographic.StringToken;
import plus.lexicographic.Token;

public class SetSemanticToken extends SemanticToken{
	public ArrayList<String> semantics;
	
	public SetSemanticToken(StringToken st) {
		this(st, null);
	}
	
	public SetSemanticToken(StringToken st, ArrayList<String> semantics) {
		this(st.raw_token, semantics);
	}
	
	public SetSemanticToken(String fullform) {
		this(fullform, null);
	}
	
	public SetSemanticToken(String fullform, ArrayList<String> semantics) {
		super.fullform = fullform;
		super.baseform = null;
		this.semantics = semantics;
		Collections.sort(semantics);
	}
	
	public String toString() {
		String ret = fullform+"->("+baseform+"): [";
		for(String s : semantics) {
			ret+=s+", ";
		}
		ret+="]";
		return ret;
	}

	@Override
	public double distance(Token other_token) {
		// TODO Auto-generated method stub
		return 0;
	}
}
