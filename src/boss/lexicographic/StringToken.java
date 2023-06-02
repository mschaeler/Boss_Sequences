package boss.lexicographic;

import plus.lexicographic.Token;

public class StringToken extends Token {
	public final String raw_token;
	
	public StringToken(String _raw_token) {
		this.raw_token = _raw_token;
	}

	/**
	 * Checks whether the raw_tokens are the same.
	 * @param other_token
	 * @return
	 */
	public boolean matches(Token other_token) {
		if(other_token instanceof StringToken) {
			if(this==other_token) {
				System.err.println("StringToken.matches(Token): Both tokens have same reference, returning false");
				return false;
			}
			return this.raw_token.equals(((StringToken)other_token).raw_token);//We need to cast ...
		}else{
			System.err.println("StringToken.matches(Token): Wrong Token types. Expecting StringToken, but got "+other_token.getClass().getCanonicalName()+" Returning false.");
			return false;
		}
	}
	
	@Override
	public double distance(Token other_token) {
		if(other_token instanceof StringToken) {
			if(this==other_token) {
				System.err.println("StringToken.distance(Token): Both tokens have same reference, returning Double.POSITIVE_INFINITY");
				return Double.POSITIVE_INFINITY;
			}
			StringToken t = (StringToken) other_token;
			double distance = edit_dist(raw_token, t.raw_token, raw_token.length(), t.raw_token.length());
			//normalize distance to [0,1] by dividing by the longest token length
			distance /= Math.max(this.raw_token.length(), t.raw_token.length());
			return distance;
		}else{
			System.err.println("StringToken.distance(Token): Wrong Token types. Expecting StringToken, but got "+other_token.getClass().getCanonicalName()+" Returning Double.POSITIVE_INFINITY.");
			return Double.POSITIVE_INFINITY;
		}
	}
	
	public static int edit_dist(String s1, String s2){
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
	
	public String toString() {
		return raw_token;
	}
	
	static final int min(final int x, final int y, final int z){
        if (x <= y && x <= z)
            return x;
        if (y <= x && y <= z)
            return y;
        else
            return z;
    }
}
