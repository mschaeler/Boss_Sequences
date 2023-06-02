package plus.lexicographic;

public abstract class Token {
	public Token() {
		//Nothing to do
	}
	
	/**
	 * Checks whether the raw_tokens are the same.
	 * @param other_token
	 * @return
	 */
	public abstract boolean matches(Token other_token);
	/**
	 * Computes a distance value between this Token and the one provided as argument.
	 * @param other_token
	 * @return
	 */
	public abstract double distance(Token other_token);
}
