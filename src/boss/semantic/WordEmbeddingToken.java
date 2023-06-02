package boss.semantic;

import boss.lexicographic.StringToken;
import plus.lexicographic.Token;

public class WordEmbeddingToken extends SemanticToken{
	/**
	 * That's the semantic. We compare two similarities by computing the cosine of the enclosed angle. Larger value (esp. towards 1) mean similar semantic. 
	 * The distance is 1-similarity.
	 */
	public double[] word_vector;
	
	public WordEmbeddingToken(StringToken token) {
		this(token.raw_token, null);
	}
	public WordEmbeddingToken(StringToken token, double[] word_vector) {
		this(token.raw_token, word_vector);
	}
	public WordEmbeddingToken(String raw_token, double[] word_vector) {
		super.fullform = raw_token;
		super.baseform = null;
		this.word_vector = word_vector;
	}
	
	@Override
	public double distance(Token other_token) {
		if(other_token instanceof WordEmbeddingToken) {
			if(this.matches(other_token)) {
				return 0;
			}
			double distance = cosine_distance(this.word_vector, ((WordEmbeddingToken) other_token).word_vector);
			//TODO if one of the word vectors = null, i.e., we did not find a good enough match, Use String edit distance
			/*if(similarity==0) {
				super.distance(other_token);//Than use edit distance
			}*/
			return distance;
		}
		System.err.println("WordEmbeddingToken.distance() needs WordEmbeddingToken");
		return Double.MAX_VALUE;
	}
	public static double cosine_distance(double[] vectorA, double[] vectorB) {//Optional TODO - normalize all vectors to length = 1. Then, computation is much simpler.
	    double dotProduct = 0.0;
	    double normA = 0.0;
	    double normB = 0.0;
	    for (int i = 0; i < vectorA.length; i++) {
	        dotProduct += vectorA[i] * vectorB[i];
	        normA += Math.pow(vectorA[i], 2);
	        normB += Math.pow(vectorB[i], 2);
	    }   
	    return 1.0d - (dotProduct / (Math.sqrt(normA) * Math.sqrt(normB)));
	}

	public String toString(){
		return super.fullform+" ["+this.word_vector[0]+",..]";
	}
}
