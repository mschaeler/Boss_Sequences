package boss.token_mapping;

import java.util.HashMap;

import boss.lexicographic.StringToken;
import boss.semantic.SemanticToken;
import boss.semantic.WordEmbeddingToken;

public class WordEmbedding extends MyDictionary{
	private HashMap<String,double[]> embedding = null;
	
	public void set_emebdding(HashMap<String,double[]> embedding) {
		this.embedding = embedding;
	}
	
	@Override
	public SemanticToken map_to_semantic(StringToken t) {
		WordEmbeddingToken wet = new WordEmbeddingToken(t);//only fullform
		//determine baseform
		String baseform = map_to_word(t);
		wet.baseform = baseform;
		//determine semantic
		if(embedding != null) {
			double[] word_vector = embedding.get(baseform);
			if(word_vector==null) {
				//should never happen
				System.err.println("WordEmbedding.map_to_semantic() word_vector==null");
			}
			wet.word_vector = word_vector;
		}else {
			System.out.println("Warning: WordEmbedding.map_to_semantic() embedding = null");
		}
		return wet;
	}

}
