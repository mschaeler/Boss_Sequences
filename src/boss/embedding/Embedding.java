package boss.embedding;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;

import plus.data.Book;

public class Embedding {
	String[] words;
	double[][] vectors = null;//loaded on demand
	HashSet<String> hash_index;
	String file;
	
	public Embedding(String file) {
		ArrayList<String> raw_entries = Loader.clean(new File(file));
		this.words = new String[raw_entries.size()];
		this.hash_index = new HashSet<String>(raw_entries.size());
		this.file = file;
		
		for(int i=0;i<raw_entries.size();i++) {
			String word = raw_entries.get(i);
			this.words[i] = word;
			hash_index.add(word);
			//TODO get the vectors, but only for the words that we need
		}
	}
	
	public Embedding(int language) {
		this(Loader.get_file(language));
	}
	public ArrayList<MatchesWithEmbeddings> get_minimal_embedding(Match[] matches){
		BufferedReader br;
		HashMap<String, ArrayList<Integer>> words = new HashMap<String, ArrayList<Integer>>(matches.length);//(String m.s1, position(s) in matches)
		ArrayList<Integer> positions;
		for(int i=0;i<matches.length;i++) {
			String key = matches[i].string_in_embedding; 
			if((positions=words.get(key))==null) {
				//entry does not exist yet
				ArrayList<Integer> temp = new ArrayList<Integer>();
				temp.add(i);
				words.put(key,temp);//these are all the words we need:	
			}else{
				positions.add(i);
			}
			 
		}
		
		try {
			br = new BufferedReader(new FileReader(this.file));
			String line;
			int i = 0;
			String meta_data = br.readLine();
			String[] meta_data_tokens = meta_data.split(" ");

			ArrayList<MatchesWithEmbeddings> result = new ArrayList<MatchesWithEmbeddings>(matches.length);
	        while ((line = br.readLine()) != null) {
	        	i++;
	        	String[] tokens = line.split(" ");
	        	String word = tokens[0];

	        	positions = words.get(word);
	        	if(positions!=null) {
	        		for(int pos : positions) {
	        			double[] vector = new double[tokens.length-1];//first token is the word, remainder all belongs to the vector
		        		for(int dim=1;dim<tokens.length;dim++) {
		        			vector[dim-1] = Double.parseDouble(tokens[dim]);
		        		}
	        			MatchesWithEmbeddings mew = new MatchesWithEmbeddings(matches[pos], vector);
	        			result.add(mew);
	        		}
	        	}
	        	if(i%100000==0) {
	        		System.out.println(i+" of "+meta_data_tokens[0] +" "+ tokens[0]);	
	        	}
	        }
	        br.close();
	        Collections.sort(result);
	        return result;
		} catch (FileNotFoundException ex) {
			ex.printStackTrace();
		} catch (IOException ex) {
			ex.printStackTrace();
		}
		return null;
	}
	
	static boolean materialize = true;
	public static final String ENGLISH_MINIMAL_EMBEDDINGS = ".\\data\\en\\matches.en.min.tsv";
	public static final String ENGLISH_EMBEDDINGS = ".\\data\\en\\matches_stopwords.en.min.tsv";
	public static final String PAN11_ENGLISH_EMBEDDINGS = "./data/pan11/all_matches_pan.tsv";
	public static ArrayList<MatchesWithEmbeddings> get_minimal_embedding_en(){
		String file 	= ".\\data\\en\\cc.en.300.vec";
		Embedding e 	= new Embedding(file);
		Match[] matches = e.match_all_words(Words.words_en);
		ArrayList<MatchesWithEmbeddings> mew = e.get_minimal_embedding(matches);
		
		//Check whether we have everything mapped
		String[] all_words = Words.words_en;
		HashSet<String> words_mapped = new HashSet<String>();
		for(Match m : matches) {
			words_mapped.add(m.string_in_text);
		}
		HashSet<String> words_mapped_to_embeddeding = new HashSet<String>();
		for(MatchesWithEmbeddings m : mew) {
			if(m!=null)
				words_mapped_to_embeddeding.add(m.string_in_text);
		}
		for(String s : all_words){
			if(!words_mapped.contains(s)) {
				System.err.println("Error: matches does not contain "+s);
			}else{ // word is contained in matches
				if(!words_mapped_to_embeddeding.contains(s)) {
					System.err.println("Error: matches contains, but embedding does not "+s);	
				}
			}
		}
		if(materialize) {
			MatchesWithEmbeddings.materialize(ENGLISH_MINIMAL_EMBEDDINGS, mew);
		}
		return mew;
	}
	static final String GERMAN_MINIMAL_EMBEDDINGS = ".\\data\\de\\matches.de.min.tsv";
	public static ArrayList<MatchesWithEmbeddings> get_minimal_embedding_de(){
		String file 	= ".\\data\\de\\cc.de.300.vec";
		Embedding e 	= new Embedding(file);
		Match[] matches = e.match_all_words(Words.words_de);
		ArrayList<MatchesWithEmbeddings> mew = e.get_minimal_embedding(matches);
		
		//Check whether we have everything mapped
		String[] all_words = Words.words_de;
		HashSet<String> words_mapped = new HashSet<String>();
		for(Match m : matches) {
			words_mapped.add(m.string_in_text);
		}
		HashSet<String> words_mapped_to_embeddeding = new HashSet<String>();
		for(MatchesWithEmbeddings m : mew) {
			if(m!=null)
				words_mapped_to_embeddeding.add(m.string_in_text);
		}
		for(String s : all_words){
			if(!words_mapped.contains(s)) {
				System.err.println("Error: matches does not contain "+s);
			}else{ // word is contained in matches
				if(!words_mapped_to_embeddeding.contains(s)) {
					System.err.println("Error: matches contains, but embedding does not "+s);	
				}
			}
		}
		if(materialize) {
			MatchesWithEmbeddings.materialize(".\\data\\de\\matches.de.min.tsv", mew);
		}
		return mew;
	}
	
	Match[] match_all_words(String[] word_list){
		Match[] ret = new Match[word_list.length];
		for(int i=0;i<word_list.length;i++) {
			ret[i] = match(word_list[i]);
		}
		Arrays.sort(ret);
		return ret;
	}
	
	Match match(String to_match) {
		if(this.hash_index.contains(to_match)) {
			return new Match(to_match, to_match, 1.0);//score = 1.0 means it is the same
		}else{
			return best_match_entry(to_match);
		}
	}
	
	Match best_match_entry(String word) {
		int max_similarity = Integer.MIN_VALUE;
		String best_match = null;
		
		for(String s : this.words) {
			int common_prefix = common_prefix(word, s);
			if(common_prefix > max_similarity) {
				max_similarity = common_prefix;
				best_match = s;
			}
		}
		double score = (double) common_prefix(word, best_match) / (double) (Math.max(word.length(), best_match.length()));
		return new Match(best_match, word, score);
	}
	/**
	 * Finds best match of the provided key in the dictionary or word embedding having 
	 * Longest common prefix. We use this method to match lexicographic tokens to their semantic
	 * 
	 * @param key
	 * @return
	 */
	final int common_prefix(final String s1, final String s2) {
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
	
	public static String get_embedding_path(final int language) {
		return get_embedding_path(language, false);
	}
	/**
	 * 
	 * @param language
	 * @param ignore_stopwords
	 * @return
	 */
	public static String get_embedding_path(final int language, boolean ignore_stopwords) {
		if(language == Book.LANGUAGE_ENGLISH) {
			if(ignore_stopwords) {
				return ENGLISH_MINIMAL_EMBEDDINGS;
			}else{
				// return ENGLISH_EMBEDDINGS;
				return PAN11_ENGLISH_EMBEDDINGS;
			}
		}else if(language == Book.LANGUAGE_GERMAN){
			return GERMAN_MINIMAL_EMBEDDINGS;
		}else{
			System.err.println("get_embedding_path() Unknown embedding for language="+language);
			return null;
		}
	} 
}
