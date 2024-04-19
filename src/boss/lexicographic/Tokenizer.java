package boss.lexicographic;

import java.util.ArrayList;
import java.util.HashSet;
import plus.data.Book;
import org.apache.lucene.analysis.PorterStemmer;

import boss.util.Config;


public abstract class Tokenizer {
	public Tokenizer() {
		
	}
	/**
	 * Prepares tokenization. It is applied to every String based Paragraph.
	 * @param s
	 * @return
	 */
	abstract String[] basic_tokenization(String s);
	
	/**
	 * We apply this method to every TokenizedParagraph
	 */
	abstract void tokenize(TokenizedParagraph p);
	/**
	 * Does what it says at Paragraph level
	 * @param books
	 * @return
	 */
	protected abstract ArrayList<Book> run(ArrayList<Book> books);
	
	static String[] replace_non_alphabetic_characters(String[] array, int language){
		String[] ret = new String[array.length];
		for(int i=0;i<ret.length;i++) {
			ret[i]=replace_non_alphabetic_characters(array[i], language);
		}
		return ret;
	}
	
	static String replace_non_alphabetic_characters(String s, int language){
		String regex_characters_to_keep = null;
		if(language == Book.LANGUAGE_LATIN) {
			regex_characters_to_keep = "[^a-zA-Z ]";
		}else if(language == Book.LANGUAGE_OLD_GREEK){
			regex_characters_to_keep = "[^Α-Ωα-ωίϊ�?όάέ�?ϋΰήώ ]";
		}else if(language == Book.LANGUAGE_ARAMAIC){
			regex_characters_to_keep = "[^a-zA-ZΑ-Ωα-ωäöüÄÖÜßίϊ�?όάέ�?ϋΰήώ ]";
		}else if(language == Book.LANGUAGE_OLD_GEORGIAN){
			regex_characters_to_keep = "[^a-zA-ZΑ-Ωα-ωäöüÄÖÜßίϊ�?όάέ�?ϋΰήώ ]";
		}else if(language == Book.LANGUAGE_GERMAN){
			regex_characters_to_keep = "[^a-zA-Z������� ]";
		}else if(language == Book.LANGUAGE_ENGLISH){
			if(Config.REMOVE_NUMBERS) {
				regex_characters_to_keep = "[^a-zA-Z ]";//default
			}else {
				regex_characters_to_keep = "[^a-zA-Z0-9 ]";//for Jaccard
			}
		}else{
			System.err.println("Tokenizer.replace_non_alphabetic_characters(String,int): unknown language using [^a-zA-ZΑ-Ωα-ωäöüÄÖÜßίϊ�?όάέ�?ϋΰήώ ]");
			regex_characters_to_keep = "[^a-zA-ZΑ-Ωα-ωäöüÄÖÜßίϊ�?όάέ�?ϋΰήώ ]";
		}
		String ret = s.replaceAll(regex_characters_to_keep, "");
		return ret;
	}
	
	static String[] split_at_lower_case_whitespace(String s){
		return s.toLowerCase().split("\\s+");
	}
	static String[] to_lower_case(String[] array){
		String[] ret = new String[array.length];
		for(int i=0;i<ret.length;i++) {
			ret[i] = array[i].toLowerCase();
		}
		return ret;
	}
	
	static String[] stop_word_removal(String[] words, int language) {
		HashSet<String> stop_words = StopWords.get_stop_words(language);
		String[] ret = new String[words.length];
		for(int i=0;i<words.length;i++) {
			String t = words[i];
			if(!stop_words.contains(t)) {
				ret[i] = t;//Simply keep it
			}else{
				ret[i] = null;//This is the way how we delete a word. Note, the length of the array must remain the same.
			}
		}
		return ret;
	}
	public static ArrayList<Book> run(ArrayList<Book> books, Tokenizer my_tokenizer) {
		return my_tokenizer.run(books);
	}

	/** API to use **/
	static void remove_non_alphabetic_characters(TokenizedParagraph p) {
		String[] s = replace_non_alphabetic_characters(p.last_intermediate_result(), p.my_chapter.book.language);
		p.inter_mediate_results.add(s);
		p.step_name.add("Non character removal");
	}
	static void to_lower_case(TokenizedParagraph p) {
		String[] s = to_lower_case(p.last_intermediate_result()); 
		p.inter_mediate_results.add(s);
		p.step_name.add("To lower case");
	}
	static void remover_stop_words(TokenizedParagraph p) {
		String[] s = stop_word_removal(p.last_intermediate_result(), p.my_chapter.book.language); 
		p.inter_mediate_results.add(s);
		p.step_name.add("Stopword removal");
	}
	static void stem(TokenizedParagraph p) {
		String[] s = stemming(p.last_intermediate_result(), p.my_chapter.book.language); 
		p.inter_mediate_results.add(s);
		p.step_name.add("Stemming");
	}
	private static String[] stemming(String[] words, int language) {
		if(language!=Book.LANGUAGE_ENGLISH) {
			System.err.println("Stemmer supports only english");
			return words;
		}
		PorterStemmer stemmer = new PorterStemmer();
		String[] ret = new String[words.length];
		for(int i=0;i<words.length;i++) {
			String t = words[i];
			String stemmed_t = stemmer.stem(t);
			ret[i] = stemmed_t;
		}
		return ret;
	}
}
