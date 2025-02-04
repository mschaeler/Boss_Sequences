package boss.lexicographic;

import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import plus.data.Book;

import org.apache.lucene.analysis.Analyzer;
import org.apache.lucene.analysis.PorterStemmer;
import org.apache.lucene.analysis.TokenStream;
import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.analysis.tokenattributes.CharTermAttribute;
import org.apache.lucene.util.Version;

import boss.util.Config;
import boss.util.Util;
import pan.Jaccard;


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
	
	public static String replace_non_alphabetic_characters(String s, int language){
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
	public static String[] to_lower_case(String[] array){
		String[] ret = new String[array.length];
		for(int i=0;i<ret.length;i++) {
			ret[i] = array[i].toLowerCase();
		}
		return ret;
	}
	
	public static String[] stop_word_removal(String[] words, int language) {
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
	public static String[] stemming(String[] words, int language) {
		if(language!=Book.LANGUAGE_ENGLISH) {
			System.err.println("Stemmer supports only english");
			return words;
		}
		PorterStemmer stemmer = new PorterStemmer();
		String[] ret = new String[words.length];
		for(int i=0;i<words.length;i++) {
			String t = words[i];
			String stemmed_t = null;
			if(t!=null) {
				stemmed_t = stemmer.stem(t);
			}
			ret[i] = stemmed_t;
		}
		return ret;
	}
	public static String remove_duplicate_whitespaces(String s) {
		return s.trim().replaceAll(" +", " ");
	}
	
	public static ArrayList<String> tokenize(Book b, boolean use_stemming){
		String org = b.to_single_line_string();
		if(Jaccard.verbose_level>=Jaccard.print_everything) {System.out.println("org=\t\t\t"+org);}
		String temp = Tokenizer.replace_non_alphabetic_characters(org, b.language);
		if(Jaccard.verbose_level>=Jaccard.print_everything) {System.out.println("alpha numeric=\t\t"+temp);}
		temp = Tokenizer.remove_duplicate_whitespaces(temp);
		if(Jaccard.verbose_level>=Jaccard.print_everything) {System.out.println("whitespaces=\t\t"+temp);}
		String[] tokens = temp.split(" ");
		
		tokens = Tokenizer.to_lower_case(tokens);
		//if(verbose_level>=print_everything) System.out.println("to_lower_case\t\t"+Util.outTSV(tokens)); 
		//tokens = Tokenizer.stop_word_removal(tokens, language);//sometimes stop words are modified upon stemming. Thus, check it twice.
		//if(Jaccard.verbose_level>=Jaccard.print_everything) System.out.println("stop_word_removal\t"+Util.outTSV(tokens));
		if(use_stemming) {
			tokens = Tokenizer.stemming(tokens, b.language);
			if(Jaccard.verbose_level>=Jaccard.print_everything) System.out.println("stemming\t\t"+Util.outTSV(tokens));
		}
		tokens = Tokenizer.stop_word_removal(tokens, b.language);
		if(Jaccard.verbose_level>=Jaccard.print_everything) System.out.println("stop_word_removal\t"+Util.outTSV(tokens));
		//TODO check for empty Strings, remove nulls
		ArrayList<String> ret = new ArrayList<String>();
		for(String s : tokens) {
			if(s != null) {
				if(s.length()!=0) {
					ret.add(s);	
				}
			}
		}
		
		return ret;
	}
	
	public static ArrayList<String> tokenize_bible_de(String org){
		if(Jaccard.verbose_level>=Jaccard.print_everything) {System.out.println("org=\t\t\t"+org);}
		String temp = Tokenizer.replace_non_alphabetic_characters(org, Book.LANGUAGE_GERMAN);
		if(Jaccard.verbose_level>=Jaccard.print_everything) {System.out.println("alpha numeric=\t\t"+temp);}
		temp = Tokenizer.remove_duplicate_whitespaces(temp);
		if(Jaccard.verbose_level>=Jaccard.print_everything) {System.out.println("whitespaces=\t\t"+temp);}
		String[] tokens = temp.split(" ");
		
		tokens = Tokenizer.to_lower_case(tokens);
		//if(verbose_level>=print_everything) System.out.println("to_lower_case\t\t"+Util.outTSV(tokens)); 
		//tokens = Tokenizer.stop_word_removal(tokens, language);//sometimes stop words are modified upon stemming. Thus, check it twice.
		//if(Jaccard.verbose_level>=Jaccard.print_everything) System.out.println("stop_word_removal\t"+Util.outTSV(tokens));

		tokens = Tokenizer.stop_word_removal(tokens, Book.LANGUAGE_GERMAN);
		if(Jaccard.verbose_level>=Jaccard.print_everything) System.out.println("stop_word_removal\t"+Util.outTSV(tokens));
		//TODO check for empty Strings, remove nulls
		ArrayList<String> ret = new ArrayList<String>();
		for(String s : tokens) {
			if(s != null) {
				if(s.length()!=0) {
					ret.add(s);	
				}
			}
		}
		
		return ret;
	}
	
	public static ArrayList<String> tokenize(String org, boolean use_stemming){
		if(Jaccard.verbose_level>=Jaccard.print_everything) {System.out.println("org=\t\t\t"+org);}
		String temp = Tokenizer.replace_non_alphabetic_characters(org, Book.LANGUAGE_ENGLISH);
		if(Jaccard.verbose_level>=Jaccard.print_everything) {System.out.println("alpha numeric=\t\t"+temp);}
		temp = Tokenizer.remove_duplicate_whitespaces(temp);
		if(Jaccard.verbose_level>=Jaccard.print_everything) {System.out.println("whitespaces=\t\t"+temp);}
		String[] tokens = temp.split(" ");
		
		tokens = Tokenizer.to_lower_case(tokens);
		//if(verbose_level>=print_everything) System.out.println("to_lower_case\t\t"+Util.outTSV(tokens)); 
		//tokens = Tokenizer.stop_word_removal(tokens, language);//sometimes stop words are modified upon stemming. Thus, check it twice.
		//if(Jaccard.verbose_level>=Jaccard.print_everything) System.out.println("stop_word_removal\t"+Util.outTSV(tokens));
		if(use_stemming) {
			tokens = Tokenizer.stemming(tokens, Book.LANGUAGE_ENGLISH);
			if(Jaccard.verbose_level>=Jaccard.print_everything) System.out.println("stemming\t\t"+Util.outTSV(tokens));
		}
		tokens = Tokenizer.stop_word_removal(tokens, Book.LANGUAGE_ENGLISH);
		if(Jaccard.verbose_level>=Jaccard.print_everything) System.out.println("stop_word_removal\t"+Util.outTSV(tokens));
		//TODO check for empty Strings, remove nulls
		ArrayList<String> ret = new ArrayList<String>();
		for(String s : tokens) {
			if(s != null) {
				if(s.length()!=0) {
					ret.add(s);	
				}
			}
		}
		
		return ret;
	}
	
	public static ArrayList<String> tokenize_txt_align(Book b){
		HashSet<String> stopwords = StopWords.get_DONG_DENG_STOPWORDS();
		String org = b.to_single_line_string();
		
		
		Analyzer analyzer = new StandardAnalyzer(Version.LUCENE_35, stopwords); //TODO
		/*Reader reader = new StringReader(org);
		TokenStream stream = analyzer.tokenStream("field", reader);
		try {
			stream.reset();
			while (stream.incrementToken()) {
			    String stem = stream.getAttribute(CharTermAttribute.class).toString();
			    // doing something with the stem
			    System.out.print(stem+ " ");
			}
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}*/
		
		if(Jaccard.verbose_level>=Jaccard.print_everything) {System.out.println("org=\t\t\t"+org);}
		final String delim = "[\"#$%&\'()*+,-./:;<=>?@\\[\\]^_`{|}~ ]";
		// final String period_str = "\n\t\r\x0b\x0c,.!?;!";
		final String period_str = "[\n\t\r,.!?;!]";// no line tabs in Java
		
		String[] sentences = org.split(period_str);
		if(Jaccard.verbose_level>=Jaccard.print_everything) {System.out.println("sentences=\t\t"+Arrays.toString(sentences));}
		
		ArrayList<String> ret = new ArrayList<String>();
		PorterStemmer stemmer = new PorterStemmer();
		
		for(String sentence : sentences) {
			String[] tokens = sentence.split(delim);
			for(String t : tokens) {
				if(Config.USE_TXT_ALIGN_CORRECT_STEMMING) {
					if(Config.USE_TXT_ALIGN_LEMMATIZING) {//Lemmatizing. This way we prevent empty words, find all the stop words and make better use of the word list.
						Reader r = new StringReader(t);
						TokenStream ts = analyzer.tokenStream("field", r);
						try {
							ts.reset();
							while (ts.incrementToken()) {
							    t = ts.getAttribute(CharTermAttribute.class).toString();
							    if(!stopwords.contains(t)) {//@FIX in original code
									String stem_word = stemmer.stem(t.toLowerCase());
									if(!stopwords.contains(stem_word)) {
										ret.add(stem_word);	
									}
							    }
							}
						} catch (IOException e) {
							// TODO Auto-generated catch block
							e.printStackTrace();
						}
					}else if(!stopwords.contains(t)) {//@FIX in original code
						String stem_word = stemmer.stem(t.toLowerCase());
						if(!stopwords.contains(stem_word)) {
							ret.add(stem_word);	
						}
					}
				}else{//Stemming only. As in original code.
					String stem_word = stemmer.stem(t.toLowerCase());
					if(!stopwords.contains(stem_word)) {
						ret.add(stem_word);	
					}
				}
			}
		}
		analyzer.close();
		return ret;
	}
	
	/*private static ArrayList<String> tokenize_me(Book b) {
		return (Config.USE_TXT_ALIGN_PREPROCESSING) ? Tokenizer.tokenize_txt_align(b) : Tokenizer.tokenize(b);
	}*/
	public static ArrayList<ArrayList<String>>[] tokenize(ArrayList<Book>[] all_src_plagiats_pairs) {
		ArrayList<ArrayList<String>>[] ret = new ArrayList[all_src_plagiats_pairs.length];
		for(int i=0;i<all_src_plagiats_pairs.length;i++) {
			System.out.println("Tokenize pair "+i);
			ArrayList<Book> al = all_src_plagiats_pairs[i];
			ret[i] = tokenize(al);
		}
		return ret;
	}
	public static ArrayList<ArrayList<String>> tokenize(ArrayList<Book> list) {
		ArrayList<ArrayList<String>> ret = new ArrayList<ArrayList<String>>(list.size());
		for(Book b :list) {
			ret.add(tokenize(b, false));
		}
		return ret;
	}
	public static ArrayList<ArrayList<String>> tokenize_(ArrayList<String> documents) {
		ArrayList<ArrayList<String>> ret = new ArrayList<ArrayList<String>>();
		for(String doc : documents) {
			ArrayList<String> temp = tokenize(doc, false);
			ret.add(temp);
		}
		return ret;
	}
}
