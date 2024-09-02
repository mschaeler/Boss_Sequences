package boss.semantic;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;

import boss.lexicographic.*;
import boss.token_mapping.MyDictionary;
import plus.data.*;

public class Sequence {
	ArrayList<SemanticToken> my_tokens;
	
	public static ArrayList<Sequence> to_sequence(Book b) {
		ArrayList<Sequence> result = new ArrayList<Sequence>();
		to_sequence(b, null, new ArrayList<Sequence>());
		return result;
	}
	static void to_sequence(Book b, ArrayList<Sequence> result) {
		to_sequence(b, null, result);
	}
	
	public static void to_sequence(Book b, MyDictionary dict, ArrayList<Sequence> result) {
		for(Chapter c : b.my_chapters) {
			to_sequence(c, dict, result);
		}
		//TODO zwischenschritt alle Unique Tokens?
	}
	
	
	static void to_sequence(Chapter c, MyDictionary dict, ArrayList<Sequence> result) {
		for(Paragraph p : c.my_paragraphs) {
			Sequence seq = to_sequence(p, dict);
			result.add(seq);
		}
	}
	
	static Sequence to_sequence(Paragraph p, MyDictionary dict) {
		if(p instanceof TokenizedParagraph) {
			TokenizedParagraph tp = (TokenizedParagraph) p;
			ArrayList<StringToken> string_tokens = tp.get_tokens();
			Sequence seq = new Sequence();
			for(StringToken st : string_tokens) {
				SemanticToken add_me = dict.map_to_semantic(st);
				seq.my_tokens.add(add_me);
			}
			return seq;
		}else{
			//TODO Syserror
			return null;
		}
	}
	/**
	 * 
	 * @param tokenized_books
	 * @param dict
	 * @return
	 */
	public static ArrayList<ArrayList<Sequence>> to_sequence(ArrayList<Book> tokenized_books, MyDictionary dict) {
		ArrayList<ArrayList<Sequence>> result= new ArrayList<ArrayList<Sequence>>(tokenized_books.size());
		
		for(Book b : tokenized_books) {
			ArrayList<Sequence> result_book = new ArrayList<Sequence>();
			to_sequence(b, dict, result_book);
			result.add(result_book);
		}
		return result;
	}
	
	/**
	 * 
	 * @param tokenized_books
	 * @return all unique counts and their occurrence frequency
	 */
	public static HashMap<String, Integer> get_unique_tokens(ArrayList<Book> tokenized_books){
		HashMap<String, Integer> unique_tokens = new HashMap<String, Integer>(1000);
		for(Book b : tokenized_books) {
			for(Chapter c : b.my_chapters) {
				for(Paragraph p : c.my_paragraphs) {
					if(p instanceof TokenizedParagraph) {
						TokenizedParagraph tp = (TokenizedParagraph) p;
						ArrayList<StringToken> string_tokens = tp.get_tokens();
						for(StringToken st : string_tokens) {
							/*if(st.raw_token==null || st.raw_token.equals("null")) {
								continue;//may have been removed, e.g., for stop words
							}*/
							//do we know this token already?
							Integer count = unique_tokens.get(st.raw_token);
							if(count == null) {
								unique_tokens.put(st.raw_token, 1);//create a new entry for this token and state we have seen it once
							}else{
								unique_tokens.put(st.raw_token, count.intValue()+1);//increment the count value
							}
						}
					}
				}
			}
		}
		return unique_tokens;
	}
	
	//TODO should these methods remain here?
	public static ArrayList<String> get_ordered_token_list(HashMap<String, Integer> unique_tokens){
		ArrayList<String> ordered_tokens = new ArrayList<String>(unique_tokens.entrySet().size());
		for(Entry<String, Integer> entry : unique_tokens.entrySet()) {
			String token = entry.getKey();
			ordered_tokens.add(token);
		}
		Collections.sort(ordered_tokens);
		return ordered_tokens;
	}
	public static ArrayList<String> get_ordered_token_list(HashSet<String> unique_tokens){
		ArrayList<String> ordered_tokens = new ArrayList<String>(unique_tokens.size());
		for(String token : unique_tokens) {
			ordered_tokens.add(token);
		}
		Collections.sort(ordered_tokens);
		return ordered_tokens;
	}
	public static HashSet<String> get_unique_tokens_(ArrayList<String>[] src_plagiats_pair) {
		HashSet<String> unique_tokens = new HashSet<String>();
		
		for(ArrayList<String> document : src_plagiats_pair) {
			for(String token : document) {
				unique_tokens.add(token);
			}
		}
		
		return unique_tokens;
	}
	public static void out_unique_tokens_ordered(ArrayList<ArrayList<String>>[] all_src_plagiats_pairs_tokinized) {
		System.out.println("Unique tokens *********************************");
		HashSet<String> hs = new HashSet<String>();
		for(ArrayList<ArrayList<String>> pair : all_src_plagiats_pairs_tokinized) {
			for(ArrayList<String> document : pair) {
				for(String token : document) {
					hs.add(token);
				}
			}
		}
		ArrayList<String> al = get_ordered_token_list(hs);
		Collections.sort(al);
		for(String s : al) {
			System.out.println(s);
		}
		System.out.println("Unique tokens *********************************");
	}
}
