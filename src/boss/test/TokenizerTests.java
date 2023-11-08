package boss.test;

import java.util.ArrayList;

import boss.lexicographic.*;
import boss.load.*;
import boss.semantic.SemanticToken;
import boss.semantic.Sequence;
import boss.token_mapping.MyDictionary;
import boss.token_mapping.WordEmbedding;
import plus.data.*;

public class TokenizerTests {
	public static void main(String[] args) {
		ArrayList<Book> books; 
		ArrayList<Book> tokenized_books;
		
		//books = ImporterAPI.get_all_german_books();
		books = ImporterAPI.get_all_pan_books();
		
		tokenized_books = Tokenizer.run(books, new BasicTokenizer()); 
		/*for(Book b : tokenized_books) {
			System.out.println(b);
		} */
		
		//MyDictionary dict = new WordEmbedding();
		//ArrayList<ArrayList<Sequence>> sequences = Sequence.to_sequence(tokenized_books, dict);
		
		ArrayList<String> all_tokens_ordered = Sequence.get_ordered_token_list(Sequence.get_unique_tokens(tokenized_books));
		System.out.println(all_tokens_ordered.toString());
		for(String s : all_tokens_ordered) {
			System.out.println(s);
		}
		
		/*books = ImporterAPI.get_all_english_books();
		tokenized_books = Tokenizer.run(books, new BasicTokenizer());
		
		all_tokens_ordered = Sequence.get_ordered_token_list(Sequence.get_unique_tokens(tokenized_books));
		System.out.println(all_tokens_ordered.toString());*/
	}
}
