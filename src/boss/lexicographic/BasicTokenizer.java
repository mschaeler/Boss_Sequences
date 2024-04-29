package boss.lexicographic;

import java.util.ArrayList;

import boss.util.Config;
import plus.data.Book;
import plus.data.Chapter;
import plus.data.Paragraph;

public class BasicTokenizer extends Tokenizer {
	//Book[] my_books  = null;

	@Override
	String[] basic_tokenization(String s) {
		String[] tokenized_paragraph = s.split(" ");
		return tokenized_paragraph;
	}

	@Override
	void tokenize(TokenizedParagraph p) {
		Tokenizer.remove_non_alphabetic_characters(p);
		Tokenizer.to_lower_case(p);
		if(Config.REMOVE_STOP_WORDS){
			Tokenizer.remover_stop_words(p);
		}
	}

	@Override
	protected ArrayList<Book> run(ArrayList<Book> books) {
		ArrayList<Book> ret = new ArrayList<Book>(books.size());//We do not modify the original data, but entirely create a new collection of books
		for(Book b : books) {
			Book b_prime = new Book(b); //Copies container meta data only
			ret.add(b_prime);
			for(Chapter c : b.my_chapters) {
				Chapter c_prime = new Chapter(b_prime, c.chapter_name);//Copies container meta data only
				b_prime.my_chapters.add(c_prime);
				for(Paragraph p : c.my_paragraphs) {
					TokenizedParagraph p_prime = new TokenizedParagraph(c_prime, p.paragraph_name, p.my_paragraph_as_text, this);
					tokenize(p_prime);
					c_prime.my_paragraphs.add(p_prime);
				}
			}
		}
		return ret;
	}

}
