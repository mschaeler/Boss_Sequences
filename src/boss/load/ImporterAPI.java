package boss.load;

import java.util.ArrayList;

import plus.data.Book;

public class ImporterAPI {
	public static Book get_book(int book_id) {
		return Importer.get_book(book_id);
	}
	public static ArrayList<Book> get_books(int[] book_id) {
		ArrayList<Book> ret = new ArrayList<Book>(book_id.length);
		for(int id : book_id) {
			ret.add(get_book(id));
		}
		return ret;
	}
	
	public static ArrayList<Book> get_all_german_books() {
		return get_books(Book.GERMAN_BOOKS);
	}
	
	public static ArrayList<Book> get_all_english_books() {
		return get_books(Book.ENGLISH_BOOKS);
	}
	
	public static ArrayList<Book> get_all_latin_books() {
		return get_books(Book.LATIN_BOOKS);
	}
	
	public static ArrayList<Book> get_all_greek_books() {
		return get_books(Book.GREEK_BOOKS);
	}

	public static ArrayList<Book> get_pan_11_books(int src_id, int susp_id) {
		ArrayList<Book> ret = new ArrayList<Book>(2);
		ret.add(Importer.get_book_pan11(susp_id, false));
		ret.add(Importer.get_book_pan11(src_id, true));
		return ret;
	}
}
