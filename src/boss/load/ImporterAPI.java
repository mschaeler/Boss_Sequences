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
	
	public static ArrayList<String> get_raw_books_pan(boolean src) {
		ArrayList<String> result = new ArrayList<String>();
		if(src) {
			for(String file_path : Importer.PAN11_SRC) {
				ArrayList<String[]> raw_book = Importer.get_data_from_file_pan11(file_path);
				Book b = Importer.to_book_pan11(raw_book, Book.LANGUAGE_ENGLISH, file_path);
				String s = b.to_single_line_string();
				result.add(s);
			}
		}else{//query
			for(String file_path : Importer.PAN11_SUSP) {
				ArrayList<String[]> raw_book = Importer.get_data_from_file_pan11(file_path);
				Book b = Importer.to_book_pan11(raw_book, Book.LANGUAGE_ENGLISH, file_path);
				String s = b.to_single_line_string();
				result.add(s);
			}
		}
		return result;
	}
	
	public static ArrayList<Book> get_all_pan_books() {
		ArrayList<Book> ret = new ArrayList<Book>();
		for(int id=0;id<Importer.PAN11_SRC.length; id++) {
			Book b = Importer.get_book_pan11(id, true);
			ret.add(b);
		}
		for(int id=0;id<Importer.PAN11_SUSP.length; id++) {
			Book b = Importer.get_book_pan11(id, false);
			ret.add(b);
		}
		
		return ret;
	}
}
