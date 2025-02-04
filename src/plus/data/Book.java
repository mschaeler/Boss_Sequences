package plus.data;
/**
 * E.g., Esther
 * @author b1074672
 *
 */

import java.util.ArrayList;


public class Book {
	public static final int AT = 1;
	public static final int JA = AT+1;
	public static final int MT = JA+1;
	public static final int OG = MT+1;
	public static final int Vg = OG+1;
	public static final int VL = Vg+1;
	//German versions below
	public static final int LUTHER 		= VL+1;
	public static final int ELBERFELDER = LUTHER+1;
	public static final int NE 			= ELBERFELDER+1;
	public static final int SCHLACHTER 	= NE+1; // Neue Evangelistische
	public static final int VOLXBIBEL  	= SCHLACHTER+1;
	//English versions below
	public static final int ESV  		= VOLXBIBEL+1; // English Standard Version
	public static final int KING_JAMES  = ESV+1;
	
	//Language ids
	public static final int LANGUAGE_LATIN 			= 0;
	public static final int LANGUAGE_OLD_GREEK 		= 1;
	public static final int LANGUAGE_ARAMAIC 		= 2;
	public static final int LANGUAGE_OLD_GEORGIAN 	= 3;
	public static final int LANGUAGE_GERMAN		 	= 4;
	public static final int LANGUAGE_ENGLISH	 	= 5;
	
	static String get_language_string(Book b) {
		return get_language_string(b.language);
	}
	
	public static String get_language_string(int language) {
		String ret;
		if(language==LANGUAGE_LATIN){
			ret = "Latin";
		}else if(language==LANGUAGE_OLD_GREEK) {
			ret = "Old Greek";
		}else if(language==LANGUAGE_ARAMAIC) {
			ret = "Aramaic";
		}else if(language==LANGUAGE_OLD_GEORGIAN) {
			ret = "Old Georgian";
		}else if(language==LANGUAGE_GERMAN) {
			ret = "German";
		}else if(language==LANGUAGE_ENGLISH) {
			ret = "English";
		}else{
			ret = "Unknown language code: "+language;
		}
		return ret;
	}
	
	//Collections of text corpora in the same language
	public static final int[] GERMAN_BOOKS  = {LUTHER,ELBERFELDER,NE,SCHLACHTER,VOLXBIBEL};
	public static final int[] ENGLISH_BOOKS = {ESV,KING_JAMES};
	public static final int[] LATIN_BOOKS   = {Vg,VL};
	public static final int[] GREEK_BOOKS   = {AT,JA,OG};
	public static final int[] ARAMAIC_BOOKS = {MT};
	
	/**
	 * Something like Vulgata
	 */
	public final String text_name;
	/**
	 * The name of the biblical book. E.g., Esther
	 */
	public String book_name;
	/**
	 * A Book contains chapters, a chapter paragraphs. The paragraphs are the sequences, we intend to align. 
	 */
	public ArrayList<Chapter> my_chapters = new ArrayList<Chapter>();
	
	/**
	 * Language Code according to plus.alignement.Sequence language encoding.
	 */
	public final int language;
	
	public Book(String _text_name, String _book_name, int _language) {
		this.book_name = _book_name;
		this.text_name = _text_name;
		this.language  = _language;
	}
	
	/**
	 * Copies the containers meta data, but not the content.
	 * @param b
	 */
	public Book(Book b) {
		this(b.text_name,b.book_name,b.language);
	}
	
	public String toString(){
		String ret = "Book "+book_name+" of "+text_name+" in language "+Book.get_language_string(language)+"\n";
		for(Chapter c : my_chapters) {
			ret +="\n"+c.toString();
		}
		return ret;
	}
	public String to_single_line_string(){
		String ret = "";
		for(Chapter c : my_chapters) {
			ret += c.to_single_line_string();
		}
		return ret;
	}
	public ArrayList<String> to_list(){
		ArrayList<String> ret = new ArrayList<>(1000);
		for(Chapter c : my_chapters) {
			c.to_list(ret);
		}
		return ret;
	}

	public int size() {
		int size = 0;
		for(Chapter c : my_chapters) {
			size+=c.size();
		}
		return size;
	}
}
