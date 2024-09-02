package plus.data;

import java.util.ArrayList;

public class Chapter {
	//Reference to the book the chapter is in.
	public final Book book;
	/**
	 * Usually something like "1"
	 */
	public final String chapter_name;
	public ArrayList<Paragraph> my_paragraphs;
	
	public Chapter(Book _b, String _chapter_name){
		this.book = _b;
		this.chapter_name = _chapter_name;
		my_paragraphs = new ArrayList<Paragraph>();
	}
	/**
	 * 
	 * @param _b
	 * @param get_chapter_name
	 * @param get_text_in_paragraphs
	 */
	public Chapter(Book _b, String _chapter_name, String[][] _raw_text_paragraphs){
		this.book = _b;
		this.chapter_name = _chapter_name;
		my_paragraphs = new ArrayList<Paragraph>(_raw_text_paragraphs.length);
		int i = 1;//We ignore the first array entry, there one finds the text name
		for(;i<_raw_text_paragraphs.length;i++) {
			String[] raw_paragraph = _raw_text_paragraphs[i];//[0] paragraph name, [1] paragraph text
			Paragraph p = new Paragraph(this, raw_paragraph[0], raw_paragraph[1]);
			my_paragraphs.add(p);
		}
	}
	/**
	 * 
	 * @param _text_name The name of the text. E.g., Vulgata
	 * @param _book_name The name of the biblical book. E.g., Esther
	 * @param language 
	 * @param _chapter_name
	 * @param _raw_text_paragraphs
	 */
	public Chapter(String _text_name, String _book_name, int _language, String _chapter_name, String[][] _raw_text_paragraphs){
		this(new Book(_text_name, _book_name, _language), _chapter_name, _raw_text_paragraphs);
	}
	
	public Chapter(Book _b, String _chapter_name, ArrayList<Paragraph> _my_paragraphs) {
		this.book 		  = _b;
		this.chapter_name = _chapter_name;
		my_paragraphs 	  = _my_paragraphs;
	}
	
	public String toString() {
		String ret = chapter_name+" of book "+book.book_name+" in "+book.text_name+" corpus. Language is "+Book.get_language_string(book.language);
		for(Paragraph p : my_paragraphs) {
			ret += "\n"+p;
		}
		return ret;
	}
	public String to_single_line_string() {
		String ret = ""+my_paragraphs.get(0).to_single_line_string();
		for(int i=1;i<my_paragraphs.size();i++) {
			ret += " "+my_paragraphs.get(i).to_single_line_string();
		}
		
		return ret+" ";
	}
	public int size() {
		int size = 0;
		for(Paragraph p : my_paragraphs) {
			size+=p.size();
		}
		return size;
	}
}
