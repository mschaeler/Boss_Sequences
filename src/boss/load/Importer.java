package boss.load;

import java.io.*;
import java.util.ArrayList;
import java.util.Arrays;

import boss.lexicographic.StopWords;
import boss.util.Config;
import boss.util.Util;
import plus.data.*;

public class Importer {
	// static final String path_prefix = "C:\\Users\\b1074672\\Nextcloud\\PLUS\\code\\BibleData";//TODO Remove upon final merge into one project
	static final String path_prefix = "./";
	
	public static final String AT_PATH = null;
	public static final String JA_PATH = null;
	public static final String MT_PATH = null;
	public static final String OG_PATH = null;
	public static final String Vg_PATH = null;
	public static final String VL_PATH = null;
	//German versions below
	public static final String LUTHER_PATH 		= path_prefix+"data/de/luther.txt";
	public static final String ELBERFELDER_PATH = path_prefix+"data/de/elberfelder.txt";
	public static final String NE_PATH 			= path_prefix+"data/de/ne.txt";
	public static final String SCHLACHTER_PATH 	= path_prefix+"data/de/schlachter.txt";
	public static final String VOLXBIBEL_PATH 	= path_prefix+"data/de/volxbibel.txt";
	//English versions below
	public static final String ESV_PATH 		= path_prefix+"data/en/esv.txt";
	public static final String KING_JAMES_PATH 	= path_prefix+"data/en/king_james_bible.txt";

	//Pan11 documents 
	public static final String PAN11_PREFIX_SRC = path_prefix+"data/pan11/01-manual-obfuscation-highjac/src/source-document";
	public static final String PAN11_PREFIX_SUSP = path_prefix+"data/pan11/01-manual-obfuscation-highjac/susp/suspicious-document";
	public static final String[] PAN11_SRC = {PAN11_PREFIX_SRC+"00732.txt", PAN11_PREFIX_SRC+"01107.txt", PAN11_PREFIX_SRC+"01537.txt", PAN11_PREFIX_SRC+"02661.txt", PAN11_PREFIX_SRC+"03302.txt", PAN11_PREFIX_SRC+"05889.txt", PAN11_PREFIX_SRC+"06392.txt", PAN11_PREFIX_SRC+"06489.txt", PAN11_PREFIX_SRC+"06521.txt", PAN11_PREFIX_SRC+"06586.txt", PAN11_PREFIX_SRC+"06991.txt", PAN11_PREFIX_SRC+"07640.txt", PAN11_PREFIX_SRC+"07742.txt", PAN11_PREFIX_SRC+"08779.txt", PAN11_PREFIX_SRC+"10065.txt", PAN11_PREFIX_SRC+"10603.txt", PAN11_PREFIX_SRC+"10886.txt"};
	public static final String[] PAN11_SUSP = {PAN11_PREFIX_SUSP+"00228.txt", PAN11_PREFIX_SUSP+"00574.txt", PAN11_PREFIX_SUSP+"00815.txt", PAN11_PREFIX_SUSP+"02161.txt", PAN11_PREFIX_SUSP+"02841.txt", PAN11_PREFIX_SUSP+"04032.txt", PAN11_PREFIX_SUSP+"04617.txt", PAN11_PREFIX_SUSP+"04751.txt", PAN11_PREFIX_SUSP+"04953.txt", PAN11_PREFIX_SUSP+"08405.txt", PAN11_PREFIX_SUSP+"09029.txt", PAN11_PREFIX_SUSP+"09922.txt", PAN11_PREFIX_SUSP+"10497.txt", PAN11_PREFIX_SUSP+"10751.txt"};
	
	public static final String INPUT_ENCODING  = "UTF8";
	public static final String FIELD_SEPERATOR = "�";//looks like s space, but it is not...
	public static final String FIELD_SEPERATOR_PAN11 = " ";//looks like s space, but it is not...
	
	/**
	 * Expected file structure:
	 * Schlachter 2000 Esther Copyright � 2000 Genfer Bibelgesellschaft
	 *--1
     *1�Und es geschah in den Tagen des Ahasveros[1] � desselben Ahasveros, der von Indien bis �thiopien �ber 127 Provinzen regierte �,
	 * @param path
	 * @return
	 */
	private static ArrayList<String[]> get_data_from_file(String path){
		ArrayList<String[]> lines = new ArrayList<String[]>(1000);
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path), INPUT_ENCODING));
			String line = br.readLine();
			lines.add(line.split(" "));
			while ((line = br.readLine()) != null) {
				//System.out.println(line);
				String[] raw_tokens = line.split(FIELD_SEPERATOR,2);
				if(raw_tokens.length==1){
					raw_tokens = line.split(" ", 2);//Try alternative field seperator
				}
				lines.add(raw_tokens);
			}
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return lines;
	}


	/**
	 * 
	 * @param path
	 * @return
	 */
	private static ArrayList<String[]> get_data_from_file_pan11(String path){
		ArrayList<String[]> lines = new ArrayList<String[]>(1000);
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(path), INPUT_ENCODING));
			String line;// = br.readLine();
			// String[] meta_info = {"PAN11", path, "PAN11"};
			// lines.add(meta_info);
			// lines.add(line.split(" "));
			while ((line = br.readLine()) != null) {
				//System.out.println(line);
				String[] raw_tokens = line.split(FIELD_SEPERATOR_PAN11);
				// if(raw_tokens.length==1){
				// 	raw_tokens = line.split(" ", 2);//Try alternative field seperator
				// }
				lines.add(raw_tokens);
			}
			br.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		return lines;
	}
	
	
	@SuppressWarnings("unused") //mainly for debugging
	private static String out(ArrayList<String[]> list){
		StringBuffer buffer = new StringBuffer();
		for(String[] array : list) {
			buffer.append("\n");
			//buffer.append(array[0]);
			buffer.append(Arrays.deepToString(array));
		}
		return buffer.toString();
	}
	
	private static Book to_book_pan11(ArrayList<String[]> raw_book, int language, String filename) {
		// String meta_info[] = raw_book.get(0);
		String[] meta_info = {"PAN11", filename, "PAN11"};
		Book b = new Book(get_text_name(meta_info), get_book_name(meta_info), language);
		Chapter current_chapter = new Chapter(b, "proxy");
		b.my_chapters.add(current_chapter);
		int para_number = 0;
		for(int i=0;i<raw_book.size();i++) {//ignore first line 
			String[] line = raw_book.get(i);
			if(is_chapter_start(line)) {
				current_chapter = new Chapter(b, get_chapter_name(line));
				b.my_chapters.add(current_chapter);
				para_number = 0;
			}else{
				if(line.length>0) {
					StringBuilder result = new StringBuilder(line[0]);
					for (int j = 1; j < line.length; j++) {
						result.append(" ").append(line[j]);
					}
					Paragraph p = new Paragraph(current_chapter, String.valueOf(para_number), result.toString());
					current_chapter.my_paragraphs.add(p);
					para_number++;
				}
			}
		}
		return b;
	}

	public static Book to_book(ArrayList<String[]> raw_book, int language) {
		String meta_info[] = raw_book.get(0);
		Book b = new Book(get_text_name(meta_info), get_book_name(meta_info), language);
		Chapter current_chapter = null;
		for(int i=1;i<raw_book.size();i++) {//ignore first line 
			String[] line = raw_book.get(i);
			if(is_chapter_start(line)) {
				current_chapter = new Chapter(b, get_chapter_name(line));
				b.my_chapters.add(current_chapter);
			}else{
				Paragraph p = new Paragraph(current_chapter, line[0], line[1]);
				current_chapter.my_paragraphs.add(p);
			}
		}
		return b;
	}
	
	private static String get_chapter_name(String[] line) {
		return line[0].replace(CHAPTER_START, "");
	}

	static final String CHAPTER_START = "--";
	private static boolean is_chapter_start(String[] line) {
		if(line.length>0)
			return line[0].startsWith(CHAPTER_START);
		else
			return false;
	}

	/**
	 * For our data should always return 'Esther' or 'Ester'
	 * @param meta_info
	 * @return
	 */
	private static String get_book_name(String[] meta_info) {
		return meta_info[2];
	}

	private static String get_text_name(String[] meta_info) {
		return meta_info[0]+" "+meta_info[1];
	}

	public static void main(String[] args) {
		//ArrayList<String[]> lines = get_data_from_file(SCHLACHTER_PATH);
		//System.out.println(out(lines));
		Book b = get_book(Book.VOLXBIBEL);
		System.out.println(b);
	}

	private static Book get_book(String file_path, int language) {
		ArrayList<String[]> raw_book = get_data_from_file(file_path);
		Book b = to_book(raw_book, language);
		return b;
	}

	public static Book get_book_pan11(String file_path, int language) {
		ArrayList<String[]> raw_book = get_data_from_file_pan11(file_path);
		Book b = to_book_pan11(raw_book, language, file_path);
		return b;
	}
	
	
	static Book get_book(int book_id) {
		Book b;
		
		if(book_id == Book.AT) {
			b = get_book(AT_PATH, Book.LANGUAGE_OLD_GREEK); 
		}else if(book_id == Book.JA){
			b = get_book(JA_PATH, Book.LANGUAGE_OLD_GREEK); 
		}else if(book_id == Book.MT){
			b = get_book(MT_PATH, Book.LANGUAGE_ARAMAIC); 
		}else if(book_id == Book.OG){
			b = get_book(OG_PATH, Book.LANGUAGE_OLD_GREEK); 
		}else if(book_id == Book.Vg){
			b = get_book(Vg_PATH, Book.LANGUAGE_LATIN); 
		}else if(book_id == Book.VL){
			b = get_book(VL_PATH, Book.LANGUAGE_LATIN); 
		}else if(book_id == Book.LUTHER){
			b = get_book(LUTHER_PATH, Book.LANGUAGE_GERMAN); 
		}else if(book_id == Book.ELBERFELDER){
			b = get_book(ELBERFELDER_PATH, Book.LANGUAGE_GERMAN); 
		}else if(book_id == Book.NE){
			b = get_book(NE_PATH, Book.LANGUAGE_GERMAN); 
		}else if(book_id == Book.SCHLACHTER){
			b = get_book(SCHLACHTER_PATH, Book.LANGUAGE_GERMAN); 	
		}else if(book_id == Book.VOLXBIBEL){
			b = get_book(VOLXBIBEL_PATH, Book.LANGUAGE_GERMAN); 	
		}else if(book_id == Book.ESV){
			b = get_book(ESV_PATH, Book.LANGUAGE_ENGLISH); 	
		}else if(book_id == Book.KING_JAMES){
			b = get_book(KING_JAMES_PATH, Book.LANGUAGE_ENGLISH); 	
		}else{
			System.err.println("Unknown book: "+book_id);
			b = null;
		}
		return b;
	}

	public static Book get_book_pan11(int id, boolean src) {
		Book b;
		if (src) {
			b = get_book_pan11(PAN11_SRC[id], Book.LANGUAGE_ENGLISH);
		} else {
			b = get_book_pan11(PAN11_SUSP[id], Book.LANGUAGE_ENGLISH);
		}
		return b;
	}
}
