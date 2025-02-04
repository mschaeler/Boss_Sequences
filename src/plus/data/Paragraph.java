package plus.data;

import java.util.ArrayList;

/**
 * The text is without any String pre-preprocessing. Pure text, with commas, Capitalization etc.
 * The token sequence is pre-processed in an arbitrary way.
 * 
 * @author b1074672
 *
 */
public class Paragraph {
	public final Chapter my_chapter;
	public final String paragraph_name;
	public final String my_paragraph_as_text;
	
	public Paragraph(Chapter _c, String _paragraph_name, String _paragraph){
		this.my_chapter 			  = _c;
		this.paragraph_name 		  = _paragraph_name;
		this.my_paragraph_as_text 	  = _paragraph;
	}
	
	public String toString(){
		String ret = "Paragraph "+paragraph_name;
		ret += "\t"+my_paragraph_as_text;
		return ret;
	}
	public String to_single_line_string(){
		return my_paragraph_as_text;
	}
	public void to_list(ArrayList<String> ret) {
		ret.add(my_paragraph_as_text);
	}
	public int size() {
		return my_paragraph_as_text.split(" ").length;
	}
}
