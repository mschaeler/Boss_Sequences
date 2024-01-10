package pan;

import java.util.ArrayList;

import boss.load.Importer;
import plus.data.Book;
import plus.data.Chapter;
import plus.data.Paragraph;

/**
 * 
 * @author Martin
 *
 */
public interface Data {
	
	boolean verbose = false;
	
	/**
	 * L{susp_id,source_id}
	 */
	String[][] plagiats = {
			{"00228", "05889"}
			,{"00574", "06991"}
			,{"00574", "06586"}
			,{"00815", "01537"}
			,{"04617", "01107"}
			,{"10751", "06521"}
			,{"02161", "06392"}
			,{"02841", "10886"}
			,{"04032", "07742"}
			,{"04032", "02661"}
			,{"04032", "07640"}
			,{"04751", "08779"}
			,{"04953", "00732"}
			,{"08405", "10603"}
			,{"09029", "03302"}
			,{"09922", "10065"}
			,{"08405", "10603"}
			,{"10497", "06489"}
	};
	/**
	 * L[sus_begin,sus_end,src_begin,src_end] - in Characters 
	 * Note, minor corrections in offsets to have whole words. Seems to be an issue of how to count whitespaces.
	 */
	int[][] offsets = {
			{1925,1925+520,581,581+490-1}			//id = 0
			,{1120,1120+428,42881,42881+459}		//id = 1
			,{3899-1,3899+1668,3674,3674+2050}		//id = 2
			,{5386,5386+175,11322,11322+228}		//id = 3
			,{5689-4,5689-4+1745,10173,10173+1505}	//id = 4
			,{2241,2241+229,11396,11396+226-1}		//id = 5
			,{16068-3,16068+267-3,1290,1290+277-40}	//id = 6
			,{3342,3342+205,4679+4,4679+277-20}		//id = 7
			,{4693,4693+404,2837,2837+496}			//id = 8
			,{5883,5883+748,17454,17454+769}		//id = 9
			,{7878-1,7878+2338-2,32313,32313+2249}	//id = 10
			,{4192-1,4192+1839,12477,12477+2018}	//id = 11
			,{1450,1450+3514,13034,13034+2878}		//id = 12
			,{18351-1,18351+265-3,2261,2261+226}	//id = 13
			,{2699,2699+268,7222,7222+267}			//id = 14
			,{14034,14034+278,0+1,0+272}			//id = 15
			,{18351-1,18351+265-2,2261,2261+226}	//id = 16
			,{3786,3786+210,37876,37876+250}		//id = 17
	};
	
	static int[] offsets_in_tokens(int pair_id) {
		String pan_id_susp = plagiats[pair_id][susp];
		String pan_id_src = plagiats[pair_id][src];
		
		String path = Importer.PAN11_PREFIX_SUSP+pan_id_susp+".txt";
		Book plagiat = Importer.get_book_pan11(path, Book.LANGUAGE_ENGLISH);
		path = Importer.PAN11_PREFIX_SRC+pan_id_src+".txt";
		Book original = Importer.get_book_pan11(path, Book.LANGUAGE_ENGLISH);
		
		ArrayList<Book> excerpts = load(pair_id);
		
		//TODO get indices
		System.err.println("TODO");
		
		return null;
	}
	
	int susp = 0;
	int src = 1;
	static ArrayList<Book> load(int pair_id) {
		if(verbose) System.out.println("***** Loading pair "+pair_id);
		String pan_id_susp = plagiats[pair_id][susp];
		String pan_id_src = plagiats[pair_id][src];
		
		ArrayList<Book> ret_values = new ArrayList<Book>(2);
		
		String path = Importer.PAN11_PREFIX_SUSP+pan_id_susp+".txt";
		Book plagiat = Importer.get_book_pan11(path, Book.LANGUAGE_ENGLISH);
		path = Importer.PAN11_PREFIX_SRC+pan_id_src+".txt";
		Book original = Importer.get_book_pan11(path, Book.LANGUAGE_ENGLISH);
		
		//System.out.println(plagiat.to_single_line_string());
		//System.out.println(original.to_single_line_string());
		
		String excerpt_susp = plagiat.to_single_line_string().substring(offsets[pair_id][0], offsets[pair_id][1]).trim();
		String excerpt_src  = original.to_single_line_string().substring(offsets[pair_id][2], offsets[pair_id][3]).trim();
		
		//System.out.println("Relevant parts");
		
		if(verbose) System.out.println("Susp\t"+excerpt_susp);
		if(verbose) System.out.println("Src\t"+excerpt_src);
		
		//Make the plagiat a book
		Book ret = new Book(plagiat);
		ret.book_name = plagiats[pair_id][susp]+" excerpt ["+offsets[pair_id][0]+","+offsets[pair_id][1]+"]";
		Chapter content = new Chapter(ret, "Plagiat excerpt");
		ret.my_chapters.add(content);
		Paragraph p = new Paragraph(content, "Copy", excerpt_susp);
		content.my_paragraphs.add(p);
		if(verbose) System.out.println(ret);
		ret_values.add(ret);
		
		if(verbose) System.out.println();
		
		//Make the source a book
		ret = new Book(original);
		ret.book_name = plagiats[pair_id][src]+" excerpt ["+offsets[pair_id][2]+","+offsets[pair_id][3]+"]";
		content = new Chapter(ret, "Source excerpt");
		ret.my_chapters.add(content);
		p = new Paragraph(content, "Org", excerpt_src);
		content.my_paragraphs.add(p);
		if(verbose) System.out.println(ret);
		ret_values.add(ret);
		
		
		return ret_values;
		
	}
	
	static ArrayList<Book> load_entire_documents(int pair_id) {
		if(verbose) System.out.println("***** Loading pair "+pair_id);
		String pan_id_susp = plagiats[pair_id][susp];
		String pan_id_src = plagiats[pair_id][src];
		
		ArrayList<Book> ret_values = new ArrayList<Book>(2);
		
		String path = Importer.PAN11_PREFIX_SUSP+pan_id_susp+".txt";
		Book plagiat = Importer.get_book_pan11(path, Book.LANGUAGE_ENGLISH);
		path = Importer.PAN11_PREFIX_SRC+pan_id_src+".txt";
		Book original = Importer.get_book_pan11(path, Book.LANGUAGE_ENGLISH);
		
		ret_values.add(plagiat);
		ret_values.add(original);
		
		return ret_values;
		
	}
	
	static ArrayList<Book>[] load_all_entire_documents(){
		@SuppressWarnings("unchecked")
		ArrayList<Book>[] all_pairs = new ArrayList[plagiats.length];
		for(int pair_id=0;pair_id<plagiats.length;pair_id++) {
			ArrayList<Book> pair = load_entire_documents(pair_id);
			all_pairs[pair_id] = pair;
		}
		return all_pairs;
	}
	
	static ArrayList<Book>[] load_all_plagiarism_excerpts(){
		@SuppressWarnings("unchecked")
		ArrayList<Book>[] all_pairs = new ArrayList[plagiats.length];
		for(int pair_id=0;pair_id<plagiats.length;pair_id++) {
			ArrayList<Book> pair = load(pair_id);
			all_pairs[pair_id] = pair;
		}
		return all_pairs;
	}
	
	public static void main(String[] args) {
		ArrayList<Book>[] all_pair = load_all_plagiarism_excerpts();
		for(ArrayList<Book> pair : all_pair) {
			System.out.println("Next pair ************************************************");
			System.out.println(pair.get(0));
			System.out.println();
			System.out.println(pair.get(1));
		}
	}
	
	
	/**
	<?xml version="1.0" ?><document reference="suspicious-document00228.txt">
    	<feature name="plagiarism" type="simulated" this_language="en" this_offset="1925" this_length="520" source_reference="source-document05889.txt" source_language="en" source_offset="581" source_length="490"/>
  	</document>
	 */
	String suspicious_document00228_1925_520 = "To be sure,\r\n" + 
			"I know him much better than she does. I was with the Warths all day yesterday and we played\r\n" + 
			"Place for the King, I got causght by Robert and I had to give him a kiss. Ema said it didn't\r\n" + 
			"count because I allowed him to catch me, but Robert was mean and said she was just a nuisance\r\n" + 
			"and that she spoils the fun for others. Acttually, he's right, but there is another one who is\r\n" + 
			"just like that, too. However, I hope Ema didn't tell Dora that I kissed him. If so, everyone will\r\n" + 
			"be aware and I would be embarrassed.";
	String source_document05889_581_490 = "Anyhow she does not know him the way I do. Yesterday I was with the\r\n" + 
			"Warths all day. We played Place for the King and Robert caught me and I had to give him a kiss.\r\n" + 
			"And Erna said, that doesn't count, for I had let myself be caught. But Robert got savage and\r\n" + 
			"said: Erna is a perfect nuisance, she spoils everyone's pleasure. He's quite right, but there's\r\n" + 
			"some one else just as bad. But I do hope Erna has not told Dora about the kiss. If she has\r\n" + 
			"everyone will know and I shouldn't like that";
	/**
	 <?xml version="1.0" ?><document reference="suspicious-document00574.txt">
  		<feature name="plagiarism" type="simulated" this_language="en" this_offset="1120" this_length="428" source_reference="source-document06991.txt" source_language="en" source_offset="42881" source_length="459"/>
    	<feature name="plagiarism" type="simulated" this_language="en" this_offset="3899" this_length="1668" source_reference="source-document06586.txt" source_language="en" source_offset="3674" source_length="2050"/>
  	 </document>
	*/
	String suspicious_document00574_1120_428 = "Why take the risk of not winning a race, but simply ask Planchette? Of course, this would\r\n" + 
			"only work if this were universal, and if everybody is a winner, who would be a loser? Then Planchette\r\n" + 
			"would stop virtually all speculation. Planchette would start a new time of total success. There\r\n" + 
			"is not doubt that Mr. Charles Wyndham consulted Planchette before\r\n" + 
			"making The Fring of Society, wand was rewarded by trusting Planchette.";
	String source_documentdocument06991_42881_459 = "Why run any chance of losing\r\n" + 
			"on a race, but simply ask Planchette? Only, by the way, if this were universal, and if everyone\r\n" + 
			"is to win, who is to lose? Thus Planchette would put an end to nearly all speculation. Planchette\r\n" + 
			"would inaugurate a new era of complete and unqualified success. No doubt Mr. CHARLES WYNDHAM\r\n" + 
			"consulted Planchette before producing The Fringe of Society, and is in consequence being amply\r\n" + 
			"rewarded for placing his trust in Planchette.";
	
	String suspicious_document00574_3899_1668 = "The manuscript was transcribed by Miss Elstob in 1710. Mr. George Ballard had a copy of the\r\n" + 
			"transcript. But where is the original now?\r\n" + 
			"\r\n" + 
			"3.\r\n" + 
			"\r\n" + 
			"\"A Memorandum-book in the handwriting of Paul Bowes, Esq, son of Sir Thomas Bowes, of London,\r\n" + 
			"and of Bromley Hall, Essex, Knight and dated 1673.\"\r\n" + 
			"\r\n" + 
			"In 1783, this manuscript was owned by a gentleman named Broke, who lived in Nacton in Suffolk. He\r\n" + 
			"was a descendant of the Bowes family.\r\n" + 
			"\r\n" + 
			"But I have not been able to trace the manuscript further\r\n" + 
			"\r\n" + 
			"4. \"The Negotiations of Thomas Wolsey, Cardinal. This manuscript, which was very valuable, was\r\n" + 
			"in the collection of Dr. Farmer, who wrote on the fly leaf: \"I believe several of the Letters\r\n" + 
			"and State Papers in this volume have not been published; three or four are printed in the collections\r\n" + 
			"at the end of Dr. Fiddes' Life of Wolsey, from a manuscript in the\r\n" + 
			"Yelverton Library.\"\r\n" + 
			"\r\n" + 
			"if I remember correctly, the late Richard Heber\r\n" + 
			"later obtained this unusual and important book.\r\n" + 
			"\r\n" + 
			"Sadly, Heber's collection of manuscripts was broken up.\r\n" + 
			"\r\n" + 
			"Edward F. Rimbault\r\n" + 
			"\r\n" + 
			"Minor Queries.\r\n" + 
			"\r\n" + 
			"Chantrey's Sleeping Children in Lichfield Cathedral. Mr. Peter Cunningham stated in the Literary\r\n" + 
			"Gazette published June 5 that the composition was created by Chantrey\r\n" + 
			"and Stothard. However, as a regular reader of the \"NOTES AND QUERIES\",\r\n" + 
			"I feel it is necessary to point out that absent evidence from Mr.\r\n" + 
			"Cunningham that Stothard shares in the creation of the piece of sculpture,\r\n" + 
			"it is generally attributed only to Chantrey.\r\n" + 
			"\r\n" + 
			"PLECTRUM\r\n" + 
			"\r\n" + 
			"Viscount Dundee's\r\n" + 
			"Ring.--The engraving on the ring is described in the Letters of John\r\n" + 
			"Grahame of Claverhouse, Viscount of Dundee, which was printed for\r\n" + 
			"the Bannatyne Club in 1826.";
	String source_document06586_3674_2050 = "The MS. was transcribed by Miss Elstob in 1710, and\r\n" + 
			"a copy of her transcript was in the possession of Mr. George Ballard. Where now is the original?\r\n" + 
			" \r\n" + 
			"3. \"A Memorandum-book in the handwriting of Paul Bowes, Esq., son of Sir Thomas Bowes, of London,\r\n" + 
			"and of Bromley Hall, Essex, Knight, and dated 1673.\" In 1783 this MS., which contains some\r\n" + 
			"highly interesting and important information, was in the possession of a gentleman named Broke,\r\n" + 
			"of Nacton in Suffolk, a descendant from the Bowes family; but I have not been able to trace\r\n" + 
			"it further.\r\n" + 
			" \r\n" + 
			"4. \"The Negotiations of Thomas Wolsey, Cardinall.\" This valuable MS. was in the collection\r\n" + 
			"of Dr. Farmer, who wrote on the fly-leaf,--\r\n" + 
			" \r\n" + 
			"    \"I believe several of the Letters and State Papers in this volume have\r\n" + 
			"    not been published; three or four are printed in the collections at the\r\n" + 
			"    end of Dr. Fiddes' Life of Wolsey, from a MS. in the Yelverton\r\n" + 
			"    Library.\"\r\n" + 
			" \r\n" + 
			"If I remember rightly, the late Richard Heber afterwards came into the possession of this curious\r\n" + 
			"and important volume. It is lamentable to think of the dispersion of poor Heber's manuscripts.\r\n" + 
			" \r\n" + 
			"EDWARD F. RIMBAULT.\r\n" + 
			" \r\n" +  
			" \r\n" + 
			"Minor Queries.\r\n" + 
			" \r\n" + 
			"Chantrey's Sleeping Children in Lichfield Cathedral.--In reference to a claim recently put\r\n" + 
			"forth on behalf of an individual to the merit of having designed and executed this celebrated\r\n" + 
			"monument, Mr. Peter Cunningham says (Literary Gazette, June 5.),--\"The merit of the composition\r\n" + 
			"belongs to Chantrey and Stothard.\" As a regular reader of the \"NOTES AND QUERIES,\" I shall\r\n" + 
			"feel obliged to Mr. Cunningham (whose name I am always glad to see as a correspondent) if he\r\n" + 
			"will be kind enough to inform me on what evidence he founds the title of Mr. Stothard to a\r\n" + 
			"share of the merit of a piece of sculpture, which is so generally attributed to the genius\r\n" + 
			"of Chantrey?\r\n" + 
			" \r\n" + 
			"PLECTRUM.\r\n" + 
			" \r\n" + 
			"Viscount Dundee's Ring.--In the Letters of John Grahame of Claverhouse, Viscount of Dundee,\r\n" + 
			"printed for the Bannatyne Club in 1826, is a description and engraving of a ring containing\r\n" + 
			"some of Ld.";
}
