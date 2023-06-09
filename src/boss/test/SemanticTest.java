package boss.test;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;

import boss.embedding.Embedding;
import boss.embedding.MatchesWithEmbeddings;
import boss.hungarian.HungarianExperiment;
import boss.lexicographic.BasicTokenizer;
import boss.lexicographic.TokenizedParagraph;
import boss.lexicographic.Tokenizer;
import boss.load.ImporterAPI;
import boss.semantic.Sequence;
import boss.util.Pair;
import plus.data.Book;
import plus.data.Chapter;
import plus.data.Paragraph;

public class SemanticTest {	
	static int MAPPING_GRANUALRITY;
	static final int GRANULARITY_PARAGRAPH_TO_PARAGRAPH = 0;
	static final int GRANULARITY_PARAGRAPH_TO_CHAPTER   = 1;
	static final int GRANULARITY_CHAPTER_TO_CHAPTER     = 2;
	static final int GRANULARITY_BOOK_TO_BOOK     		= 3;
	
	public static void main(String[] args) {
		final int[] k_s= {3,4,5,6};
		final double threshold = 0.7;
		
		ArrayList<Book> books = ImporterAPI.get_all_english_books();
		//ArrayList<Book> books = ImporterAPI.get_all_german_books();
		
		for(int k : k_s) {
			ArrayList<HungarianExperiment> hes = prepare_experiment(books,k,threshold);
			for(HungarianExperiment he :hes) {
				//he.run_baseline();
				he.run_pruning();
			}
		}
		
		/*String file_path = Embedding.get_embedding_path(books.get(0).language);
		ArrayList<MatchesWithEmbeddings> embeddings = MatchesWithEmbeddings.load(file_path);
		
		for(double threshold = 0.0; threshold<1.0;threshold+=0.1) {
			no_match_words(embeddings, threshold);	
		}*/	
	}
	
	/*static void prepare_experiment_2(Book b_1, Book b_2) {
		System.out.println("SemanticTest.prepare_experiment() [START]");
		ArrayList<Book> temp = new ArrayList<Book>(2); temp.add(b_1); temp.add(b_2);//to get all tokens
		ArrayList<Book> tokenized_books = Tokenizer.run(temp, new BasicTokenizer());
		
		ArrayList<String> all_tokens_ordered = Sequence.get_ordered_token_list(Sequence.get_unique_tokens(tokenized_books));
		HashMap<String, Integer> token_ids = strings_to_int(all_tokens_ordered);
		String file_path = Embedding.get_embedding_path(b_1.language);
		HashMap<Integer, double[]> embedding_vector_index = create_embedding_vector_index(token_ids,all_tokens_ordered,file_path);
		
		//System.out.println(all_tokens_ordered.toString());
		
		ArrayList<String[]> raw_book_1 = new ArrayList<String[]>(100); 
		ArrayList<String[]> raw_book_2 = new ArrayList<String[]>(100); 
		
		// MAPPING_GRANUALRITY = GRANULARITY_CHAPTER_TO_CHAPTER;
		MAPPING_GRANUALRITY = GRANULARITY_CHAPTER_TO_CHAPTER;
		
		get_tokens(tokenized_books.get(0), tokenized_books.get(1), raw_book_1, raw_book_2);
		
		ArrayList<int[]> raw_paragraphs_b1  = encode(raw_book_1, token_ids);
		ArrayList<int[]> raw_paragraphs_b2  = encode(raw_book_2, token_ids);
		
		HungarianExperiment exp = new HungarianExperiment(raw_paragraphs_b1, raw_paragraphs_b2, k, threshold, embedding_vector_index);
		exp.run_baseline();
		//exp.run_idea_nikolaus();
		System.out.println("SemanticTest.prepare_experiment() [DONE]");
	}*/
	
	/**
	 * Returns the int-token representation based on the current MAPPING_GRANUALRITY
	 * 
	 * @param tokenized_book_1
	 * @param tokenized_book_2
	 * @param raw_book_1 - return value
	 * @param raw_book_2 - return value
	 */
	static void get_tokens(final Book tokenized_book_1, final Book tokenized_book_2, final ArrayList<String[]> raw_book_1, final ArrayList<String[]> raw_book_2) {
		if(MAPPING_GRANUALRITY == GRANULARITY_PARAGRAPH_TO_CHAPTER) {
			get_tokens_paragraph_to_chapter(tokenized_book_1, tokenized_book_2, raw_book_1, raw_book_2);
		}else if(MAPPING_GRANUALRITY == GRANULARITY_PARAGRAPH_TO_PARAGRAPH){
			get_tokens(tokenized_book_1, raw_book_1);
			get_tokens(tokenized_book_2, raw_book_2);
		}else if(MAPPING_GRANUALRITY == GRANULARITY_CHAPTER_TO_CHAPTER){
			get_tokens_chapter(tokenized_book_1, raw_book_1);
			get_tokens_chapter(tokenized_book_2, raw_book_2);
		}else if(MAPPING_GRANUALRITY == GRANULARITY_BOOK_TO_BOOK){
			get_tokens_book(tokenized_book_1, raw_book_1);
			get_tokens_book(tokenized_book_2, raw_book_2);
		}else{
			System.err.println("Unknown granulartiy");
		}
	}
	
	static ArrayList<HungarianExperiment> prepare_experiment(ArrayList<Book> books, final int k, final double threshold) {
		System.out.println("SemanticTest.prepare_experiment() [START]");
		ArrayList<HungarianExperiment> ret = new ArrayList<HungarianExperiment>(books.size());
		
		ArrayList<Book> tokenized_books = Tokenizer.run(books, new BasicTokenizer());
		ArrayList<String> all_tokens_ordered = Sequence.get_ordered_token_list(Sequence.get_unique_tokens(tokenized_books));
		HashMap<String, Integer> token_ids = strings_to_int(all_tokens_ordered);
		String file_path = Embedding.get_embedding_path(books.get(0).language,false);
		HashMap<Integer, double[]> embedding_vector_index = create_embedding_vector_index(token_ids,all_tokens_ordered,file_path);
		
		// MAPPING_GRANUALRITY = GRANULARITY_CHAPTER_TO_CHAPTER;
		MAPPING_GRANUALRITY = GRANULARITY_CHAPTER_TO_CHAPTER;
		
		//For each pair of books (i,j)
		for(int i=0;i<tokenized_books.size();i++) {
			Book tokenized_book_1 = tokenized_books.get(i);
			for(int j=i+1;j<tokenized_books.size();j++) {
				Book tokenized_book_2 = tokenized_books.get(j);	
				System.out.println("New book pair "+tokenized_book_1.text_name+" vs. "+tokenized_book_2.text_name);
				
				ArrayList<String[]> raw_book_1 = new ArrayList<String[]>(100); 
				ArrayList<String[]> raw_book_2 = new ArrayList<String[]>(100); 
				
				get_tokens(tokenized_book_1, tokenized_book_2, raw_book_1, raw_book_2);
				
				ArrayList<int[]> raw_paragraphs_b1  = encode(raw_book_1, token_ids);
				ArrayList<int[]> raw_paragraphs_b2  = encode(raw_book_2, token_ids);
				
				HungarianExperiment exp = new HungarianExperiment(raw_paragraphs_b1, raw_paragraphs_b2, k, threshold, embedding_vector_index);
				ret.add(exp);
			}
		}
		System.out.println("SemanticTest.prepare_experiment() [DONE]");
		return ret;
	}
	
	private static void get_tokens_paragraph_to_chapter(final Book b_1, final Book b_2, final ArrayList<String[]> raw_book_b1, final ArrayList<String[]> raw_book_b2){
		for(int c=0;c<b_1.my_chapters.size();c++) {
			Chapter chapter_b1 = b_1.my_chapters.get(c);
			Chapter chapter_b2 = b_2.my_chapters.get(c);
			
			String[] chapter_b2_as_string_array = null;
			{
				//inlined method to get one String[] representing the entire Chapter of b_2
				//The idea is to this string to each paragraph of the other book
				ArrayList<String> temp = new ArrayList<String>(1000);
				for(Paragraph p : chapter_b2.my_paragraphs) {
					TokenizedParagraph tp = (TokenizedParagraph) p;
					String[] paragraph_as_array = tp.last_intermediate_result();
					for(String s : paragraph_as_array) {
						temp.add(s);
					}
				}
				//Make it the desired String array
				chapter_b2_as_string_array = new String[temp.size()];
				for(int i=0;i<temp.size();i++) {
					chapter_b2_as_string_array[i]=temp.get(i);
				}
			}
			
			for(Paragraph p : chapter_b1.my_paragraphs) {
				TokenizedParagraph tp = (TokenizedParagraph) p;
				String[] temp = tp.last_intermediate_result();
				raw_book_b1.add(temp);
				raw_book_b2.add(chapter_b2_as_string_array);//This is the trick: we add here the entire chapter
				System.out.println(c+1);
			}
		}
	}

	private static HashSet<Integer> no_match_words(final ArrayList<MatchesWithEmbeddings> embeddings, final double threshold){
		HashSet<Integer> no_match_words = new HashSet<Integer>(100);
		for(int id=0;id<embeddings.size();id++) {
			ArrayList<Pair> sets_in_neighborhood = get_sets_above_similarity_threshold(embeddings, id, threshold);
			if(sets_in_neighborhood.isEmpty()) {
				no_match_words.add(id);
			}
			/*{
				MatchesWithEmbeddings mew = embeddings.get(id);
				System.out.println(mew.string_in_embedding+"\t"+mew.string_in_text+"\t"+to_tsv(sets_in_neighborhood));	
			}*/
		}
		//System.out.println("For theta="+threshold+": Out of "+embeddings.size()+" tokens, "+no_match_words.size()+" are no match words");
		System.out.println(threshold+"\t"+embeddings.size()+"\t"+no_match_words.size());
		
		return no_match_words;
	}
	private static String to_tsv(ArrayList<Pair> sets_in_neighborhood) {
		String ret = "";
		for(Pair p : sets_in_neighborhood) {
			ret+=p.similarity+"\t";
		}
		return ret;
	}

	private static ArrayList<Pair> get_sets_above_similarity_threshold(final ArrayList<MatchesWithEmbeddings> embeddings, final int id, final double threshold){
		ArrayList<Pair> result = new ArrayList<>(100);
		final double[] vec_1 = embeddings.get(id).vector;
		for(int other_id=0;other_id<embeddings.size();other_id++) {
			if(id==other_id) {
				continue;
			}
			final double[] vec_2 = embeddings.get(other_id).vector;
			final double dist = HungarianExperiment.dist(id,other_id,vec_1,vec_2);
			final double similarity = 1.0-dist;
			if(similarity>=threshold) {
				Pair p = new Pair(other_id, similarity);
				result.add(p);
			}
		}
		Collections.sort(result);
		return result;
	}
	
	private static HashMap<Integer, double[]> create_embedding_vector_index(HashMap<String, Integer> token_ids, ArrayList<String> all_tokens_ordered, String file_path) {
		ArrayList<MatchesWithEmbeddings> embeddings = MatchesWithEmbeddings.load(file_path);
		HashMap<Integer, double[]> index = new HashMap<Integer, double[]>(token_ids.size());
		HashMap<String,Integer> look_up_index = new HashMap<String,Integer>(embeddings.size());
		
		for(int i=0;i<embeddings.size();i++) {
			String s2 = embeddings.get(i).string_in_text;
			look_up_index.put(s2, i);
		}
		
		for(String token : all_tokens_ordered) {
			Integer id = token_ids.get(token);
			if(id==null) {
				System.err.println("Token id not found:" + token);
			}else{
				Integer position = look_up_index.get(token);
				if(position==null){
					System.err.println("Token not found:" + token);
				}else{
					MatchesWithEmbeddings mew = embeddings.get(position);
					index.put(id, mew.vector);
				}
			}
		}
		
		return index;
	}

	/**
	 * Ignores Chapter structure
	 * @param book
	 * @return
	 */
	private static ArrayList<String[]> get_tokens(Book book, ArrayList<String[]> result) {
		for(Chapter c : book.my_chapters) {
			for(Paragraph p : c.my_paragraphs) {
				TokenizedParagraph tp = (TokenizedParagraph) p;
				String[] temp = tp.last_intermediate_result();
				result.add(temp);
			}
		}
		return result;
	}
	
	/**
	 * Ignores Chapter structure
	 * @param book
	 * @return
	 */
	private static ArrayList<String[]> get_tokens_chapter(Book book, final ArrayList<String[]> result) {
		for(Chapter c : book.my_chapters) {
			ArrayList<String> chapter_tokens = new ArrayList<String>(1000);
			for(Paragraph p : c.my_paragraphs) {
				TokenizedParagraph tp = (TokenizedParagraph) p;
				String[] temp = tp.last_intermediate_result();
				for(String s : temp) {
					chapter_tokens.add(s);
				}
			}
			String[] chapter_tokens_array = new String[chapter_tokens.size()];
			for(int i=0;i<chapter_tokens_array.length;i++) {
				chapter_tokens_array[i] = chapter_tokens.get(i);
			}
			result.add(chapter_tokens_array);
		}
		return result;
	}
	
	/**
	 * Ignores Chapter structure
	 * @param book
	 * @return
	 */
	private static ArrayList<String[]> get_tokens_book(final Book book, final ArrayList<String[]> result) {
		ArrayList<String> temp = new ArrayList<String>(1000);
		for(Chapter c : book.my_chapters) {
			for(Paragraph p : c.my_paragraphs) {
				TokenizedParagraph tp = (TokenizedParagraph) p;
				String[] array = tp.last_intermediate_result();
				for(String s : array) {
					temp.add(s);
				}
			}
		}
		String[] book_tokens = new String[temp.size()];
		for(int i=0;i<book_tokens.length;i++) {
			book_tokens[i] = temp.get(i);
		}
		result.add(book_tokens);
		return result;
	}

	private static ArrayList<int[]> encode(ArrayList<String[]> raw_book, HashMap<String, Integer> token_ids) {
		ArrayList<int[]> result = new ArrayList<int[]>(raw_book.size());
		//we ignore Chapters
		
		for(String[] paragraph : raw_book) {
			int[] paragraph_token_ids = new int[paragraph.length];
			for(int i=0;i<paragraph_token_ids.length;i++) {
				String token = paragraph[i];
				Integer id = token_ids.get(token);
				if(id!=null) {
					paragraph_token_ids[i] = id.intValue();
				}else{
					System.err.println("id==null for "+token);
				}
			}
			result.add(paragraph_token_ids);
		}
		
		return result;
	}

	private static HashMap<String,Integer> strings_to_int(ArrayList<String> all_string_tokens){
		HashMap<String,Integer> encoding = new HashMap<String,Integer>(all_string_tokens.size());
		int id = 0;
		for(String s : all_string_tokens){
			encoding.put(s, id++);
		}
		return encoding;
	}

}
