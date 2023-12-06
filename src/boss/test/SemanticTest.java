package boss.test;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;

import boss.embedding.Embedding;
import boss.embedding.MatchesWithEmbeddings;
import boss.hungarian.HungarianAlgorithmPranayImplementation;
import boss.hungarian.HungarianAlgorithmWiki;
import boss.hungarian.HungarianExperiment;
import boss.hungarian.HungarianKevinStern;
import boss.hungarian.HungarianKevinSternAlmpified;
import boss.hungarian.Solutions;
import boss.hungarian.StupidSolver;
import boss.lexicographic.BasicTokenizer;
import boss.lexicographic.TokenizedParagraph;
import boss.lexicographic.Tokenizer;
import boss.load.Importer;
import boss.load.ImporterAPI;
import boss.semantic.Sequence;
import boss.util.Pair;
import pan.PanResult;
import plus.data.Book;
import plus.data.Chapter;
import plus.data.Paragraph;

public class SemanticTest {	
	static int MAPPING_GRANUALRITY;
	static final int GRANULARITY_PARAGRAPH_TO_PARAGRAPH = 0;
	static final int GRANULARITY_PARAGRAPH_TO_CHAPTER   = 1;
	static final int GRANULARITY_CHAPTER_TO_CHAPTER     = 2;
	static final int GRANULARITY_BOOK_TO_BOOK     		= 3;
	
	static String result_path_pan = "./results/pan_results_"+System.currentTimeMillis()+".tsv";
	static String result_path_bible = "./results/bible_results_"+System.currentTimeMillis()+".tsv";
	
	public static void run_bible_experiments() {
		final int[] k_s= {3,4,5,6,7,8};
		final double threshold = 0.7;
		
		MAPPING_GRANUALRITY = GRANULARITY_BOOK_TO_BOOK;
		//MAPPING_GRANUALRITY = GRANULARITY_CHAPTER_TO_CHAPTER;
		int solution_enum = NAIVE; //SOLUTION, BASELINE, NAIVE
		result_path_bible+="_"+solution_enum;
		
		Solutions.dense_global_matrix_buffer = null;
		
		int num_repititions = 1;
		//English corpus
		ArrayList<Book> books = ImporterAPI.get_all_english_books();
		run(books, threshold, num_repititions, k_s, solution_enum);
		
		//Now the German corpus
		SemanticTest.embedding_vector_index_buffer = null;//Need to laod the German embediings. Before we had the English ones.
		books = ImporterAPI.get_all_german_books();
		for(int i=0;i<books.size();i++) {
			for(int j=i+1;j<books.size();j++) {
				Solutions.dense_global_matrix_buffer = null;
				ArrayList<Book> book_pair = new ArrayList<Book>(2);
				book_pair.add(books.get(i));
				book_pair.add(books.get(j));
				run(book_pair, threshold, num_repititions, k_s, solution_enum);
			}
		}
	}
	
	static final int SOLUTION = 0;
	static final int BASELINE = 1;
	static final int NAIVE    = 2;
	
	static boolean header_written = false;
	static void run(ArrayList<Book> books, double threshold, int num_repititions, int[] k_s, int solution_enum) {
		ArrayList<double[]> all_run_times = new ArrayList<double[]>();
		double[] run_times=null;
		
		for(int k : k_s) {
			boolean pan_embeddings = false;
			ArrayList<Solutions> solutions = prepare_solution(books,k,threshold, pan_embeddings);
			for(Solutions s : solutions) {
				int repitions = 0;
				double run_time = 0;
				while(repitions++<num_repititions) {
					if(solution_enum == SOLUTION) {
						run_times = s.run_solution();
					}else if(solution_enum == BASELINE) {
						run_times = s.run_baseline();
					}else if(solution_enum == NAIVE) {
						run_times = s.run_naive();
					}else{
						System.err.println("SemanticTest.run() unknown solution enum: "+solution_enum);
					}
					//run_times = s.run_naive();
					//run_times = s.run_baseline();
					//run_times = s.run_incremental_cell_pruning();
					//run_times = s.run_incremental_cell_pruning_deep();
					//run_times = s.run_candidates();
					//run_times = s.run_candidates_deep();
					//run_times = s.run_solution();
					//run_times = s.run_solution_no_candidates();
					//run_times = s.run_bound_tightness_exp();
					run_time += run_times[0];
				}
				run_time /= repitions-1;
				double[] temp = {run_time};
				all_run_times.add(temp);
			}
		}
		
		for(int i=0;i<k_s.length;i++) {
			System.out.print("k="+k_s[i]+"\t");
		}
		System.out.println();
		
		for(int p=0;p<all_run_times.get(0).length;p++) {
			for(int i=0;i<k_s.length;i++) {
				run_times = all_run_times.get(i);
				System.out.print(run_times[p]+"\t");
			}
			System.out.println();
		}
		boolean RESULTS_TO_FILE = true;
		if(RESULTS_TO_FILE) {
			//String result_path = "./results/pan_results_"+System.currentTimeMillis()+".tsv";
		    try {
		        BufferedWriter output = new BufferedWriter(new FileWriter(result_path_bible, true));

		        // Writes the string to the file
		        if(!header_written) {
			        for(int i=0;i<k_s.length;i++) {
			        	output.write("k="+k_s[i]+"\t");
					}
					output.newLine();
					header_written = true;
		        }
				
				for(int p=0;p<all_run_times.get(0).length;p++) {
					for(int i=0;i<k_s.length;i++) {
						run_times = all_run_times.get(i);
						output.write(run_times[p]+"\t");
					}
					output.newLine();
				}
		        

		        // Closes the writer
		        output.close();
		      }catch (Exception e) {
		          e.getStackTrace();
		      }
		}
		all_run_times.clear();
	}
	
	public static void run_pan_experiments() {
		final int[] k_s= {3,4,5,6,7,8};
		final double threshold = 0.7;
		boolean header_written = false;
		
		MAPPING_GRANUALRITY = GRANULARITY_BOOK_TO_BOOK;
		
		ArrayList<Book> books;
		for (int susp_id = 0; susp_id < Importer.PAN11_SRC.length; susp_id++) {
			ArrayList<double[]> all_run_times = new ArrayList<double[]>();
			double[] run_times;
		
			for (int src_id = 0; src_id < Importer.PAN11_SUSP.length; src_id++) {
				books = ImporterAPI.get_pan_11_books(src_id, susp_id);
				for(int k : k_s) {
					Solutions.dense_global_matrix_buffer = null;
					boolean pan_embeddings = true;
					ArrayList<Solutions> solutions = prepare_solution(books,k,threshold, pan_embeddings);
					for(Solutions s : solutions) {
						//run_times = s.run_naive();
						//run_times = s.run_baseline();
						//run_times = s.run_incremental_cell_pruning();
						//run_times = s.run_incremental_cell_pruning_deep();
						//run_times = s.run_candidates();
						//run_times = s.run_candidates_deep();
						run_times = s.run_solution();
						//run_times = s.run_bound_tightness_exp();
						
						all_run_times.add(run_times);
					}
				}
				for(int i=0;i<k_s.length;i++) {
					System.out.print("k="+k_s[i]+"\t");
				}
				System.out.println();
				
				for(int p=0;p<all_run_times.get(0).length;p++) {
					for(int i=0;i<k_s.length;i++) {
						run_times = all_run_times.get(i);
						System.out.print(run_times[p]+"\t");
					}
					System.out.println();
				}
				boolean RESULTS_TO_FILE = true;
				if(RESULTS_TO_FILE) {
					//String result_path = "./results/pan_results_"+System.currentTimeMillis()+".tsv";
				    try {
				        BufferedWriter output = new BufferedWriter(new FileWriter(result_path_pan, true));

				        // Writes the string to the file
				        if(!header_written) {
					        for(int i=0;i<k_s.length;i++) {
					        	output.write("k="+k_s[i]+"\t");
							}
							output.newLine();
							header_written = true;
				        }
						
						for(int p=0;p<all_run_times.get(0).length;p++) {
							for(int i=0;i<k_s.length;i++) {
								run_times = all_run_times.get(i);
								output.write(run_times[p]+"\t");
							}
							output.newLine();
						}
				        

				        // Closes the writer
				        output.close();
				      }catch (Exception e) {
				          e.getStackTrace();
				      }
				}
				all_run_times.clear();
			}
		}
	}
	
	
	public static void main(String[] args) {
		if(args.length==0) {
			String[] temp = {"pc"};//if no experiment specified run the bible experiment 
			args = temp;
		}
		if(contains(args, "b")) {
			run_bible_experiments();
		}else if(contains(args, "p")) {
			run_pan_experiments();
		}else if(contains(args, "pc")) {
			run_pan_correctness_experiments();
		}else{
			System.err.println("main(): No valid experiment specified "+Arrays.toString(args));
		}
		
		/*String file_path = Embedding.get_embedding_path(books.get(0).language);
		ArrayList<MatchesWithEmbeddings> embeddings = MatchesWithEmbeddings.load(file_path);
		
		for(double threshold = 0.0; threshold<1.0;threshold+=0.1) {
			no_match_words(embeddings, threshold);	
		}*/	
	}
	
	public static void run_pan_correctness_experiments() {
		//final int[] k_s= {3,4,5,6,7,8,9,10,11,12,13,14,15,16};
		final int[] k_s= {3,4,5,6,7,8};
		final double threshold = 0.5;//XXX - should set to zero?
				
		ArrayList<Book>[] all_src_plagiats_pairs = pan.Data.load_all_plagiarism_excerpts();
		for(ArrayList<Book> src_plagiat_pair : all_src_plagiats_pairs) {
			
			for(int k : k_s) {
				Solutions.dense_global_matrix_buffer = null;
				boolean pan_embeddings = true;
				ArrayList<Solutions> solutions = prepare_solution(src_plagiat_pair,k,threshold, pan_embeddings);
				for(Solutions s : solutions) {
					s.run_solution();					
					PanResult r = new PanResult(s);//also adds it to the static class variable collecting all results
				}
			}
		}
		PanResult.out();
		PanResult.out_agg();
		
		
		/*for(int k : k_s) {
			PanResult.all_results[k].get(0).out_my_matrices();
			System.out.println();
		}*/
		
		PanResult.clear();
		all_src_plagiats_pairs = pan.Data.load_all_entire_documents();
		for(ArrayList<Book> src_plagiat_pair : all_src_plagiats_pairs) {
			
			for(int k : k_s) {
				Solutions.dense_global_matrix_buffer = null;
				boolean pan_embeddings = true;
				ArrayList<Solutions> solutions = prepare_solution(src_plagiat_pair,k,threshold, pan_embeddings);
				for(Solutions s : solutions) {
					s.run_solution();					
							
					
					//System.out.println("******** k="+k);
					//out(alignment_matrix);
					PanResult r = new PanResult(s);//also adds it to the static class variable collecting all results
					//System.out.println(r.result_header());
					//System.out.println(r);
				}
			}
		}
		//PanResult.out();
		PanResult.out_agg();
		
	}
	
	private static void out(double[] arr) {
		for(double d : arr) {
			System.out.print(d+"\t");
		}
	}

	private static void out(double[][] alignment_matrix) {
		for(double[] arr : alignment_matrix) {
			out(arr);
			System.out.println();
		}
	}

	private static boolean contains(String[] array, String to_match) {
		for(String s : array) {
			if(s.equals(to_match)) {
				return true;
			}
		}
		return false;
	}

	public static void test_node_deletion_in_hungarian() {
		int k = 3;
		HungarianKevinSternAlmpified HKS = new HungarianKevinSternAlmpified(k);
		double[][] cost_matrix = {
				{-0.373873348,	-0.208628434,	-0.178146014}
				,{-0.43714471,	-0.299635582,	-0.206376232}
				,{-0.211175627,	-0.42104582,	-0.139257557}
		};
		HKS.solve(cost_matrix);
		int[] assignement = HKS.get_assignment();
		System.out.println(Arrays.toString(assignement));
		
		/*double[][] cost_matrix_d = {
				{-0.208628434,	-0.178146014}
				,{-0.299635582,	-0.206376232}
				,{-0.42104582,	-0.139257557}
		};*/
		
		double[][] cost_matrix_d = {
				{-0.208628434, -0.299635582,	-0.42104582}
				,{-0.178146014,	-0.206376232, -0.139257557}
		};
		
		HKS.solve(cost_matrix_d);
		assignement = HKS.get_assignment();
		System.out.println(Arrays.toString(assignement));
		
		double[][] next_matrix = {
				{-0.208628434,	-0.178146014,	-1}
				,{-0.299635582,	-0.206376232,	-0.531359443}
				,{-0.42104582,	-0.139257557,	-0.229745722}

		};
		HKS.solve(next_matrix);
		assignement = HKS.get_assignment();
		System.out.println(Arrays.toString(assignement));
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
	
	
	static HashMap<Integer, double[]> embedding_vector_index_buffer = null;
	static ArrayList<HungarianExperiment> prepare_experiment(ArrayList<Book> books, final int k, final double threshold, boolean pan_embeddings) {
		System.out.println("SemanticTest.prepare_experiment() [START]");
		ArrayList<HungarianExperiment> ret = new ArrayList<HungarianExperiment>(books.size());
		
		ArrayList<Book> tokenized_books = Tokenizer.run(books, new BasicTokenizer());
		ArrayList<String> all_tokens_ordered = Sequence.get_ordered_token_list(Sequence.get_unique_tokens(tokenized_books));
		HashMap<String, Integer> token_ids = strings_to_int(all_tokens_ordered);
		
		HashMap<Integer, double[]> embedding_vector_index;
		if(embedding_vector_index_buffer==null) {
			boolean ignore_stopwords = false;
			String file_path = Embedding.get_embedding_path(books.get(0).language,ignore_stopwords, pan_embeddings);
			embedding_vector_index = create_embedding_vector_index(token_ids,all_tokens_ordered,file_path);
		}else{
			embedding_vector_index = embedding_vector_index_buffer;
		}
		
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
	
	static ArrayList<Solutions> prepare_solution(ArrayList<Book> books, final int k, final double threshold, boolean pan_embeddings) {
		MAPPING_GRANUALRITY = GRANULARITY_BOOK_TO_BOOK;
		System.out.println("SemanticTest.prepare_experiment() [START]");
		ArrayList<Solutions> ret = new ArrayList<Solutions>(books.size());
		
		ArrayList<Book> tokenized_books = Tokenizer.run(books, new BasicTokenizer());
		ArrayList<String> all_tokens_ordered = Sequence.get_ordered_token_list(Sequence.get_unique_tokens(tokenized_books));
		HashMap<String, Integer> token_ids = strings_to_int(all_tokens_ordered);
		
		HashMap<Integer, double[]> embedding_vector_index;
		if(embedding_vector_index_buffer==null) {
			boolean ignore_stopwords = false;//XXX
			String file_path = Embedding.get_embedding_path(books.get(0).language,ignore_stopwords, pan_embeddings);
			embedding_vector_index = create_embedding_vector_index(token_ids,all_tokens_ordered,file_path);
			embedding_vector_index_buffer = embedding_vector_index;
		}else{
			embedding_vector_index = embedding_vector_index_buffer;
		}
		
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
				
				Solutions exp = new Solutions(raw_paragraphs_b1, raw_paragraphs_b2, k, threshold, embedding_vector_index);
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
