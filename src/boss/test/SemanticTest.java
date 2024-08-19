package boss.test;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;

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
import boss.util.Config;
import boss.util.HistogramDouble;
import boss.util.Pair;
import boss.util.Util;
import pan.Jaccard;
import pan.MatrixLoader;
import pan.PanMetrics;
import pan.PanResult;
import plus.data.Book;
import plus.data.Chapter;
import plus.data.Paragraph;
import txtalign.TxtAlign;
import wikipedia.WikiDataLoader;

public class SemanticTest {	
	static int MAPPING_GRANUALRITY;
	static final int GRANULARITY_PARAGRAPH_TO_PARAGRAPH = 0;
	static final int GRANULARITY_PARAGRAPH_TO_CHAPTER   = 1;
	static final int GRANULARITY_CHAPTER_TO_CHAPTER     = 2;
	static final int GRANULARITY_BOOK_TO_BOOK     		= 3;
	
	static String result_path_pan = "./results/pan_results_"+System.currentTimeMillis()+".tsv";
	static String result_path_bible = "./results/bible_results_"+System.currentTimeMillis()+".tsv";
	
	static void run_bible_experiments(int solution_enum, int[] k_s, double threshold, boolean only_english) {
		MAPPING_GRANUALRITY = GRANULARITY_BOOK_TO_BOOK;
		//MAPPING_GRANUALRITY = GRANULARITY_CHAPTER_TO_CHAPTER;
		result_path_bible+="_"+solution_enum;
		
		Solutions.dense_global_matrix_buffer = null;
		
		int num_repititions = 1;
		//English corpus
		ArrayList<Book> books = ImporterAPI.get_all_english_books();
		run(books, threshold, num_repititions, k_s, solution_enum);
		
		if(only_english) return;
		
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
	
	public static final int SOLUTION = 0;
	public static final int BASELINE = 1;
	public static final int NAIVE    = 2;
	static final int BOUND_TIGHTNESS    = 3;
	static final int MEMORY_CONSUMPTION    = 4;
	
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
					}else if(solution_enum == BOUND_TIGHTNESS) {
						run_times = s.run_bound_tightness_exp();
					}else if(solution_enum == MEMORY_CONSUMPTION) {
						run_times = s.run_memory_consumption_measurement();
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
	
	static void run_pan_experiments() {
		final int[] k_s= {3,4,5,6,7,8,9,10,11,12,13,14,15};
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
			String[] temp = {"p"};//if no experiment specified run the bible experiment 
			args = temp;
		}
		if(contains(args, "b")) {//Bible response time
			final int[] k_s= {3,4,5,6,7,8};
			final double threshold = 0.7;
			int solution_enum = SOLUTION; //SOLUTION, BASELINE, NAIVE
			run_bible_experiments(solution_enum, k_s, threshold, false);
		}else if(contains(args, "p")) {//pan response time
			run_pan_experiments();
		}else if(contains(args, "pc")) {//pan correctness SeDA time old
			run_pan_correctness_experiments();
		}else if(contains(args, "j")) {//jaccard
			run_pan_jaccard_experiments();
		}else if(contains(args, "bound")) {
			final int[] k_s= {3,4,5,6,7,8};
			double[] thetas = {0.5,0.6,0.7,0.8,0.9};
			for(double threshold : thetas) {
				run_bible_experiments(SOLUTION,k_s, threshold, true);	
			}
		}else if(contains(args, "m")) {
			final int[] k_s= {3};//The same for all k's
			double[] thetas = {0.5,0.6,0.7,0.8,0.9};
			int solution_enum = MEMORY_CONSUMPTION; //SOLUTION, BASELINE, NAIVE, MEMORY_CONSUMPTION
			for(double threshold : thetas) {
				run_bible_experiments(solution_enum,k_s, threshold, false);	
			}
			System.out.println("theta\t"+"Size Matrix A\t"+"Size GSM\t"+"Size B\t"+" Size N\t"+" Size All candidate runs");
			for(String[] line : Solutions.memory_consumptions) {
				System.out.println(to_tsv(line));
			}
		}else if(contains(args, "exmpl")){
			//Print example from the paper
			print_example();
		}else if(contains(args, "eval_jacc")){
			//Print results to console and write aggregated results to file: Jaccard
			//MatrixLoader.run_eval_jaccard();
			Config.REMOVE_STOP_WORDS = true;
			PanMetrics.run_jaccard();
		}else if(contains(args, "eval_seda")){
			//Print results to console and write aggregated results to file: SeDA
			//MatrixLoader.run_eval_seda();
			Config.REMOVE_STOP_WORDS = false;
			PanMetrics.run_seda();
		}else if(contains(args, "pjp")){//print jaccard paragraphs
			//Print results to console and write aggregated results to file: Jaccard
			print_jaccard_texts();
		}else if(contains(args, "psp")){//print seda paragraphs
			//Print results to console and write aggregated results to file: SeDA
			print_seda_texts();
		}else if(contains(args, "stats")) {
			statistics();
		}else if(contains(args, "wiki")) {
			WikiDataLoader.run();
		}else{
			System.err.println("main(): No valid experiment specified "+Arrays.toString(args));
		}
		
		/*String file_path = Embedding.get_embedding_path(books.get(0).language);
		ArrayList<MatchesWithEmbeddings> embeddings = MatchesWithEmbeddings.load(file_path);
		
		for(double threshold = 0.0; threshold<1.0;threshold+=0.1) {
			no_match_words(embeddings, threshold);	
		}*/	
	}

	private static String to_tsv(String[] arr) {
		String ret = arr[0];
		for(int i=1;i<arr.length;i++) {
			ret+="\t"+arr[i];
		}
		return ret;
	}

	private static void print_example() {
		
		String[] mets_info_esv = {"ESV","1:2","Ester"};
		String[] chapter_name = {"--1"};
		String[] verse_esv = {"2","That in those days, when the king Ahasuerus sat on the throne of his kingdom, which was in Shushan the palace,"};
		String[] mets_info_kj = {"King James","1:2","Ester"};
		String[] verse_kj = {"2","in those days when King Ahasuerus sat on his royal throne in Susa, the citadel,"};
		
		ArrayList<String[]> raw_book = new ArrayList<String[]>(2);
		raw_book.add(mets_info_esv);
		raw_book.add(chapter_name);
		raw_book.add(verse_esv);
		Book esv = Importer.to_book(raw_book, Book.LANGUAGE_ENGLISH);
		
		raw_book.clear();
		raw_book.add(mets_info_kj);
		raw_book.add(chapter_name);
		raw_book.add(verse_kj);
		Book kj = Importer.to_book(raw_book, Book.LANGUAGE_ENGLISH);
		
		System.out.println(esv);
		System.out.println(kj);
		
		ArrayList<Book> books = new ArrayList<Book>(2); 
		ArrayList<Book> tokenized_books;
		
		books.add(esv);
		books.add(kj);
		
		tokenized_books = Tokenizer.run(books, new BasicTokenizer()); 
		for(Book b : tokenized_books) {
			System.out.println(b);
		}
		int k=3;
		double threshold = 0.5;
		ArrayList<Solutions> solutions = prepare_solution(books,k,threshold, false);
		for(Solutions s : solutions) {
			System.out.println("Tokenized seqeuences");
			Solutions.out(s.raw_paragraph_b1);
			Solutions.out(s.raw_paragraph_b2);
			s.run_baseline();
			System.out.println("Semantic alignment matrix");
			out(s.alignement_matrix);
			System.out.println("Jaccard alignment matrix");
			out(s.jaccard_windows());
			System.out.println("GSM");
			out(s.fill_similarity_matrix_deep());
			System.out.println("SIM()");
			out(Solutions.dense_global_matrix_buffer);
		}
	}

	static void run_pan_jaccard_experiments() {
		Config.STEM_WORDS = true;
		Config.REMOVE_NUMBERS = false;
		System.out.println("run_pan_jaccard_experiments()");
		//final int[] k_s = { 3, 4, 5, 6, 7, 8 };// TODO make global s.t. we can use it everywhere
		final double threshold = 0.5;// XXX - should set to zero?
		
		//TODO create result dir if not exists
		File  f = new File("./results/jaccard_results"); 
		if(!f.exists()) {
			f.mkdir();
		}
		
		ArrayList<Book>[] all_document_pairs = pan.Data.load_all_entire_documents();
		HashMap<String, Integer> token_ids = w2i_pan(all_document_pairs);
		
		ArrayList<Book>[] all_src_plagiats_pairs;
		
		System.out.println("****************************************************************");
		System.out.println("******************************* Excerpt Documents****************");
		System.out.println("****************************************************************");
		
		all_src_plagiats_pairs = pan.Data.load_all_plagiarism_excerpts();
		for(ArrayList<Book> src_plagiat_pair : all_src_plagiats_pairs) {
			Jaccard j = new Jaccard(src_plagiat_pair, token_ids);
			
			for (int k : Config.k_s) {
				double[][] matrix = j.jaccard_windows(k);
				String name = "./results/jaccard_results/" + src_plagiat_pair.get(0).book_name + "_"
						+ src_plagiat_pair.get(1).book_name;
				matrix_to_file(name, k, matrix);
			}
		}
		boolean quit = false;
		if(quit) return;//XXX
		
		System.out.println("****************************************************************");
		System.out.println("******************************* Entire Documents****************");
		System.out.println("****************************************************************");
		
		for (ArrayList<Book> src_plagiat_pair : all_document_pairs) {
			Jaccard j = new Jaccard(src_plagiat_pair, token_ids);
			
			for (int k : Config.k_s) {
				double[][] matrix = j.jaccard_windows(k);
				String name = "./results/jaccard_results/" + src_plagiat_pair.get(0).book_name + "_"
						+ src_plagiat_pair.get(1).book_name;
				matrix_to_file(name, k, matrix);
			}
		}
		Config.STEM_WORDS = false;
		Config.REMOVE_NUMBERS = true;
	}
	
	public static void print_seda_texts() {
		System.out.println("print_seda_texts()");
		MAPPING_GRANUALRITY = GRANULARITY_BOOK_TO_BOOK;
		
		//TODO create result dir if not exists
		File  f = new File("./results/seda_paragraphs"); 
		if(!f.exists()) {
			f.mkdir();
		}
		
		ArrayList<Book>[] all_document_pairs = pan.Data.load_all_entire_documents();
		
		HashMap<String, Integer> token_ids = w2i_seda(all_document_pairs);
		ArrayList<int[]> susp_excerpts_encoded = new ArrayList<int[]>();
		ArrayList<int[]> src_excerpts_encoded = new ArrayList<int[]>();
		ArrayList<Book>[] plagiarism_pairs = pan.Data.load_all_plagiarism_excerpts();
		for (ArrayList<Book> src_plagiat_pair : plagiarism_pairs) {
			String name = "./results/seda_paragraphs/" + src_plagiat_pair.get(0).book_name + "_"
					+ src_plagiat_pair.get(1).book_name;
			System.out.println(name);	
			ArrayList<Book> tokenized_books = Tokenizer.run(src_plagiat_pair, new BasicTokenizer());
			ArrayList<String[]> raw_book_1 = new ArrayList<String[]>(100); 
			ArrayList<String[]> raw_book_2 = new ArrayList<String[]>(100); 
			
			get_tokens(tokenized_books.get(0), tokenized_books.get(1), raw_book_1, raw_book_2);
			ArrayList<int[]> raw_paragraphs_b1  = encode(raw_book_1, token_ids);
			ArrayList<int[]> raw_paragraphs_b2  = encode(raw_book_2, token_ids);
			
			out(raw_book_1.get(0), raw_paragraphs_b1.get(0));
			out(raw_book_2.get(0), raw_paragraphs_b2.get(0));
			
			susp_excerpts_encoded.add(raw_paragraphs_b1.get(0));
			src_excerpts_encoded.add(raw_paragraphs_b2.get(0));
		}

		System.out.println("****************************************************************");
		System.out.println("******************************* Entire Documents****************");
		System.out.println("****************************************************************");

		ArrayList<int[]> susp_encoded= new ArrayList<int[]>();
		ArrayList<int[]> src_encoded = new ArrayList<int[]>();
		for (int i=0;i<all_document_pairs.length;i++) {
			ArrayList<Book> src_plagiat_pair = all_document_pairs[i];
			String name = "./results/seda_paragraphs/" + src_plagiat_pair.get(0).book_name + "_"
					+ src_plagiat_pair.get(1).book_name;
			System.out.println(name);
			ArrayList<Book> tokenized_books = Tokenizer.run(src_plagiat_pair, new BasicTokenizer());
			ArrayList<String[]> raw_book_1 = new ArrayList<String[]>(100); 
			ArrayList<String[]> raw_book_2 = new ArrayList<String[]>(100); 
			
			get_tokens(tokenized_books.get(0), tokenized_books.get(1), raw_book_1, raw_book_2);
			ArrayList<int[]> raw_paragraphs_b1  = encode(raw_book_1, token_ids);
			ArrayList<int[]> raw_paragraphs_b2  = encode(raw_book_2, token_ids);
			
			out(raw_book_1.get(0), raw_paragraphs_b1.get(0));
			out(raw_book_2.get(0), raw_paragraphs_b2.get(0));
			
			susp_encoded.add(raw_paragraphs_b1.get(0));
			src_encoded.add(raw_paragraphs_b2.get(0));
		}
		
		System.out.println("****************************************************************");
		System.out.println("******************************* Ground Truths ******************");
		System.out.println("****************************************************************");
		Jaccard.materialized_ground_truth = new HashMap<String, int[]>();
		for(int i=0;i<all_document_pairs.length;i++) {
			ArrayList<Book> books = all_document_pairs[i];
			String name = "./results/seda_paragraphs/" + books.get(0).book_name + "_"
					+ books.get(1).book_name;
			int[] ground_truth = Jaccard.get_ground_truth(susp_encoded.get(i), src_encoded.get(i), susp_excerpts_encoded.get(i), src_excerpts_encoded.get(i));
			System.out.println(name+"\t"+Util.outTSV(ground_truth));
			Jaccard.materialized_ground_truth.put( books.get(0).book_name + "_"+  books.get(1).book_name, ground_truth);
		}
	}
	
	public static void print_jaccard_texts() {
		Config.STEM_WORDS = true;
		Config.REMOVE_NUMBERS = false;
		Jaccard.verbose_level = Jaccard.print_final_token;
		System.out.println("print_jaccard_texts()");

		
		//TODO create result dir if not exists
		File  f = new File("./results/jaccard_paragraphs"); 
		if(!f.exists()) {
			f.mkdir();
		}
		
		ArrayList<Book>[] all_document_pairs = pan.Data.load_all_entire_documents();
		
		HashMap<String, Integer> token_ids = w2i_pan(all_document_pairs);
		
		ArrayList<Book>[] plagiarism_pairs = pan.Data.load_all_plagiarism_excerpts();
		ArrayList<Jaccard> plagiarism_pairs_jaccard = new ArrayList<Jaccard>(plagiarism_pairs.length);
		for (ArrayList<Book> src_plagiat_pair : plagiarism_pairs) {
			String name = "./results/jaccard_paragraphs/" + src_plagiat_pair.get(0).book_name + "_"
					+ src_plagiat_pair.get(1).book_name;
			System.out.println(name);	
			plagiarism_pairs_jaccard.add(new Jaccard(src_plagiat_pair.get(0), src_plagiat_pair.get(1), token_ids));
		}

		boolean quit = false;
		if (quit) {
			Config.STEM_WORDS = false;
			return;// XXX
		}

		System.out.println("****************************************************************");
		System.out.println("******************************* Entire Documents****************");
		System.out.println("****************************************************************");

		ArrayList<Jaccard> docs = new ArrayList<Jaccard>(plagiarism_pairs.length);
		for (int i=0;i<all_document_pairs.length;i++) {
			ArrayList<Book> src_plagiat_pair = all_document_pairs[i];
			String name = "./results/jaccard_paragraphs/" + src_plagiat_pair.get(0).book_name + "_"
					+ src_plagiat_pair.get(1).book_name;
			System.out.println(name);
			Jaccard j = new Jaccard(src_plagiat_pair.get(0), src_plagiat_pair.get(1), token_ids);
			j.get_ground_truth(plagiarism_pairs_jaccard.get(i));
			docs.add(j);
		}
		
		System.out.println("****************************************************************");
		System.out.println("******************************* Ground Truths ******************");
		System.out.println("****************************************************************");
		Jaccard.materialized_ground_truth = new HashMap<String, int[]>();
		for(int i=0;i<docs.size();i++) {
			Jaccard j = docs.get(i);
			String name = "./results/jaccard_paragraphs/" + j.b_1.book_name + "_"
					+ j.b_2.book_name;
			System.out.println(name+"\t"+Util.outTSV(j.ground_truth));
			Jaccard.materialized_ground_truth.put(j.b_1.book_name + "_"+ j.b_2.book_name, j.ground_truth);
		}
		
		Config.STEM_WORDS = false;
		Config.REMOVE_NUMBERS = true;
	}
	
	private static void inc(HashMap<Double, Double> map, double key, double value) {
		Double old_value = map.get(key);
		if(old_value==null) {
			map.put(key, value);
		}else{
			map.put(key, value+old_value.doubleValue());
		}
	}

	static void run_pan_correctness_experiments() {
		Config.REMOVE_STOP_WORDS = false;
		//final int[] k_s= {3,4,5,6,7,8};
		final double threshold = 0.0;//XXX - should set to zero?
				
		File  f = new File("./results/pan_results"); 
		if(!f.exists()) {
			f.mkdir();
		}
		
		ArrayList<Book>[] all_src_plagiats_pairs;
		
		all_src_plagiats_pairs = pan.Data.load_all_plagiarism_excerpts();
		for(ArrayList<Book> src_plagiat_pair : all_src_plagiats_pairs) {
			Solutions.dense_global_matrix_buffer = null;
			SemanticTest.embedding_vector_index_buffer = null;
			for(int k : Config.k_s) {
				
				boolean pan_embeddings = true;
				ArrayList<Solutions> solutions = prepare_solution(src_plagiat_pair,k,threshold, pan_embeddings);
				for(Solutions s : solutions) {
					s.run_naive();//Threshold is 0. So, we should the naive version.
					String name = "./results/pan_results/"+src_plagiat_pair.get(0).book_name+"_"+src_plagiat_pair.get(1).book_name;
					matrix_to_file(name, k, s.alignement_matrix);
				}
			}
		}
		boolean quit = false;
		if(quit) return;//XXX
		
		System.out.println("****************************************************************");
		System.out.println("******************************* Entire Documents****************");
		System.out.println("****************************************************************");

		all_src_plagiats_pairs = pan.Data.load_all_entire_documents();
		for(ArrayList<Book> src_plagiat_pair : all_src_plagiats_pairs) {
			Solutions.dense_global_matrix_buffer = null;
			SemanticTest.embedding_vector_index_buffer = null;
			for(int k : Config.k_s) {
				
				boolean pan_embeddings = true;
				ArrayList<Solutions> solutions = prepare_solution(src_plagiat_pair,k,threshold, pan_embeddings);
				for(Solutions s : solutions) {
					s.run_naive();					
					String name = "./results/pan_results/"+src_plagiat_pair.get(0).book_name+"_"+src_plagiat_pair.get(1).book_name;
					matrix_to_file(name, k, s.alignement_matrix);
				}
			}
		}
		Config.REMOVE_STOP_WORDS = true;
	}
	
	public static String outTSV(double[] array) {
		if (array == null)
			return "Null array";

		StringBuffer buffer = new StringBuffer(array.length*5);
		for (int index = 0; index < array.length - 1; index++) {
			buffer.append(array[index] + "\t ");
		}
		buffer.append(array[array.length - 1] + "\t");

		return buffer.toString();
	}
	
	private static void matrix_to_file(String directory_path, int k, double[][] alignement_matrix) {
		//double start = System.currentTimeMillis();
		File directory = new File(String.valueOf(directory_path));

		if (!directory.exists()) {
			new File("./results/pan_results/").mkdir();
			directory.mkdir();
		}
		/*String file_path = directory_path+"/"+k+".tsv";
	    try {
	        BufferedWriter output = new BufferedWriter(new FileWriter(file_path, false));
	        for(double[] line : alignement_matrix) {
	        	String line_as_string = outTSV(line);
	        	output.write(line_as_string);
	        	output.newLine();
	        }

	        // Closes the writer
	        output.close();
	      }catch (Exception e) {
	          System.err.println(e);
	      }
	    */
		File f = new File(directory_path+"/"+k+".bin");
	    MatrixLoader.toFile(f, alignement_matrix);
	    //System.out.println("matrix_to_file(String,int,double[][]) DONE in "+(System.currentTimeMillis()-start)+" ms");
	    /*
	    double[][] matrix = MatrixLoader.fromFile(f);
	    if(alignement_matrix.length!=matrix.length) {
	    	System.err.println("alignement_matrix.length!=matrix.length");
	    }
	    if(alignement_matrix[0].length!=matrix[0].length) {
	    	System.err.println("alignement_matrix[0].length!=matrix[0].length");
	    }
	    
	    for(int line = 0;line<alignement_matrix.length;line++) {
	    	for(int column = 0;column<alignement_matrix[0].length;column++) {
	    		if(alignement_matrix[line][column]!=matrix[line][column]) {
	    			System.err.println("alignement_matrix[line][column]!=matrix[line][column] at "+line+" "+column);
	    		}
	    	}
	    }*/
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

	static void test_node_deletion_in_hungarian() {
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
	
	public static HashMap<String, Integer> w2i_pan(){
		return w2i_pan(pan.Data.load_all_entire_documents());
	}
	
	/**
	 * Words to integer for Jaccard
	 * @return
	 */
	static HashMap<String, Integer> w2i_pan(ArrayList<Book>[] all_document_pairs){
		if(Config.USE_TXT_ALIGN_PREPROCESSING) {
			return TxtAlign.read_all_words("./data/all_words.txt");
		}
		HashSet<String> all_tokens = new HashSet<String>(10000);
		
		for (ArrayList<Book> books : all_document_pairs) {
			Jaccard j = new Jaccard(books.get(0), books.get(1));
			
			for(String s : j.text_tokens_b_1) {
				all_tokens.add(s);
			}
			for(String s : j.text_tokens_b_2) {
				all_tokens.add(s);
			}
		}
		
		ArrayList<String> all_tokens_ordered = new ArrayList<String>(all_tokens.size());
		for(String s : all_tokens) {
			all_tokens_ordered.add(s);
		}
		Collections.sort(all_tokens_ordered);
		HashMap<String, Integer> dict = new HashMap<String,Integer>(all_tokens_ordered.size());//TODO
		for(int key=0;key<all_tokens_ordered.size();key++) {
			String s = all_tokens_ordered.get(key);
			dict.put(s, key);
		}
		return dict;
	}
	
	/**
	 * Words to integer for SeDA
	 * Difference is the Tokenizer
	 * @return
	 */
	static HashMap<String, Integer> w2i_seda(ArrayList<Book>[] all_document_pairs){
		HashSet<String> all_tokens = new HashSet<String>(10000);
		
		for (ArrayList<Book> books : all_document_pairs) {
			ArrayList<Book> tokenized_books = Tokenizer.run(books, new BasicTokenizer());
			ArrayList<String> temp = Sequence.get_ordered_token_list(Sequence.get_unique_tokens(tokenized_books));
			
			for(String s : temp) {
				all_tokens.add(s);
			}
		}
		
		ArrayList<String> all_tokens_ordered = new ArrayList<String>(all_tokens.size());
		for(String s : all_tokens) {
			all_tokens_ordered.add(s);
		}
		Collections.sort(all_tokens_ordered);
		HashMap<String, Integer> dict = new HashMap<String,Integer>(all_tokens_ordered.size());//TODO
		for(int key=0;key<all_tokens_ordered.size();key++) {
			String s = all_tokens_ordered.get(key);
			dict.put(s, key);
		}
		return dict;
	}
	
	static void prepare_and_print(ArrayList<Book> books, String file_name, HashMap<String, Integer> token_ids) {
		MAPPING_GRANUALRITY = GRANULARITY_BOOK_TO_BOOK;
		//System.out.println("SemanticTest.prepare_experiment() [START]");
		ArrayList<Book> tokenized_books = Tokenizer.run(books, new BasicTokenizer());
		
		//For each pair of books (i,j)
		for(int i=0;i<tokenized_books.size();i++) {
			Book tokenized_book_1 = tokenized_books.get(i);
			for(int j=i+1;j<tokenized_books.size();j++) {
				Book tokenized_book_2 = tokenized_books.get(j);	
				System.out.println("New book pair "+tokenized_book_1.text_name+" vs. "+tokenized_book_2.text_name);
				
				System.out.println(tokenized_book_1);
				System.out.println(tokenized_book_2);
				
				ArrayList<String[]> raw_book_1 = new ArrayList<String[]>(100); 
				ArrayList<String[]> raw_book_2 = new ArrayList<String[]>(100); 
				
				get_tokens(tokenized_book_1, tokenized_book_2, raw_book_1, raw_book_2);
				
				ArrayList<int[]> raw_paragraphs_b1  = encode(raw_book_1, token_ids);
				ArrayList<int[]> raw_paragraphs_b2  = encode(raw_book_2, token_ids);
				
				out(raw_book_1.get(0), raw_paragraphs_b1.get(0));
				out(raw_book_2.get(0), raw_paragraphs_b2.get(0));
				
				to_file(tokenized_book_1, tokenized_book_2, raw_book_1.get(0), raw_paragraphs_b1.get(0),raw_book_2.get(0), raw_paragraphs_b2.get(0), file_name);
			}
		}
	}
	
	private static void to_file(Book tokenized_book_1, Book tokenized_book_2, String[] raw_book_1, int[] set_ids_1,
			String[] raw_book_2, int[] set_ids_2, String file_name) {
		File f = new File(file_name+".txt");
		try (FileWriter writer = new FileWriter(f);
				BufferedWriter bw = new BufferedWriter(writer)) {

				//bw.write(tokenized_book_1.toString()+"\n");
				//bw.write(tokenized_book_2.toString()+"\n");
				
				String words = "";
				String ids = "";
				String pos ="";
				for(int i=0;i<raw_book_1.length;i++) {
					words += raw_book_1[i]+"\t";
					ids += set_ids_1[i]+"\t";
					pos += i+"\t";
				}
				
				bw.write(words+"\n");
				bw.write(ids+"\n");
				bw.write(pos+"\n");
				
				words = "";
				ids = "";
				pos ="";
				for(int i=0;i<raw_book_2.length;i++) {
					words += raw_book_2[i]+"\t";
					ids += set_ids_2[i]+"\t";
					pos += i+"\t";
				}
				
				bw.write(words+"\n");
				bw.write(ids+"\n");
				bw.write(pos+"\n");

			} catch (IOException e) {
				System.err.format("IOException: %s%n", e);
			}
	}
	
	/**
	 * For Jaccard only
	 * @param books
	 * @param k
	 * @param threshold
	 * @param pan_embeddings
	 * @param token_ids
	 * @return
	 */
	static ArrayList<Solutions> prepare_solution_jaccard(ArrayList<Book> books, final int k, final double threshold, HashMap<String, Integer> token_ids) {
		MAPPING_GRANUALRITY = GRANULARITY_BOOK_TO_BOOK;
		//System.out.println("SemanticTest.prepare_experiment() [START]");
		ArrayList<Solutions> ret = new ArrayList<Solutions>(books.size());
		
		ArrayList<Book> tokenized_books = Tokenizer.run(books, new BasicTokenizer());
		
		//For each pair of books (i,j)
		for(int i=0;i<tokenized_books.size();i++) {
			Book tokenized_book_1 = tokenized_books.get(i);
			for(int j=i+1;j<tokenized_books.size();j++) {
				Book tokenized_book_2 = tokenized_books.get(j);	
				System.out.println("New book pair "+tokenized_book_1.text_name+" vs. "+tokenized_book_2.text_name);
				
				//System.out.println(tokenized_book_1);
				//System.out.println(tokenized_book_2);
				
				ArrayList<String[]> raw_book_1 = new ArrayList<String[]>(100); 
				ArrayList<String[]> raw_book_2 = new ArrayList<String[]>(100); 
				
				get_tokens(tokenized_book_1, tokenized_book_2, raw_book_1, raw_book_2);
				
				ArrayList<int[]> raw_paragraphs_b1  = encode(raw_book_1, token_ids);
				ArrayList<int[]> raw_paragraphs_b2  = encode(raw_book_2, token_ids);
				
				//out(raw_book_1.get(0), raw_paragraphs_b1.get(0));
				//out(raw_book_2.get(0), raw_paragraphs_b2.get(0));
				
				Solutions exp = new Solutions(raw_paragraphs_b1, raw_paragraphs_b2, k, threshold);
				ret.add(exp);
			}
		}
		//System.out.println("SemanticTest.prepare_experiment() [DONE]");
		return ret;
	}

	public static ArrayList<Solutions> prepare_solution(ArrayList<Book> books, final int k, final double threshold, boolean pan_embeddings) {
		MAPPING_GRANUALRITY = GRANULARITY_BOOK_TO_BOOK;
		//System.out.println("SemanticTest.prepare_experiment() [START]");
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
				
				System.out.println(tokenized_book_1);
				System.out.println(tokenized_book_2);
				
				ArrayList<String[]> raw_book_1 = new ArrayList<String[]>(100); 
				ArrayList<String[]> raw_book_2 = new ArrayList<String[]>(100); 
				
				get_tokens(tokenized_book_1, tokenized_book_2, raw_book_1, raw_book_2);
				
				ArrayList<int[]> raw_paragraphs_b1  = encode(raw_book_1, token_ids);
				ArrayList<int[]> raw_paragraphs_b2  = encode(raw_book_2, token_ids);
				
				out(raw_book_1.get(0), raw_paragraphs_b1.get(0));
				out(raw_book_2.get(0), raw_paragraphs_b2.get(0));
				
				Solutions exp = new Solutions(raw_paragraphs_b1, raw_paragraphs_b2, k, threshold, embedding_vector_index);
				ret.add(exp);
			}
		}
		//System.out.println("SemanticTest.prepare_experiment() [DONE]");
		return ret;
	}

	private static void out(String[] raw_book, int[] raw_paragraphs) {
		if(raw_book.length!=raw_paragraphs.length) {
			System.err.println("out(String[] raw_book, int[] raw_paragraphs) raw_book.length!=raw_paragraphs.length");
		}
		String words = "";
		String ids = "";
		for(int i=0;i<raw_book.length;i++) {
			words += raw_book[i]+"\t";
			ids += raw_paragraphs[i]+"\t";
		}
		System.out.println(words);
		System.out.println(ids);
		
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
	
	public static HashMap<Integer, double[]> create_embedding_vector_index(HashMap<String, Integer> token_ids, ArrayList<String> all_tokens_ordered, String file_path) {
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
					if(s!=null && !s.equals("")) {
						temp.add(s);
					}
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

	public static HashMap<String,Integer> strings_to_int(ArrayList<String> all_string_tokens){
		HashMap<String,Integer> encoding = new HashMap<String,Integer>(all_string_tokens.size());
		int id = 0;
		for(String s : all_string_tokens){
			encoding.put(s, id++);
		}
		return encoding;
	}

	static void statistics(String path){
		MatrixLoader.path_to_matrices = path;
		List<String> l = MatrixLoader.get_all_susp_src_directories();
		List<String> ex = MatrixLoader.get_all_excerpt_directories();
		
		HashMap<Integer, ArrayList<Double>> my_matrices_values = new HashMap<Integer, ArrayList<Double>>();
		HashMap<Integer, ArrayList<Double>> my_excerpts_matrices_values = new HashMap<Integer, ArrayList<Double>>();
		
		for(int i=0;i<l.size();i++) {//For each data set pair //XXX l.size()
			//String dir  = l.get(i);
			String name = l.get(i);
			String excerpts = PanMetrics.get_excerpts(name, ex);
			
			HashMap<Integer,double[][]> my_matrices = MatrixLoader.load_all_matrices_of_pair_hashed(name);
			HashMap<Integer,double[][]> my_excerpts_matrices = MatrixLoader.load_all_matrices_of_pair_hashed(excerpts);
			
			for(Entry<Integer, double[][]> e : my_matrices.entrySet()) {//For each window size k
				final int k = e.getKey();
				if(k!=15) {
					continue;
				}
				ArrayList<Double> vals = my_matrices_values.get(k); 
				if(vals==null) {
					vals = new ArrayList<Double>();
					my_matrices_values.put(k, vals);
				}
				ArrayList<Double> ex_vals = my_excerpts_matrices_values.get(k);
				if(ex_vals==null) {
					ex_vals = new ArrayList<Double>();
					my_excerpts_matrices_values.put(k, ex_vals);
				}
				System.out.println(k+" |vals|="+vals.size()+" |ex_vals|="+ex_vals.size()+" ");
				{
					double[][] matrix = e.getValue();
					double[] col_max = new double[matrix[0].length];
					double[] line_max = new double[matrix.length];
					for(int line=0;line<matrix.length;line++) {
						double[] arr = matrix[line];
						for(int col=0;col<matrix[0].length;col++) {
							double d = arr[col];
							if(d>line_max[line]) {
								line_max[line] = d;
							}
							if(d>col_max[col]) {
								col_max[col] = d;
							}
						}
					}
					for(double d : line_max) {
						vals.add(d);
					}
					for(double d : col_max) {
						vals.add(d);
					}
				}
				{
					double[][] e_matrix = my_excerpts_matrices.get(k);
					double[] col_max = new double[e_matrix[0].length];
					double[] line_max = new double[e_matrix.length];
					for(int line=0;line<e_matrix.length;line++) {
						double[] arr = e_matrix[line];
						for(int col=0;col<e_matrix[0].length;col++) {
							double d = arr[col];
							if(d>line_max[line]) {
								line_max[line] = d;
							}
							if(d>col_max[col]) {
								col_max[col] = d;
							}
						}
					}
					for(double d : line_max) {
						ex_vals.add(d);
					}
					for(double d : col_max) {
						ex_vals.add(d);
					}
				}
			}
			System.out.println();
		}
		for(Entry<Integer, ArrayList<Double>> e : my_matrices_values.entrySet()) {//For each window size k
			final int k = e.getKey();
			ArrayList<Double> vals = e.getValue(); 
			ArrayList<Double> ex_vals = my_excerpts_matrices_values.get(k);
			System.out.println();
			System.out.println(k);
			HistogramDouble h_vals = new HistogramDouble(vals);
			HistogramDouble h_ex_vals = new HistogramDouble(ex_vals);
			System.out.println("All values");
			System.out.println(h_vals);
			System.out.println("Values in Excerpts");
			System.out.println(h_ex_vals);
		}
	}
	
	
	static void statistics() {
		//Jaccard
		System.out.println("Jaccard");
		statistics(MatrixLoader.path_to_jaccard_matrices);
		//SeDA
		System.out.println("SeDA");
		statistics(MatrixLoader.path_to_pan_matrices);
	}
}
