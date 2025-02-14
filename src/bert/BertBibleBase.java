package bert;

import java.util.ArrayList;
import java.util.HashMap;

import boss.embedding.Embedding;
import boss.hungarian.Solutions;
import boss.lexicographic.Tokenizer;
import boss.load.ImporterAPI;
import boss.semantic.Sequence;
import boss.test.SemanticTest;
import boss.util.Config;
import boss.util.Util;
import pan.Jaccard;
import plus.data.Book;

public class BertBibleBase {
	static final String path = "./data/de/";
	static SentenceEmbedding query = null;
	static ArrayList<SentenceEmbedding> corpus = null;
	
	static ArrayList<Double> all_scores = new ArrayList<Double>(10000);
	
	static ArrayList<Book> books = null;
	
	static ArrayList<Book> get_bible_books(){
		books = ImporterAPI.get_all_german_books();
		return books;
	}
	
	static void get_corpus(){
		if(BertBibleBase.books==null) {
			get_corpus_and_query(get_bible_books());
		}else{
			get_corpus_and_query(BertBibleBase.books);
		}
	}
	/**
	 * The first book becomes the query, all others are part of the corpus. Note, there is no return value, but we use the static member variables
	 * @param all_books
	 */
	static void get_corpus_and_query(ArrayList<Book> all_books){
		boolean found_query = false;
		ArrayList<SentenceEmbedding> corpus = new ArrayList<SentenceEmbedding>();
		
		for(Book b : all_books) {
			ArrayList<String> sentences = b.to_list();
			SentenceEmbedding se = null;
			if(b.text_name.equals("Lutherbibel 2017")) {
				se = new SentenceEmbedding(path, "luther", sentences);
			}else if(b.text_name.equals("Elberfelder Bibel")){
				se = new SentenceEmbedding(path, "elberfelder", sentences);
			}else if(b.text_name.equals("Neue evangelistische")){
				se = new SentenceEmbedding(path, "ne", sentences);
			}else if(b.text_name.equals("Schlachter 2000")){
				se = new SentenceEmbedding(path, "schlachter", sentences);
			}else if(b.text_name.equals("Volxbibel 2023")){
				se = new SentenceEmbedding(path, "volxbibel", sentences);
			}else{
				System.err.println("???????");
			}
			
			if (!found_query) {
				query = se;
				found_query = true;
			}else {
				corpus.add(se);
			}
		}
	}
	
	public static void materialize_results() {
		for(int k : Config.k_s) {
			//fast_text_experiment(query, corpus, books, k);
			seda_experiment(query, corpus, books, k);
			all_scores.clear();
		}
		for(int k : Config.k_s) {
			fast_text_experiment(query, corpus, books, k);
			all_scores.clear();
		}
		for(int k : Config.k_s) {
			jaccard_experiment(query, corpus, books, k);
			all_scores.clear();
		}
	}
	
	
	public static void main(String[] args) {
		materialize_results();
	}
	
	static BibleResult get_bert_result(SentenceEmbedding query, ArrayList<SentenceEmbedding> corpus) {
		double[][] all_sims = new double[query.sentences.size()][corpus.size()];
		int[][] all_indexes = new int[query.sentences.size()][corpus.size()];
		
		for(int i=0;i<query.sentences.size();i++) {
			String paragraph = query.sentences.get(i);
			double[] vector_query = query.vectors.get(i);
			System.out.println("**Find most similar paragraph to i="+i+" "+paragraph);
			
			for(int i_c=0;i_c<corpus.size();i_c++) {
				SentenceEmbedding se = corpus.get(i_c);
				int index_most_similar = 1;
				double max_similarity = Double.NEGATIVE_INFINITY;
				for(int j=0;j<se.sentences.size();j++) {
					double[] vector_2_cmpr = se.vectors.get(j);
					double sim = Solutions.cosine_similarity(vector_query, vector_2_cmpr);
					all_scores.add(sim);
					if(sim>max_similarity) {
						index_most_similar = j;
						max_similarity = sim;
					}
				}
				System.out.println("Found j="+index_most_similar+" sim="+max_similarity+" "+se.sentences.get(index_most_similar));
				all_sims[i][i_c] = max_similarity;
				all_indexes[i][i_c] = index_most_similar;
			}
			System.out.println();
		}
		System.out.println(Util.outTSV(all_sims));
		System.out.println();
		return new BibleResult(-1, "Bert", all_sims, all_indexes, Util.toPrimitive(all_scores));
	}
	
	static void jaccard_experiment(SentenceEmbedding query, ArrayList<SentenceEmbedding> corpus, ArrayList<Book> books, int k) {
		double[][] all_sims = new double[query.sentences.size()][corpus.size()];
		int[][] all_indexes = new int[query.sentences.size()][corpus.size()];
		
		ArrayList<ArrayList<String>> tokenized_books = Tokenizer.tokenize(books);
		ArrayList<String> all_tokens_ordered = Sequence.get_unique_tokens_orderd(tokenized_books);
		HashMap<String, Integer> token_ids = SemanticTest.strings_to_int(all_tokens_ordered);
		
		/**
		 * [document][paragraph][token_ids]
		 */
		ArrayList<ArrayList<int[]>> blubb = new ArrayList<ArrayList<int[]>>(); 
		
		for(SentenceEmbedding book : corpus) {
			ArrayList<int[]> all_ps = new ArrayList<int[]>();
			for(String p : book.sentences) {
				ArrayList<String> paragraph = Tokenizer.tokenize_bible_de(p);
				ArrayList<int[]> raw_paragraphs_b1  = SemanticTest.encode_(paragraph, token_ids);
				all_ps.add(raw_paragraphs_b1.get(0));//XXX is only one
			}
			blubb.add(all_ps);
		}
		//pre-compute the windows
		ArrayList<ArrayList<int[][]>> all_windows = new ArrayList<ArrayList<int[][]>>(blubb.size());
		for(ArrayList<int[]> my_ps : blubb) {
			ArrayList<int[][]> my_windows = new ArrayList<int[][]>(my_ps.size());
			for(int[] p : my_ps) {
				int[][] paragraph_winodws = Jaccard.create_windows(p, Math.min(k, p.length));
				my_windows.add(paragraph_winodws);
			}
			all_windows.add(my_windows);
		}
				
		for(int i=0;i<query.sentences.size();i++) {
			System.out.println("**Find most similar paragraph to i="+i+" "+query.sentences.get(i)+" "+query.name);
			ArrayList<String> paragraph = Tokenizer.tokenize_bible_de(query.sentences.get(i));
			int[] raw_paragraphs_b1   = SemanticTest.encode_(paragraph, token_ids).get(0);//is only one int[]
			int[][] k_with_windows_b1 = Jaccard.create_windows(raw_paragraphs_b1, Math.min(k, raw_paragraphs_b1.length));//TODO min(k)
			
			for(int i_c=0;i_c<corpus.size();i_c++) {
				final ArrayList<int[]> my_tokenized_paragraphs = blubb.get(i_c);
				double max_similarity = Double.NEGATIVE_INFINITY;
				int index_most_similar = -1;
				
				for(int p_id=0;p_id<my_tokenized_paragraphs.size();p_id++) {
					System.out.print("i="+i+"\t"+corpus.get(i_c).name+" p="+p_id+"\t");
					final int[][] k_with_windows_b2 = all_windows.get(i_c).get(p_id);
					
					double[][] matrix = new double[k_with_windows_b1.length][k_with_windows_b2.length];
					for(int row=0;row<matrix.length;row++) {
						int[] w_r = k_with_windows_b1[row];
						for(int colum=0;colum<matrix[0].length;colum++) {
							int[] w_c = k_with_windows_b2[colum];
							double jaccard_sim = Jaccard.jaccard(w_r, w_c);
							matrix[row][colum] = jaccard_sim;
						}
					}

					double sim_score_paragraphs = reduce(matrix);
					System.out.println(sim_score_paragraphs);
					all_scores.add(sim_score_paragraphs);
					if(sim_score_paragraphs>max_similarity) {
						index_most_similar = p_id;
						max_similarity = sim_score_paragraphs;
					}
					//System.out.println(Util.outTSV(matrix));
					//System.out.println();
				}
				System.out.println("Found j="+index_most_similar+" sim="+max_similarity+" "+corpus.get(i_c).sentences.get(index_most_similar));
				all_sims[i][i_c] = max_similarity;
				all_indexes[i][i_c] = index_most_similar;
			}
		}
		System.out.println(Util.outTSV(all_sims));
		System.out.println();
		new BibleResult(k, "jaccard", all_sims, all_indexes, Util.toPrimitive(all_scores)).to_file();
	}
	
	static void fast_text_experiment(SentenceEmbedding query, ArrayList<SentenceEmbedding> corpus, ArrayList<Book> books, final int k) {
		double[][] all_sims = new double[query.sentences.size()][corpus.size()];
		int[][] all_indexes = new int[query.sentences.size()][corpus.size()];
		Config.verbose = false;
		
		ArrayList<ArrayList<String>> tokenized_books = Tokenizer.tokenize(books);
		ArrayList<String> all_tokens_ordered = Sequence.get_unique_tokens_orderd(tokenized_books);
		HashMap<String, Integer> token_ids = SemanticTest.strings_to_int(all_tokens_ordered);
		
		boolean ignore_stopwords = false;//XXX
		String file_path = Embedding.get_embedding_path(books.get(0).language,ignore_stopwords, false);
		HashMap<Integer, double[]> embedding_vector_index = SemanticTest.create_embedding_vector_index(token_ids,all_tokens_ordered,file_path);
		
		double threshold = 0;
		
		/**
		 * [document][paragraph][token_ids]
		 */
		ArrayList<ArrayList<int[]>> blubb = new ArrayList<ArrayList<int[]>>(); 
		
		for(SentenceEmbedding book : corpus) {
			ArrayList<int[]> all_ps = new ArrayList<int[]>();
			for(String p : book.sentences) {
				ArrayList<String> paragraph = Tokenizer.tokenize_bible_de(p);
				ArrayList<int[]> raw_paragraphs_b1  = SemanticTest.encode_(paragraph, token_ids);
				all_ps.add(raw_paragraphs_b1.get(0));//XXX is only one
			}
			blubb.add(all_ps);
		}
		//pre-compute sim() once
		Solutions.dense_global_matrix_buffer = create_dense_matrix(all_tokens_ordered.size(),embedding_vector_index);
		
		for(int i=0;i<query.sentences.size();i++) {
			ArrayList<String> paragraph = Tokenizer.tokenize_bible_de(query.sentences.get(i));
			System.out.println("**k="+k+" i="+i+" "+query.sentences.get(i)+" "+query.name);
			ArrayList<int[]> raw_paragraphs_b1  = SemanticTest.encode_(paragraph, token_ids);
			
			for(int i_c=0;i_c<corpus.size();i_c++) {
				final ArrayList<int[]> my_tokenized_paragraphs = blubb.get(i_c);
				double max_similarity = Double.NEGATIVE_INFINITY;
				int index_most_similar = -1;
				
				for(int p_id=0;p_id<my_tokenized_paragraphs.size();p_id++) {
					if(Config.verbose) System.out.print("i="+i+" "+corpus.get(i_c).name+" p="+p_id+"\t");
					ArrayList<int[]> raw_paragraphs_b2  = new ArrayList<int[]>();//dummy container
					raw_paragraphs_b2.add(my_tokenized_paragraphs.get(p_id));
					int my_k = Math.min(raw_paragraphs_b1.get(0).length, raw_paragraphs_b2.get(0).length);
					my_k = Math.min(my_k, k);
					Solutions s = new Solutions(raw_paragraphs_b1, raw_paragraphs_b2, my_k, threshold, embedding_vector_index);
					s.run_fast_text();
					double[][] matrix = s.alignement_matrix;
					double sim_score_paragraphs = reduce(matrix);
					all_scores.add(sim_score_paragraphs);
					if(sim_score_paragraphs>max_similarity) {
						index_most_similar = p_id;
						max_similarity = sim_score_paragraphs;
					}
					//System.out.println(Util.outTSV(matrix));
					//System.out.println();
				}
				System.out.println("Found j="+index_most_similar+" sim="+max_similarity+" "+corpus.get(i_c).sentences.get(index_most_similar));
				all_sims[i][i_c] = max_similarity;
				all_indexes[i][i_c] = index_most_similar;
			}
		}
		System.out.println(Util.outTSV(all_sims));
		System.out.println();
		new BibleResult(k, "fest_text", all_sims, all_indexes, Util.toPrimitive(all_scores)).to_file();
	}
	
	static void seda_experiment(SentenceEmbedding query, ArrayList<SentenceEmbedding> corpus, ArrayList<Book> books, final int k) {
		double[][] all_sims = new double[query.sentences.size()][corpus.size()];
		int[][] all_indexes = new int[query.sentences.size()][corpus.size()];
		Config.verbose = false;
		
		ArrayList<ArrayList<String>> tokenized_books = Tokenizer.tokenize(books);
		ArrayList<String> all_tokens_ordered = Sequence.get_unique_tokens_orderd(tokenized_books);
		HashMap<String, Integer> token_ids = SemanticTest.strings_to_int(all_tokens_ordered);
		
		boolean ignore_stopwords = false;//XXX
		String file_path = Embedding.get_embedding_path(books.get(0).language,ignore_stopwords, false);
		HashMap<Integer, double[]> embedding_vector_index = SemanticTest.create_embedding_vector_index(token_ids,all_tokens_ordered,file_path);
		
		double threshold = 0;
		
		/**
		 * [document][paragraph][token_ids]
		 */
		ArrayList<ArrayList<int[]>> blubb = new ArrayList<ArrayList<int[]>>(); 
		
		for(SentenceEmbedding book : corpus) {
			ArrayList<int[]> all_ps = new ArrayList<int[]>();
			for(String p : book.sentences) {
				ArrayList<String> paragraph = Tokenizer.tokenize_bible_de(p);
				ArrayList<int[]> raw_paragraphs_b1  = SemanticTest.encode_(paragraph, token_ids);
				all_ps.add(raw_paragraphs_b1.get(0));//XXX is only one
			}
			blubb.add(all_ps);
		}
		//pre-compute sim() once
		Solutions.dense_global_matrix_buffer = create_dense_matrix(all_tokens_ordered.size(),embedding_vector_index);
		
		for(int i=0;i<query.sentences.size();i++) {
			ArrayList<String> paragraph = Tokenizer.tokenize_bible_de(query.sentences.get(i));
			System.out.println("**k="+k+" i="+i+" "+query.sentences.get(i)+" "+query.name);
			ArrayList<int[]> raw_paragraphs_b1  = SemanticTest.encode_(paragraph, token_ids);
			
			for(int i_c=0;i_c<corpus.size();i_c++) {
				final ArrayList<int[]> my_tokenized_paragraphs = blubb.get(i_c);
				double max_similarity = Double.NEGATIVE_INFINITY;
				int index_most_similar = -1;
				
				for(int p_id=0;p_id<my_tokenized_paragraphs.size();p_id++) {
					if(Config.verbose) System.out.print("i="+i+" "+corpus.get(i_c).name+" p="+p_id+"\t");
					ArrayList<int[]> raw_paragraphs_b2  = new ArrayList<int[]>();//dummy container
					raw_paragraphs_b2.add(my_tokenized_paragraphs.get(p_id));
					int my_k = Math.min(raw_paragraphs_b1.get(0).length, raw_paragraphs_b2.get(0).length);
					my_k = Math.min(my_k, k);
					Solutions s = new Solutions(raw_paragraphs_b1, raw_paragraphs_b2, my_k, threshold, embedding_vector_index);
					s.run_naive();
					double[][] matrix = s.alignement_matrix;
					double sim_score_paragraphs = reduce(matrix);
					all_scores.add(sim_score_paragraphs);
					if(sim_score_paragraphs>max_similarity) {
						index_most_similar = p_id;
						max_similarity = sim_score_paragraphs;
					}
					//System.out.println(Util.outTSV(matrix));
					//System.out.println();
				}
				System.out.println("Found j="+index_most_similar+" sim="+max_similarity+" "+corpus.get(i_c).sentences.get(index_most_similar));
				all_sims[i][i_c] = max_similarity;
				all_indexes[i][i_c] = index_most_similar;
			}
		}
		System.out.println(Util.outTSV(all_sims));
		System.out.println();
		new BibleResult(k, "seda", all_sims, all_indexes, Util.toPrimitive(all_scores)).to_file();
	}
	
	public static double reduce(double[][] matrix) {
		double[] line_max_sim = new double[matrix.length];
		double[] column_max_sim = new double[matrix[0].length];
		for(int line=0;line<matrix.length;line++) {
			for(int column=0;column<matrix[0].length;column++) {
				double sim = matrix[line][column];
				if(line_max_sim[line]<sim) {
					line_max_sim[line] = sim;
				}
				if(column_max_sim[column]<sim) {
					column_max_sim[column]=sim;
				}
			}
		}
		double avg_sim = sum(line_max_sim);
		avg_sim += sum(column_max_sim);
		avg_sim /= (line_max_sim.length+column_max_sim.length);//normalize to [0,1] 
		return avg_sim;
	}
	
	private static double sum(double[] arr){
		double sum = 0;
		for(double d : arr) {
			sum+=d;
		}
		return sum;
	}

	private static final double EQUAL = 1;
	private static double[][] create_dense_matrix(int max_id, final HashMap<Integer, double[]> embedding_vector_index) {
		double start = System.currentTimeMillis();
		double[][] dense_global_matrix_buffer = new double[max_id+1][max_id+1];//This is big....
		for(int line_id=0;line_id<dense_global_matrix_buffer.length;line_id++) {
			dense_global_matrix_buffer[line_id][line_id] = EQUAL;
			final double[] vec_1 = embedding_vector_index.get(line_id);
			for(int col_id=line_id+1;col_id<dense_global_matrix_buffer[0].length;col_id++) {//Exploits symmetry
				final double[] vec_2 = embedding_vector_index.get(col_id);
				double sim = Solutions.sim(line_id, col_id, vec_1, vec_2);
				dense_global_matrix_buffer[line_id][col_id] = sim;
				dense_global_matrix_buffer[col_id][line_id] = sim;
			}
		}
		double stop = System.currentTimeMillis();
		double check_sum = Solutions.sum(dense_global_matrix_buffer);
		int size = dense_global_matrix_buffer.length*dense_global_matrix_buffer[0].length;
		
		System.out.println("create_dense_matrix()\t"+(stop-start)+" check sum=\t"+check_sum+" size="+size);
		return dense_global_matrix_buffer;
	}

	public static BibleResult get_ground_truth() {
		//load Bible vecs
		ArrayList<Book> books = ImporterAPI.get_all_german_books();
		String path = "./data/de/";
		
		SentenceEmbedding query = null;
		ArrayList<SentenceEmbedding> corpus = new ArrayList<SentenceEmbedding>();
		boolean found_query = false;
		
		
		for(Book b : books) {
			ArrayList<String> sentences = b.to_list();
			SentenceEmbedding se = null;
			if(b.text_name.equals("Lutherbibel 2017")) {
				se = new SentenceEmbedding(path, "luther", sentences);
			}else if(b.text_name.equals("Elberfelder Bibel")){
				se = new SentenceEmbedding(path, "elberfelder", sentences);
			}else if(b.text_name.equals("Neue evangelistische")){
				se = new SentenceEmbedding(path, "ne", sentences);
			}else if(b.text_name.equals("Schlachter 2000")){
				se = new SentenceEmbedding(path, "schlachter", sentences);
			}else if(b.text_name.equals("Volxbibel 2023")){
				se = new SentenceEmbedding(path, "volxbibel", sentences);
			}else{
				System.err.println("???????");
			}
			
			if (!found_query) {
				query = se;
				found_query = true;
			}else {
				corpus.add(se);
			}
		}
		
		return get_bert_result(query, corpus);
	}
}
