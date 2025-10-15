package wikipedia;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;

import boss.hungarian.HungarianDeep2;
import boss.hungarian.Solutions;
import boss.test.SemanticTest;
import boss.util.BitSet;
import boss.util.MyArrayList;

public class Corpus {
	//FOR DEBUG
	int debug_token_id 				= 27762;
	int debug_query_article_number 	= 0;
	int debug_other_token_id		= 27762; 
	int debug_art_article_number 	= 398;
	int debug_line  				= 27;
	int debug_column 				= 68; 
	
	
	final int average_length_wikipedia_article = 706;//according to Wikipedia
	/**
	 * Used for leading the data, and tokenizing it
	 */
	WikiDataLoader wl = new WikiDataLoader();
	/**
	 * All the tokens in all documents, i.e., articles after pre-processing
	 */
	ArrayList<ArrayList<String>> raw_articles;
	/**
	 * Performs String -> int mapping
	 */
	HashMap<String, Integer> token_ids;
	/**
	 * Associates a token_id to its embedding vector
	 */
	HashMap<Integer, double[]> embedding_vector_index;
	/**
	 * Determines whether this.candidate_producing_token_pairs is loaded from file or computed from scratch. Note ms vs. mintues. 
	 */
	boolean load_from_file = true;
	/**
	 * Mapping of each token_id to all other token_id, such that sim(token_id, other token_id) >= threshold. I.e., this pair produces candidates
	 */
	MyArrayList[] candidate_producing_token_pairs;//TODO make simple array
	/**
	 * Hashed version of candidate_producing_token_pairs for O(1) access when creating the bitmap index
	 */
	HashSet<Integer>[] candidate_producing_token_pairs_hashed;
	/**
	 * inverted_token_index[token_id] -> L(CorpusArticles) containing this token_id (somewhere)
	 */
	ArrayList<CorpusArticle>[] inverted_token_index;
	/**
	 * Hashed version of
	 * inverted_token_index[token_id] -> L(CorpusArticles) containing this token_id (somewhere)
	 * containing only the article number
	 */
	HashSet<Integer>[] inverted_token_index_hashed;
	/**
	 * The list of all articles. If one concats them, one gets this.tokens.
	 */
	public CorpusArticle[] my_articles;
	/**
	 * The threshold $\theta$ in the paper.
	 */
	final double threshold = 0.7;
	/**
	 * Length of the k-width windows
	 */
	final int k = 10;//XXX
	private static String embedding_path = "wikipedia_corpus_tokens.tsv";
	
	/**
	 * Threshold for minimal article length after pre-processing
	 */
	final int min_length_article = 20;
	
	@SuppressWarnings("unchecked")
	public Corpus() {
		//String line = wl.load_file(WikiDataLoader.test_file);//All articles in one line...
		ArrayList<String> lines = wl.load_corpus(WikiDataLoader.corpus_file);//All articles in one line...
		this.raw_articles = new ArrayList<ArrayList<String>>(lines.size());
		int sum_words = 0;
		//final String regex_characters_to_keep = "[^a-zA-Z0-9 ]";//Note the not at the beginning
		final String regex_characters_to_keep = "[a-zA-Z0-9]+";//Note the not at the beginning
		for(String line : lines){
			ArrayList<String> tokens = wl.tokenize_txt_align(line);//Remove stop words etc.
			ArrayList<String> alpha_numerice_tokens = new ArrayList<String>(tokens.size());
			for(String token : tokens) {
				if(token.matches(regex_characters_to_keep)) {
					alpha_numerice_tokens.add(token);
				}else {
					System.out.println("Deleting "+token);
				}
			}
			if(alpha_numerice_tokens.size()>=min_length_article) {
				raw_articles.add(alpha_numerice_tokens);
				System.out.print("tokens=\t\t\t[");
				for(String token : tokens) {
					System.out.print(token+",");
				}
				System.out.println("]");
				sum_words += alpha_numerice_tokens.size();
			}else {
				System.out.println("To short articel-> deleting this one");
			}
		}
		
		System.out.println("Words after pre-processing= "+sum_words);
		HashSet<String> unique_tokens = new HashSet<String>(sum_words);
		for(ArrayList<String> tokens : raw_articles) {
			for(String s : tokens) {
				unique_tokens.add(s);
				//System.out.println(s);
			}
		}
		System.out.println("Unique words after pre-processing= "+unique_tokens.size());
		ArrayList<String> unique_tokens_sorted = new ArrayList<String>(unique_tokens.size());
		for(String s : unique_tokens) {
			unique_tokens_sorted.add(s);
		}
		System.out.println("unique_tokens_sorted");
		Collections.sort(unique_tokens_sorted);
		//WikiDataLoader.materialize_tokens(unique_tokens_sorted);
		
		token_ids = SemanticTest.strings_to_int(unique_tokens_sorted);//String -> int
		this.embedding_vector_index = SemanticTest.create_embedding_vector_index(token_ids, unique_tokens_sorted, WikiDataLoader.folder+embedding_path );
		
		int max_id = token_ids.size();
		candidate_producing_token_pairs = new MyArrayList[max_id+1]; 
		for(int i=0;i<candidate_producing_token_pairs.length;i++) {
			candidate_producing_token_pairs[i] = new MyArrayList();
		}
		
		//Get the candidate producing token pairs
		double start = System.currentTimeMillis();
		if(load_from_file) {
			Path path = Paths.get(WikiDataLoader.folder+"candidate_producing_token_pairs.txt");
			try {
				List<String> list = Files.readAllLines(path);
				for(int id=0;id<list.size();id++) {
					String[] temp = list.get(id).split(" ");
					if(temp.length>1) {//first entry is id and size
						candidate_producing_token_pairs[id] = new MyArrayList(temp.length-1);
						for(int i=1;i<temp.length;i++) {
							int other_id = Integer.parseInt(temp[i]);
							candidate_producing_token_pairs[id].add(other_id);
						}
						//System.out.println(list.get(id));
						//System.out.println(candidate_producing_token_pairs[id]);
					}
				}
				
			} catch (IOException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}else{
			for(int line_id=0;line_id<=max_id;line_id++) {
				final double[] vec_1 = this.embedding_vector_index.get(line_id);
				candidate_producing_token_pairs[line_id].add(line_id); //this is always a candidate
				for(int col_id=line_id+1;col_id<=max_id;col_id++) {//Exploits symmetry
					final double[] vec_2 = this.embedding_vector_index.get(col_id);
					double sim = Solutions.sim(line_id, col_id, vec_1, vec_2);
					if(sim>threshold) {
						candidate_producing_token_pairs[line_id].add(col_id); 
						candidate_producing_token_pairs[col_id].add(line_id); 
					}
				}
				System.out.print(line_id+"_of_"+max_id);
				System.out.println(candidate_producing_token_pairs[line_id]);
			}
		}
		
		this.candidate_producing_token_pairs_hashed = new HashSet[candidate_producing_token_pairs.length];
		for(int token_id=0;token_id<candidate_producing_token_pairs_hashed.length;token_id++) {
			MyArrayList pairs = candidate_producing_token_pairs[token_id];
			candidate_producing_token_pairs_hashed[token_id] = new HashSet<Integer>(pairs.size());
			for(int i=0;i<pairs.size();i++) {
				candidate_producing_token_pairs_hashed[token_id].add(pairs.ARRAY[i]);
			}
		}
		
		System.out.println("Loaded candidate_producing_token_pairs in "+(System.currentTimeMillis()-start)+" ms");
		int num_articles = raw_articles.size();
		my_articles = new CorpusArticle[num_articles];
		
		System.out.print("Creating and indexing articles");
		start = System.currentTimeMillis();
		for(int i=0;i<num_articles;i++) {
			my_articles[i] = new CorpusArticle(i);
		}
		System.out.println(" in "+(System.currentTimeMillis()-start)+" ms "+num_articles+" articles");
		/*for(Article a : my_articles) {
			System.out.println(a);
		}*/
		
		start = System.currentTimeMillis();
		inverted_token_index = new ArrayList[max_id+1];
		for(int token_id=0;token_id<inverted_token_index.length;token_id++) {
			inverted_token_index[token_id] = new ArrayList<CorpusArticle>();
			for(CorpusArticle art : my_articles) {
				if(art.token_positions.containsKey(token_id)) {
					inverted_token_index[token_id].add(art);
				}
			}
			//System.out.println(token_id+" "+inverted_token_index[token_id].size());
		}
		this.inverted_token_index_hashed = new HashSet[inverted_token_index.length];
		for(int token_id=0;token_id<inverted_token_index.length;token_id++) {
			inverted_token_index_hashed[token_id] = new HashSet<Integer>(inverted_token_index[token_id].size());
			for(CorpusArticle art : inverted_token_index[token_id]) {
				inverted_token_index_hashed[token_id].add(art.article_number);
			}
		}
		
		System.out.println("Created inverted index in "+(System.currentTimeMillis()-start)+" ms");
	}
	
	HashMap<CorpusArticle, MyArrayList> filter(int article_number) {
		return filter(my_articles[article_number]);
	}
	
	/**
	 * Returns for each document being a candidate the lines (i.e., k-width windows) one needs to chek. The candidate lines are grouped into runs.
	 * @param query
	 * @return
	 */
	HashMap<CorpusArticle, MyArrayList> filter(CorpusArticle query) {
		//System.out.println("Articles.filter "+query.article_number);
		double start = System.currentTimeMillis();
		HashMap<CorpusArticle, BitSet> candidate_documents = new HashMap<CorpusArticle, BitSet>(); 
		
		for(int token_id : query.my_tokens) {
			int[] candidate_producing_tokens = this.candidate_producing_token_pairs[token_id].ARRAY;
			//System.out.println(token_id);
			for(int other_token_id : candidate_producing_tokens) {
				ArrayList<CorpusArticle> lalala = inverted_token_index[other_token_id];
				for(CorpusArticle art : lalala) {
					/*if(token_id == debug_token_id && query.article_number == debug_query_article_number && other_token_id==debug_other_token_id && art.article_number == debug_art_article_number) {
						System.err.println(debug_token_id + "Found article in inverted_index");
					}*/
					//ignore yourself
					if(art==query) {//i.e., the pointers are the same
						continue;
					}
					BitSet candidate_lines;
					if(candidate_documents.containsKey(art)) {
						candidate_lines = candidate_documents.get(art);
					}else{
						candidate_lines = new BitSet(art.k_width_windows.length);
						candidate_documents.put(art, candidate_lines);
					}
					int[] positions = art.get_positions(other_token_id);
					for(int pos : positions) {
						candidate_lines.set(pos);
					}
					/*if(token_id == debug_token_id && query.article_number == debug_query_article_number && other_token_id==debug_other_token_id && art.article_number == debug_art_article_number) {
						System.err.println(debug_token_id);
						System.out.println(candidate_lines.get(debug_line)+" in candidate_lines.get(debug_line)");
					}*/
				}
			}
			
		}
		
		HashMap<CorpusArticle, MyArrayList> line_runs = new HashMap<CorpusArticle, MyArrayList>();
		//Condense
		for(Entry<CorpusArticle, BitSet> e : candidate_documents.entrySet()) {
			CorpusArticle art = e.getKey();
			MyArrayList runs = Solutions.condense(e.getValue()); 
			line_runs.put(art, runs);
			
			/*if(art.article_number == debug_art_article_number && query.article_number == debug_query_article_number) {
				System.out.println(e.getValue().get(debug_line));
				System.out.println("Line runs should contain "+debug_line+":"+runs);
			}*/
		}
		double stop = System.currentTimeMillis();
		
		long sum_cells = 0l;
		long sum_lines = 0l;
		long sum_lines_pruned = 0l;
		long sum_cells_pruned = 0l;
		
		for(Entry<CorpusArticle, BitSet> e : candidate_documents.entrySet()) {
			CorpusArticle art = e.getKey();
			BitSet bs = e.getValue();
			
			int size_m = art.k_width_windows.length * query.k_width_windows.length;
			int count_w_pruned = 0;
			for(int w=0;w<art.k_width_windows.length;w++) {
				if(bs.get(w)==false) {
					count_w_pruned++;
				}
			}
			int count_cells_pruned = count_w_pruned * query.k_width_windows.length;
			sum_cells += size_m;
			sum_lines += art.k_width_windows.length;
			sum_lines_pruned += count_w_pruned;
			sum_cells_pruned += count_cells_pruned;
			if(sum_cells<sum_lines_pruned) {
				System.err.println("sum_cells<sum_lines_pruned");
			}
			if(sum_lines<sum_lines_pruned) {
				System.err.println("sum_lines<sum_lines_pruned");
			}
		}
		
		System.out.println("Articles.filter "+query.article_number + " [DONE] in\t"+(stop-start)+"\tms "+line_runs.size()+"\t"+sum_cells+"\t"+sum_lines+"\t"+sum_lines_pruned+"\t"+sum_cells_pruned);
		return line_runs;
	}
	
	HashMap<CorpusArticle, MyArrayList[]> get_candidates_bit_vector(CorpusArticle query, HashMap<CorpusArticle, MyArrayList> result_line_filter){
		double start = System.currentTimeMillis();
		final HashMap<CorpusArticle, MyArrayList[]> all_candidates = new HashMap<CorpusArticle, MyArrayList[]>(result_line_filter.size());
		final BitSet[] candidate_vectors = new BitSet[candidate_producing_token_pairs.length];
		
		//TODO compute all the vectors once
		for(int query_token_id : query.unique_tokens) {
			final MyArrayList l = candidate_producing_token_pairs[query_token_id];
			final int size = l.size();
			
			for(int i=0;i<size;i++) {
				final int other_token_id = l.ARRAY[i];
				if(candidate_vectors[other_token_id]==null) {//Not yet created
					candidate_vectors[other_token_id] = create_bit_vector(other_token_id, query);
				}
				/*if(query.article_number == debug_query_article_number && query_token_id == debug_token_id && other_token_id == debug_other_token_id) {
					System.err.println(debug_token_id+" "+Arrays.toString(candidate_vectors[other_token_id].words));
					System.err.println(debug_token_id+"("+debug_column+")->"+candidate_vectors[other_token_id].get(debug_column));//TODO
				}*/
			}
		}
		//System.out.println("get_candidates_bit_vector() create_bit_vector() in "+(System.currentTimeMillis()-start)+" ");
		
		for(Entry<CorpusArticle, MyArrayList> e : result_line_filter.entrySet()) {
			MyArrayList[] article_result = get_candidates_bit_vector(query, e.getKey(), e.getValue(), candidate_vectors);
			all_candidates.put(e.getKey(), article_result);
		}
		double stop = System.currentTimeMillis();
		
		long sum_num_cells = 0;
		long cells_remaining = 0;
		
		for(Entry<CorpusArticle, MyArrayList[]> e : all_candidates.entrySet()) {
			CorpusArticle art = e.getKey();
			MyArrayList[] article_result = e.getValue();
			
			sum_num_cells+= art.k_width_windows.length * query.k_width_windows.length;
			for(MyArrayList l : article_result) {
				if(l!=null) {
					int size = l.size();
					int[] raw_runs = l.ARRAY;
					for(int c=0;c<size;c+=2) {//Contains start and stop index. Thus, c+=2.
						final int run_start = raw_runs[c];
						final int run_stop  = raw_runs[c+1];
						
						cells_remaining += run_stop-run_start;
					}
				}
			}
		}
		
		System.out.println("get_candidates_bit_vector("+query.article_number+") [DONE] in "+(stop-start)+"\t"+sum_num_cells+"\t"+cells_remaining);
		return all_candidates;
	}
	
	/**
	 * This is the wrapper to validate runs.
	 * @param query -> x, i.e., columns
	 * @param corpus_document -> y, i.e., lines
	 * @param line_runs - (from,to) inclusive, i.e., stop<to+1
	 * @param candidate_vectors - one vector per token, iff null it has no candidates
	 * @return 
	 */
	MyArrayList[] get_candidates_bit_vector(CorpusArticle query, CorpusArticle corpus_document, MyArrayList line_runs, BitSet[] candidate_vectors) {
		final int size 		= line_runs.size();
		final int[] raw_runs= line_runs.ARRAY;
		
		final ArrayList<BitSet> window_bit_vectors = new ArrayList<BitSet>(k);
		final BitSet candidates = new BitSet(query.k_width_windows.length);
		final MyArrayList[] all_candidates = new MyArrayList[corpus_document.k_width_windows.length];//Will be sparse TODO optimize?
		
		for(int c=0;c<size;c+=2) {//Contains start and stop index of lines having candidates. Thus, c+=2.
			final int run_start = raw_runs[c];
			final int run_stop  = raw_runs[c+1];
			
			//We need the BitVector for any unique token in the k-width windows of the corpus_document
			for(int line=run_start; line<=run_stop;line++) {//refers to a line in the Alignment Matrix
				/*if(line == debug_line && query.article_number == debug_query_article_number && corpus_document.article_number == debug_art_article_number) {
					System.err.println(debug_line);
				}*/
				final int[] my_window = corpus_document.k_width_windows[line];
				//Ensure inverted_window_index is filled correctly
				for(int i=0;i<my_window.length;i++) {
					final int token_id = my_window[i];
					if(candidate_vectors[token_id]!=null) {
						BitSet my_bit_vector = candidate_vectors[token_id];
						window_bit_vectors.add(my_bit_vector);
					}
					/*if(this.inverted_token_index_hashed[token_id].contains(query.article_number)) {//FIXME das sieht falsch aus
						BitSet my_bit_vector = candidate_vectors[token_id];
						if(my_bit_vector==null) {//Not yet created
							System.err.println("Schould never happen");
						}
						window_bit_vectors.add(my_bit_vector);
					}*/	
				}
				candidates.clear();//may contain result from prior line
				candidates.or(window_bit_vectors);
				window_bit_vectors.clear();
				MyArrayList candidates_condensed = Solutions.condense(candidates);
				all_candidates[line] = candidates_condensed;
			}
		}
		return all_candidates;
	}
	
	BitSet create_bit_vector(int token_id_line, CorpusArticle query) {
		final BitSet index_this_token = new BitSet(query.k_width_windows.length);//One bit for each window of the corpus document, i.e., column in A
		HashSet<Integer> my_neighborhood_index = candidate_producing_token_pairs_hashed[token_id_line];
		
		for(int i=0;i<query.my_tokens.length;i++) {//Loop over the tokens, not the windows
			final int token_id_in_cropus_doc = query.my_tokens[i];
			if(my_neighborhood_index.contains(token_id_in_cropus_doc)) {
				final int start = Math.max(0, i-k+1);
				final int stop = Math.min(query.k_width_windows.length-1, i);
				index_this_token.set(start,stop+1);
			}
		}
		
		return index_this_token;
	}

	public class CorpusArticle{
		public final int[] my_tokens;
		final int article_number;
		final int[][] k_width_windows;
		/**
		 * Mapping of token_id to all windows it is in.
		 */
		HashMap<Integer, MyArrayList> token_positions = new HashMap<Integer, MyArrayList>();
		
		final int[] unique_tokens;
		
		public CorpusArticle(int _article_number){
			article_number = _article_number;
			ArrayList<String> string_tokens = raw_articles.get(article_number);
			this.my_tokens = new int[string_tokens.size()]; 
			
			for(int i=0;i<string_tokens.size();i++) {
				String s = string_tokens.get(i);
				int token_id = token_ids.get(s).intValue();
				my_tokens[i] = token_id;
			}
			this.k_width_windows = Solutions.create_windows(my_tokens,k);
			
			//Compute mapping of each token to its windows
			for(int window=0;window<k_width_windows.length;window++) {
				for(int token : k_width_windows[window]) {
					MyArrayList my_token_positions;
					if(token_positions.containsKey(token)) {
						my_token_positions = token_positions.get(token);
					}else{
						my_token_positions = new MyArrayList();
						token_positions.put(token, my_token_positions);
					}
					my_token_positions.add(window);
				}
			}
			HashSet<Integer> temp = new HashSet<Integer>();
			for(int token : my_tokens) {
				temp.add(token);
			}
			this.unique_tokens= new int[temp.size()];
			{
				int i=0;
				for(int token : temp) {
					unique_tokens[i++] = token;
				}	
				Arrays.sort(unique_tokens);
			}
			
		}
		public int[] get_positions(int token_id) {
			return token_positions.get(token_id).getTrimmedArray();//TODO optimize me. Directly iterate over the list.
		}
		public String toString() {
			return article_number+" "+Arrays.toString(my_tokens);
		}
	}
	
	public static int num_articles = 5;
	public static boolean only_filters = false;
	
	public static void main(String[] args) {
		Corpus my_corpus = new Corpus();
		ArrayList<HashMap<CorpusArticle, MyArrayList>> corpus_filter_results = new ArrayList<HashMap<CorpusArticle, MyArrayList>>(my_corpus.my_articles.length);
		
		if(only_filters) {
			num_articles = 50;
		}

		for(int i=0;i<num_articles;i++) {
			HashMap<CorpusArticle, MyArrayList> temp = my_corpus.filter(i);
			corpus_filter_results.add(temp);
		}
		
		ArrayList<HashMap<CorpusArticle, MyArrayList[]>> candidate_runs = new ArrayList<HashMap<CorpusArticle, MyArrayList[]>>(); 
		
		for(int i=0;i<num_articles;i++) {
			HashMap<CorpusArticle, MyArrayList> line_filtered_result = corpus_filter_results.get(i);
			candidate_runs.add(my_corpus.get_candidates_bit_vector(my_corpus.my_articles[i], line_filtered_result));
		}
		
		if(only_filters)
			System.exit(0);//XXX
		
		//For each query document
		//int solution_enum = SemanticTest.SOLUTION; FAST_TEXT CORPUS
		int solution_enum = SemanticTest.SOLUTION;
		ArrayList<Double> run_times = new ArrayList<Double>();
		for(int i=0;i<num_articles;i++) {
			final CorpusArticle query = my_corpus.my_articles[i];
			
			//TODO materialize sim();
			double start = System.currentTimeMillis();
			HashMap<Integer, double[]> sim = new HashMap<Integer, double[]>(query.unique_tokens.length);
			final int num_tokens = my_corpus.candidate_producing_token_pairs.length;
			
			for(int my_token : query.unique_tokens) {
				final double[] sim_line = new double[num_tokens];
				for(int other_token=0;other_token<num_tokens;other_token++) {
					double token_sim = Solutions.sim(my_token, other_token, my_corpus.embedding_vector_index.get(my_token), my_corpus.embedding_vector_index.get(other_token));
					sim_line[other_token] = token_sim;
				}

				sim.put(my_token, sim_line);
			}
			double stop = System.currentTimeMillis();
			System.out.println("Materialized sim() in "+(stop-start)+" ms");
			
			HashMap<CorpusArticle, MyArrayList[]> corpus_candidates_all_docs = candidate_runs.get(i);
			for(Entry<CorpusArticle, MyArrayList[]> e : corpus_candidates_all_docs.entrySet()) {
				WikiCorpusSolution w = new WikiCorpusSolution(sim, query, e.getKey(), e.getValue(), my_corpus.embedding_vector_index, my_corpus.k, my_corpus.threshold, solution_enum);
				run_times.add(w.my_run_times[0]);
			}
		}
		double sum = 0.0d;
		for(double d : run_times) {
			sum+=d;
		}
		double avg = sum / (double)run_times.size();
		System.out.println("solution_enum="+solution_enum+" average run time =\t"+avg);
	}
}
