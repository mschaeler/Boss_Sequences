package oph;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;

import bert.BertBibleBase;
import bert.BibleResult;
import boss.lexicographic.Tokenizer;
import boss.load.ImporterAPI;
import boss.semantic.Sequence;
import boss.test.SemanticTest;
import boss.util.Config;
import boss.util.Util;
import plus.data.Book;
import wikipedia.WikiDataLoader;

public class Experiment {
	static void run_bible_runtime_experiment() {
		ArrayList<Book> en_books = ImporterAPI.get_all_english_books();
		ArrayList<Book> de_books = ImporterAPI.get_all_german_books();
		
		double threshold = 0.4;//the same as always
		int[] k_s = Config.k_s;
		
		ArrayList<double[]> all_results = new ArrayList<double[]>();
		{
			double[] runtime_reuslts = run_bible_runtime_experiment(0,0,en_books.get(0),en_books.get(1), threshold, k_s);
			all_results.add(runtime_reuslts);
		}
		for(int i=0;i<de_books.size();i++) {
			final Book b_1 = de_books.get(i); 
			for(int j=i+1;j<de_books.size();j++) {
				final Book b_2 = de_books.get(j);
				double[] runtime_reuslts = run_bible_runtime_experiment(i,j,b_1,b_2, threshold, k_s);
				all_results.add(runtime_reuslts);
			}
		}
		System.out.println("****************");
		double[] agg_results = new double[k_s.length];
		
		System.out.println("k\t"+Util.outTSV(k_s));
		for(int i=0;i<k_s.length;i++) {
			System.out.println("k="+k_s[i]+"\t"+agg_results[i]);
		}
	}
	
	private static double[] run_bible_runtime_experiment(int i_b_1,int i_b_2, Book b_1, Book b_2, double threshold, int[] k_s) {
		System.out.println("i="+i_b_1+" j="+i_b_2+" "+b_1.text_name+" vs. "+b_2.text_name);
		
		boolean use_stemming = false;
		
		ArrayList<String> b_1_tokens = Tokenizer.tokenize(b_1, use_stemming);
		ArrayList<String> b_2_tokens = Tokenizer.tokenize(b_2, use_stemming);
				
		ArrayList<ArrayList<String>> tokenized_books = new ArrayList<ArrayList<String>>();
		tokenized_books.add(b_1_tokens);
		tokenized_books.add(b_2_tokens);
		
		ArrayList<String> all_tokens_ordered = Sequence.get_unique_tokens_orderd(tokenized_books);
		
		HashMap<String, Integer> token_ids = SemanticTest.strings_to_int(all_tokens_ordered);
		
		int[] src  = SemanticTest.encode_(b_1_tokens, token_ids).get(0);//does the order rake a significant difference?
		int[] query= SemanticTest.encode_(b_2_tokens, token_ids).get(0);
		
		int sketch_size = 32;
		OPH index_src = new OPH(src, sketch_size);
		
		double[] run_time_results = new double[k_s.length];
		
		for(int i=0;i<run_time_results.length;i++) {
			int k = k_s[i];
			System.out.println("k="+k);
			index_src.query(query, threshold, k);
			double run_time = index_src.get_runtime();
			run_time_results[i] = run_time;
		}
		System.out.println(Util.outTSV(k_s));
		System.out.println(Util.outTSV(run_time_results));
		
		return run_time_results;
	}

	static void run_bible_test_experiment() {
		ArrayList<Book> books = ImporterAPI.get_all_english_books();
		System.out.println(books.get(0).text_name);
		System.out.println(books.get(1).text_name);
		
		boolean use_stemming = true;
		
		String esv = books.get(0).to_single_line_string();
		ArrayList<String> esv_tokens = Tokenizer.tokenize(esv, use_stemming);
		String king_james = books.get(1).to_single_line_string();
		ArrayList<String> king_james_tokens = Tokenizer.tokenize(king_james, use_stemming);
		
		ArrayList<ArrayList<String>> tokenized_books = new ArrayList<ArrayList<String>>();
		tokenized_books.add(esv_tokens);
		tokenized_books.add(king_james_tokens);
		
		ArrayList<String> all_tokens_ordered = Sequence.get_unique_tokens_orderd(tokenized_books);
		/*System.out.println("*** Unique tokens Begin");
		for(String s : all_tokens_ordered) {
			System.out.println(s);
		}
		System.out.println("*** Unique tokens End");*/
		HashMap<String, Integer> token_ids = SemanticTest.strings_to_int(all_tokens_ordered);
		
		ArrayList<int[]> raw_paragraphs_b1  = SemanticTest.encode_(esv_tokens, token_ids);
		ArrayList<int[]> raw_paragraphs_b2  = SemanticTest.encode_(king_james_tokens, token_ids);
		
		run(raw_paragraphs_b1.get(0), raw_paragraphs_b2.get(0));
		
	}
	
	static double[][] run_pan_experiment(double threshold, int[] my_k_s) {
		ArrayList<Book>[] all_pairs_excerpt = pan.Data.load_all_plagiarism_excerpts();
		ArrayList<Book>[] all_pairs = pan.Data.load_all_entire_documents();
		int num_pairs = all_pairs.length;
		//num_pairs = 3;//for debug
		
		PanResult[] all_results = new PanResult[num_pairs];
		
		for(int pair=0;pair<num_pairs;pair++) {
			System.out.println("************Pair "+pair);
			ArrayList<Book> src_plagiat_pair = all_pairs[pair];
			ArrayList<Book> excerpt_pair = all_pairs_excerpt[pair];
			
			ArrayList<ArrayList<String>> src_plagiat_pair_tokenized = Tokenizer.tokenize(src_plagiat_pair);
			ArrayList<ArrayList<String>> excerpt_pair_tokenized = Tokenizer.tokenize(excerpt_pair);
			
			ArrayList<String> all_tokens_ordered = Sequence.get_unique_tokens_orderd(src_plagiat_pair_tokenized);
			HashMap<String, Integer> token_ids = SemanticTest.strings_to_int(all_tokens_ordered);
			
			int[] raw_paragraphs_sups  = SemanticTest.encode_(src_plagiat_pair_tokenized.get(0), token_ids).get(0);
			int[] raw_paragraphs_src   = SemanticTest.encode_(src_plagiat_pair_tokenized.get(1), token_ids).get(0);
			
			int[] raw_excerpt_sups  = SemanticTest.encode_(excerpt_pair_tokenized.get(0), token_ids).get(0);
			int[] raw_excerpt_src	= SemanticTest.encode_(excerpt_pair_tokenized.get(1), token_ids).get(0);
			
			//find the excerpt start src
			int offset_src = -1;
			for(int i=0;i<raw_paragraphs_src.length;i++) {
				if(check(raw_paragraphs_src,raw_excerpt_src,i)) {
					offset_src = i;
					break;
				}
			}
			if(offset_src==-1) {
				System.err.println("offset_src==-1");
			}
			//find the excerpt start susp
			int offset_susp = -1;
			for(int i=0;i<raw_paragraphs_sups.length;i++) {
				if(check(raw_paragraphs_sups,raw_excerpt_sups,i)) {
					offset_susp = i;
					break;
				}
			}
			if(offset_susp==-1) {
				System.err.println("offset_susp==-1");
			}
			BitSet gtruth_src = new BitSet(raw_paragraphs_src.length);
			BitSet gtruth_susp = new BitSet(raw_paragraphs_sups.length);
			
			gtruth_src.set (offset_src , offset_src+raw_excerpt_src.length);
			gtruth_susp.set(offset_susp, offset_susp+raw_excerpt_sups.length);
			
			PanResult pr = new PanResult(pair, threshold, gtruth_src,gtruth_susp);
			all_results[pair] = pr;
			/*System.out.println("Offsets src ["+offset_src+","+(offset_src+raw_excerpt_src.length)+"]");
			System.out.println("Offsets susp ["+offset_susp+","+(offset_susp+raw_excerpt_sups.length)+"]");*/
			
			int sketch_size = 32;
			OPH src = new OPH(raw_paragraphs_src, sketch_size);
			
			ArrayList<Double> run_times = new ArrayList<Double>(Config.k_s.length);
			for(int k : my_k_s) {
				src.query(raw_paragraphs_sups, threshold, k);
				run_times.add(src.get_runtime());
				pr.add(k, src.marked_src(), src.marked_sup(), src.get_runtime());
				
			}
			pr.analyze();
			System.out.println(run_times);
			
		}
		for(PanResult pr : all_results) {
			System.out.println(pr);
		}
		
		return PanResult.aggregate(all_results);
	}
	
	private static boolean check(int[] raw_paragraph, int[] raw_excerpt, int offset) {
		for(int i=0;i<raw_excerpt.length;i++) {
			if(raw_paragraph[offset+i]!=raw_excerpt[i]) {
				return false;
			}
		}
		return true;
	}

	static void run(int[] src_document, int[] supicious_doc){
		int sketch_size = 32;
		OPH src = new OPH(src_document, sketch_size);
		src.query(supicious_doc, 0.3);
		ArrayList<Double> run_times = new ArrayList<Double>(Config.k_s.length);
		int[] my_k_s = {3,6,12,24,48,96};
		for(int k : my_k_s) {
			src.query(supicious_doc, 0.3, k);
			run_times.add(src.get_runtime());
		}
		System.out.println(run_times);
//		src.query(supicious_doc, 0.3, 16);
	}
	
	static void run_pan_experiment(){
		double[] thresholds = {0.4,0.41,0.42,0.43,0.43,0.44,0.45};
		//int[] my_k_s = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30};
		int[] my_k_s = Config.k_s;
		
		
		ArrayList<double[][]> all_results = new ArrayList<double[][]>(thresholds.length);
		for(double t : thresholds) {
			all_results.add(run_pan_experiment(t, my_k_s));
		}
		ArrayList<String> precision_macro = new ArrayList<String>();
		precision_macro.add(Util.concat("Precision macro", thresholds));
		for(int i=0;i<my_k_s.length;i++) {
			precision_macro.add(Util.concat("k="+my_k_s[i], i, PanResult.o_precision, all_results));	
		}
				
		ArrayList<String> recall_macro = new ArrayList<String>();
		recall_macro.add(Util.concat("Recall macro", thresholds));
		for(int i=0;i<my_k_s.length;i++) {
			recall_macro.add(Util.concat("k="+my_k_s[i], i ,PanResult.o_recall, all_results));	
		}
		
		ArrayList<String> run_times = new ArrayList<String>();
		run_times.add(Util.concat("Run times", thresholds));
		for(int i=0;i<my_k_s.length;i++) {
			run_times.add(Util.concat("k="+my_k_s[i], i ,PanResult.o_run_time, all_results));	
		}
		System.out.println("**************");
		System.out.println("Aggregated results for OPH");
		for(int i=0;i<precision_macro.size();i++) {
			System.out.println(precision_macro.get(i)+"\t\t"+recall_macro.get(i)+"\t\t"+run_times.get(i));
		}
	}
	
	static void run_bible_correctness_experiment() {
		BibleResult br = BertBibleBase.oph_experiment();
		HashMap<Integer, BibleResult> appraoch_data = new HashMap<Integer, BibleResult>(1);
		appraoch_data.put(-1, br);
		
		BibleResult.compute_mapping_accuracy("OPH", appraoch_data );
	}
	
	static void run_wiki_runtime_experiment() {
		WikiDataLoader wdl = new WikiDataLoader();
		wdl.RESULTS_TO_FILE = false;
		wdl.threshold = 0.4;//The value to detect near duplicates
		wdl.all_solutions = Util.to_array(SemanticTest.OPH);
		wdl.run(WikiDataLoader.test_file);
	}
	
	public static void main(String[] args) {
		//run_bible_test_experiment();
		//run_pan_experiment();
		//run_bible_correctness_experiment();
		//run_bible_runtime_experiment();
		run_wiki_runtime_experiment();
	}
}
