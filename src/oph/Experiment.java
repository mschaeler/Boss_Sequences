package oph;

import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;

import boss.lexicographic.Tokenizer;
import boss.load.ImporterAPI;
import boss.semantic.Sequence;
import boss.test.SemanticTest;
import boss.util.Config;
import plus.data.Book;

public class Experiment {
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
	
	static void run_pan_experiment() {
		ArrayList<Book>[] all_pairs_excerpt = pan.Data.load_all_plagiarism_excerpts();
		ArrayList<Book>[] all_pairs = pan.Data.load_all_entire_documents();
		PanResult[] all_results = new PanResult[all_pairs.length];
		
		for(int pair=0;pair<all_pairs.length;pair++) {
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
			
			PanResult pr = new PanResult(pair,gtruth_src,gtruth_susp);
			all_results[pair] = pr;
			/*System.out.println("Offsets src ["+offset_src+","+(offset_src+raw_excerpt_src.length)+"]");
			System.out.println("Offsets susp ["+offset_susp+","+(offset_susp+raw_excerpt_sups.length)+"]");*/
			
			int sketch_size = 32;
			OPH src = new OPH(raw_paragraphs_src, sketch_size);
			
			ArrayList<Double> run_times = new ArrayList<Double>(Config.k_s.length);
			int[] my_k_s = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30};
			for(int k : my_k_s) {
				src.query(raw_paragraphs_sups, 0.3, k);
				run_times.add(src.get_runtime());
				pr.add(k, src.marked_src(), src.marked_sup(), src.get_runtime());
				
			}
			pr.analyze();
			System.out.println(run_times);
			
		}
		for(PanResult pr : all_results) {
			System.out.println(pr);
		}
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
	
	
	public static void main(String[] args) {
		//run_bible_test_experiment();
		run_pan_experiment();
	}
}
