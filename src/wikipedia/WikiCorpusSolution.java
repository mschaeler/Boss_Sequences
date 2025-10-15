package wikipedia;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;

import boss.hungarian.Solutions;
import boss.test.SemanticTest;
import boss.util.MyArrayList;
import wikipedia.Corpus.CorpusArticle;

public class WikiCorpusSolution {
	double[] my_run_times = new double[2];
	
	public WikiCorpusSolution(HashMap<Integer, double[]> similarities, CorpusArticle query, CorpusArticle article, MyArrayList[] all_runs, HashMap<Integer, double[]> embedding_vector_index, int k, double threshold, int solution_enum) {
		//System.out.println("WikiCorpusSolution(query="+query.article_number+", article="+article.article_number+")");
		//double start = System.currentTimeMillis();
		HashSet<Integer> all_ids = new HashSet<Integer>();
		for(int token : query.unique_tokens) {
			all_ids.add(token);
		}
		for(int token : article.unique_tokens) {
			all_ids.add(token);
		}
		ArrayList<Integer> all_tokens_ordered = new ArrayList<Integer>(all_ids.size());
		for(int token : all_ids) {
			all_tokens_ordered.add(token);
		}
		Collections.sort(all_tokens_ordered);//TODO
		final int max_id = all_tokens_ordered.size();//Last element has max id
		
		
		HashMap<Integer, Integer> new_tokenids = new HashMap<Integer, Integer>(all_tokens_ordered.size());
		HashMap<Integer, double[]> new_embedding_vector_index = new HashMap<Integer, double[]>(all_tokens_ordered.size()); 
		for(int new_id=0;new_id<all_tokens_ordered.size();new_id++){
			int old_id = all_tokens_ordered.get(new_id);
			new_tokenids.put(old_id,new_id);
			double[] my_vector = embedding_vector_index.get(old_id);
			if(my_vector==null) {
				System.err.println("my_vector==null");
			}
			new_embedding_vector_index.put(new_id, my_vector);
		}
		
		//Now create the wrappers for the Solution class.
		
		int[] raw_paragraph_b1 = new int[article.my_tokens.length];
		for(int i=0;i<raw_paragraph_b1.length;i++) {
			int old_id = article.my_tokens[i];
			Integer new_id = new_tokenids.get(old_id);
			raw_paragraph_b1[i] = new_id.intValue();
		}
		
		int[] raw_paragraph_b2 = new int[query.my_tokens.length];
		for(int i=0;i<raw_paragraph_b2.length;i++) {
			int old_id = query.my_tokens[i];
			Integer new_id = new_tokenids.get(old_id);
			raw_paragraph_b2[i] = new_id.intValue();
		}
		
		//TODO measure time
		double start = System.currentTimeMillis();
		int max_id_query   = query.unique_tokens[query.unique_tokens.length-1];
		int max_id_article = article.unique_tokens[article.unique_tokens.length-1];
		double[][] sim = new double[new_tokenids.get(max_id_article)+1][];
		
		final int array_length = new_tokenids.get(max_id_query)+1;
		int[] unique_tokens_b1 = new int[article.unique_tokens.length];
		for(int i=0;i<article.unique_tokens.length;i++) {
			int old_id_a = article.unique_tokens[i];
			final int new_id_a = new_tokenids.get(old_id_a).intValue();
			sim[new_id_a] = new double[array_length];
			unique_tokens_b1[i] = new_id_a;
		}
		
		for(int old_id_q : query.unique_tokens){
			final int new_id_q = new_tokenids.get(old_id_q).intValue();
			final double[] sim_line_all_tokens = similarities.get(old_id_q);
			
			for(int old_id_a : article.unique_tokens) {
				int new_id_a = new_tokenids.get(old_id_a).intValue();
				double sim_value = sim_line_all_tokens[old_id_a];
				sim[new_id_a][new_id_q] = sim_value;
			}
		}
		double stop = System.currentTimeMillis();
		//System.out.println("Start-stop "+(stop-start));
		
		
		int[] unique_tokens_b2 = new int[query.unique_tokens.length];
		for(int i=0;i<query.unique_tokens.length;i++) {
			int old_id_q = query.unique_tokens[i];
			final int new_id_q = new_tokenids.get(old_id_q).intValue();
			unique_tokens_b2[i] = new_id_q;
		}
		
		start = System.currentTimeMillis();
		Solutions s = new Solutions(sim, raw_paragraph_b1, raw_paragraph_b2, k, threshold, new_embedding_vector_index, max_id, unique_tokens_b1, unique_tokens_b2);
		
		double[] run_times;
		if(solution_enum==SemanticTest.SOLUTION) {
			run_times = s.run_solution();	
		}else if(solution_enum==SemanticTest.CORPUS){
			run_times = s.run_solution_corpus(all_runs);	
		}else if(solution_enum==SemanticTest.FAST_TEXT) {
			run_times = s.run_fast_text();
		}else{
			run_times = s.run_naive();
		}
		stop = System.currentTimeMillis();
		System.out.println("WikiCorpusSolution(query="+query.article_number+", article="+article.article_number+") Create Solution in "+(stop-start)+ "ms runtime="+run_times[0]);
		my_run_times[0] = stop-start;
		my_run_times[1] = run_times[0];
		//System.out.println();
	}
}
