package wikipedia;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;

import bert.SentenceEmbedding;
import bert.TopK_Result;
import boss.hungarian.Solutions;
import boss.test.SemanticTest;
import boss.util.Config;
import boss.util.Util;

public class WikiCorrectnessExperiment {
	static int num_queries = 20;
	static Random rand = new Random(Util.seed);
	static int top_k = 30;
	static int k = 10;
	
	static void run() {
		SentenceEmbedding bert_embedding = SentenceEmbedding.load_wikipedia_emebddings();
		HashSet<Integer> query_ids = new HashSet<Integer>(num_queries);
		int num_windows = bert_embedding.vectors.size();
		
		while(query_ids.size()!=num_queries) {
			int id = rand.nextInt(num_windows);
			query_ids.add(id);
		}
		
		int[] queries = Util.toPrimitive(query_ids);
		Arrays.sort(queries);
		
		ArrayList<TopK_Result[]> all_results = new ArrayList<TopK_Result[]>(num_queries);
		
		for(int query_id : queries){
			TopK_Result[] res = get_top_k(query_id, bert_embedding.vectors);
			all_results.add(res);
		}
		
		WikiDataLoader wdl = new WikiDataLoader();
		wdl.RESULTS_TO_FILE = false;
		wdl.threshold = 0.0;
		Config.wiki_k_s = Util.to_array(k);
		wdl.intput_sequence_length = Util.to_array(4000);//TODO length parameter not hard coded
		wdl.use_entire_doc = true;
		
		
		wdl.all_solutions = Util.to_array(SemanticTest.NAIVE);
		wdl.run(WikiDataLoader.test_file);
		System.out.println("SeDA");
		double[][] alignement_matrix = wdl.last_result;
		for(int i=0;i<queries.length;i++){
			int query_id = queries[i];
			TopK_Result[] res = get_top_k(alignement_matrix[query_id], query_id);
			System.out.print("My result for id="+query_id+"\t");
			//TopK_Result.out(res);
			//TopK_Result.out(all_results.get(i));
			int overlap = TopK_Result.overlap(res, all_results.get(i));
			System.out.println(overlap+"\tof\t"+top_k+"\t"+((double)overlap/(double)top_k));
		}
		
		wdl.all_solutions = Util.to_array(SemanticTest.FAST_TEXT);
		wdl.run(WikiDataLoader.test_file);
		System.out.println("FAST_TEXT");
		
		alignement_matrix = wdl.last_result;
		for(int i=0;i<queries.length;i++){
			int query_id = queries[i];
			TopK_Result[] res = get_top_k(alignement_matrix[query_id], query_id);
			System.out.print("My result for id="+query_id+"\t");
			//TopK_Result.out(res);
			//TopK_Result.out(all_results.get(i));
			int overlap = TopK_Result.overlap(res, all_results.get(i));
			System.out.println(overlap+"\tof\t"+top_k+"\t"+((double)overlap/(double)top_k));
		}
		
		wdl.all_solutions = Util.to_array(SemanticTest.JACCARD);
		wdl.run(WikiDataLoader.test_file);
		System.out.println("Jaccard");
		
		alignement_matrix = wdl.last_result;
		for(int i=0;i<queries.length;i++){
			int query_id = queries[i];
			TopK_Result[] res = get_top_k(alignement_matrix[query_id], query_id);
			System.out.print("My result for id="+query_id+"\t");
			//TopK_Result.out(res);
			//TopK_Result.out(all_results.get(i));
			int overlap = TopK_Result.overlap(res, all_results.get(i));
			System.out.println(overlap+"\tof\t"+top_k+"\t"+((double)overlap/(double)top_k));
		}
		
		//For all solution enums get the corresponding matrix //
		
	}

	private static TopK_Result[] get_top_k(double[] matrix_line, int query_id) {
		TopK_Result[] res = new TopK_Result[top_k];
		for(int i=0;i<top_k;i++) {
			res[i] = new TopK_Result(-1, -1, -1);
		}
		for(int anwser_id=0;anwser_id<matrix_line.length;anwser_id++) {
			if(is_in(anwser_id, query_id, k)) continue;//do not return thr query itself
			final double my_score = matrix_line[anwser_id];
			if(res[0].score<my_score) {
				res[0] = new TopK_Result(anwser_id,anwser_id,my_score);
				Arrays.sort(res);
			}
		}
		return res;
	}

	private static TopK_Result[] get_top_k(int query_id, ArrayList<double[]> vectors) {
		TopK_Result[] res = new TopK_Result[top_k];
		for(int i=0;i<top_k;i++) {
			res[i] = new TopK_Result(-1, -1, -1);
		}
		
		final double[] query_vectors = vectors.get(query_id);
		for(int anwser_id=0;anwser_id<vectors.size();anwser_id++) {
			if(is_in(anwser_id, query_id, k)) continue;//do not return thr query itself
			final double[] other_vec = vectors.get(anwser_id);
			final double my_score = Solutions.cosine_similarity(query_vectors, other_vec);
			
			if(res[0].score<my_score) {
				res[0] = new TopK_Result(anwser_id,anwser_id,my_score);
				Arrays.sort(res);
			}
		}
		
		return res;
	}
	
	private static boolean is_in(int anwser_id, int query_id, int border) {
		if(Math.abs(anwser_id-query_id)<border) {
			return true;
		}
		return false;
	}

	public static void main(String[] args) {
		run();
	}
}
