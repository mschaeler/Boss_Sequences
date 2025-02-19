package wikipedia;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
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
	static int top_k = 10;
	static int k = 10;
	
	static int num_buckets = 10;
	
	static void run() {
		SentenceEmbedding bert_embedding = SentenceEmbedding.load_wikipedia_emebddings();
		double[][] m_bert = get_bert_matrix(bert_embedding.vectors);
		double[][] m_seda;
		double[][] m_jaccard;
		double[][] m_fast_text;
		
		System.out.println(Util.outTSV(get_histogram(m_bert)));
		ArrayList<TopK_Result> above_theta = get_top_k(m_bert);
		
		for(int i=0;i<above_theta.size();i++) {
			TopK_Result e = above_theta.get(i);
			System.out.println(e+"\t"+bert_embedding.sentences.get(e.offset_src)+"\t"+bert_embedding.sentences.get(e.offset_susp));
		}
		
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
		double[][] alignement_matrix = m_seda = wdl.last_result;
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
		
		alignement_matrix = m_fast_text = wdl.last_result;
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
		
		alignement_matrix = m_jaccard = wdl.last_result;
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
		
		System.out.println("SeDA");
		System.out.println(Util.outTSV(get_histogram(m_seda)));
		
		System.out.println("Jaccard");
		System.out.println(Util.outTSV(get_histogram(m_jaccard)));
		
		System.out.println("Fast Text");
		System.out.println(Util.outTSV(get_histogram(m_fast_text)));
		
		out_box_plott("SeDA", m_bert, m_seda);
		out_box_plott("Jaccard", m_bert, m_jaccard);
		out_box_plott("Fast Text", m_bert, m_fast_text);
	}

	private static void out_box_plott(String name, double[][] m_bert, double[][] other_matrix) {
		System.out.println(name);
		final int size = m_bert.length;//asserts quadratic matrix 
		
		ArrayList<Double>[] values = new ArrayList[num_buckets+1];
		for(int i=0;i<values.length;i++) {
			values[i] = new ArrayList<Double>();
		}
		
		for(int line=0;line<size;line++) {
			for(int column=start_colum(line);column<size;column++) {//contains sim(i,i) = 1
				double d = m_bert[line][column];
				d*=num_buckets;
				int bucket = (int) d;//The idea is that d is a value in [0,1], we have 100 buckets.
				values[bucket].add(other_matrix[line][column]);
			}
		}
		String max = "";
		String upper_quantile = "";
		String median = "";
		String lower_quantile = "";
		String min = "";
		for(ArrayList<Double> list : values) {
			Collections.sort(list);
			if(list.size()<4) {
				continue;
			}
			max += list.get(list.size()-1)+"\t";
			upper_quantile += list.get(list.size()/4*3)+"\t";
			median += list.get(list.size()/2)+"\t";
			lower_quantile += list.get(list.size()/4)+"\t";
			min += list.get(0)+"\t";
		}
		System.out.println(max);
		System.out.println(upper_quantile);
		System.out.println(median);
		System.out.println(lower_quantile);
		System.out.println(min);
		
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

	static double[][] get_bert_matrix(ArrayList<double[]> vectors){
		System.out.print("get_bert_matrix() ");
		double start = System.currentTimeMillis(); 
		
		final int size = vectors.size(); 
		final double[][] matrix = new double[size][size];
		
		for(int line=0;line<size;line++) {
			final double[] line_vector = vectors.get(line);
			for(int column=start_colum(line);column<size;column++) {//contains sim(i,i) = 1
				final double[] column_vector = vectors.get(column);
				final double my_score = Solutions.cosine_similarity(line_vector, column_vector);
				matrix[line][column] = my_score;
				matrix[column][line] = my_score;
			}
		}
		System.out.println("[Done] in"+(System.currentTimeMillis()-start)+" ms");
		return matrix;
	}
	
	static final int[] get_histogram(final double[][] matrix) {
		final int size = matrix.length;//asserts quadratic matrix 
		final int[] histogram = new int[num_buckets+1];
		for(int line=0;line<size;line++) {
			for(int column=start_colum(line);column<size;column++) {//contains sim(i,i) = 1
				double d = matrix[line][column];
				d*=num_buckets;
				int bucket = (int) d;//The idea is that d is a value in [0,1], we have 100 buckets.
				histogram[bucket]++;
			}
		}
		return histogram;
	}
	
	static int start_colum(final int line) {
		return line+k+100;//far away form the current sentence
	}
	
	static final ArrayList<TopK_Result> get_top_k(final double[][] matrix) {
		final int size = matrix.length;//asserts quadratic matrix 
		ArrayList<TopK_Result> res = new ArrayList<TopK_Result>(100);
		
		for(int line=0;line<size;line++) {
			for(int column=start_colum(line);column<size;column++) {
				double d = matrix[line][column];
				if(d>=0.8){
					res.add(new TopK_Result(line, column, d));
				}
			}
		}
		Collections.sort(res);
		
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
		out_box_plott();
		//run();
	}
	
	static void out_box_plott() {
		double[] max_seda 	= {0.30985871,	0.412068291,	0.497860456,	0.611231775,	0.668377259,	0.742696912,	0.881663344,	0.90332167,	1};
		double[] upper_seda = {0.21294558,	0.218886792,	0.233358929,	0.253317323,	0.298280159,	0.384644612,	0.508904817,	0.703489409,	0.919667305};
		double[] median_seda= {0.192440121,	0.196684409,	0.207624165,	0.221934423,	0.254870066,	0.327484642,	0.429701685,	0.618147205,	0.845604706};
		double[] lower_seda = {0.170073216,	0.175334106,	0.184258036,	0.195051304,	0.219740619,	0.277604083,	0.370940902,	0.541435056,	0.834873176};
		double[] min_seda 	= {0.098083906,	0.075029255,	0.061816072,	0.057501536,	0.074920837,	0.095684387,	0.176531213,	0.221907185,	0.661531527};
				
		double[] max_jaccard 	= {0.058823529,	0.111111111,	0.176470588,	0.214285714,	0.416666667,	0.545454545,	0.727272727,	0.9,	1};
		double[] upper_jaccard  = {0.01,	0.01,	0.01,	0.01,	0.052631579,	0.111111111,	0.2,	0.461538462,	0.818181818};
		double[] median_jaccard = {0,	0,	0,	0,	0,	0.055555556,	0.125,	0.357142857,	0.727272727};
		double[] lower_jaccard  = {0,	0,	0,	0,	0,	0,	0.0625,	0.25,	0.666666667};
		double[] min_jaccard 	= {0,	0,	0,	0,	0,	0,	0,	0,	0.333333333};
		
		double[] max_fast_text 	 = {0.057958063,	0.071187583,	0.07742907,		0.082596152,	0.086183515,	0.09213944,		0.096185422,	0.096648836,	0.1};
		double[] upper_fast_text = {0.04529571,		0.046424993,	0.049222993,	0.052498323,	0.0582749,		0.066237024,	0.074925704,	0.086095903,	0.09687049};
		double[] median_fast_text= {0.0404973,		0.041215902,	0.043829245,	0.046856292,	0.052466307,	0.060721489,	0.069133126,	0.080769689,	0.09428558};
		double[] lower_fast_text = {0.034040716,	0.035758667,	0.038200221,	0.040880034,	0.046170782,	0.054877489,	0.063225261,	0.074999384,	0.0926471};
		double[] min_fast_text 	 = {0.011948064,	0.008249688,	0.004842558,	0.004915028,	0.00833838,		0.013698123,	0.036041187,	0.048316543,	0.084371136};
		
		for(int i=0;i<max_seda.length;i++) {
			//SeDA
			out("\\addplot+[");
			out("boxplot prepared={");
			out("upper whisker ="+max_seda[i]+",");
			out("upper quartile="+upper_seda[i]+",");
			out("median        ="+median_seda[i]+",");
			out("lower quartile="+lower_seda[i]+",");
			out("lower whisker ="+min_seda[i]+"");
			out("},");
			out(",black,fill=blue,solid] coordinates {};");
			//Jaccard
			out("\\addplot+[");
			out("boxplot prepared={");
			out("upper whisker ="+max_jaccard[i]+",");
			out("upper quartile="+upper_jaccard[i]+",");
			out("median        ="+median_jaccard[i]+",");
			out("lower quartile="+lower_jaccard[i]+",");
			out("lower whisker ="+min_jaccard[i]+"");
			out("},");
			out(",black,fill=red,solid] coordinates {};");
			//Fast Text
			out("\\addplot+[");
			out("boxplot prepared={");
			out("upper whisker ="+max_fast_text[i]+",");
			out("upper quartile="+upper_fast_text[i]+",");
			out("median        ="+median_fast_text[i]+",");
			out("lower quartile="+lower_fast_text[i]+",");
			out("lower whisker ="+min_fast_text[i]+"");
			out("},");
			out(",black,fill=gray,solid] coordinates {};");
		}
	}
	static void out(String s) {
		System.out.println("\t"+s);
	}
}
