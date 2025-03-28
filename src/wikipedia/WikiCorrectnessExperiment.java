package wikipedia;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashSet;
import java.util.Random;

import org.apache.commons.math4.legacy.stat.correlation.PearsonsCorrelation;

import bert.SentenceEmbedding;
import bert.TopK_Result;
import boss.hungarian.Solutions;
import boss.test.SemanticTest;
import boss.util.Config;
import boss.util.Util;
import oph.MinHash;
import oph.OPH;

public class WikiCorrectnessExperiment {
	static int num_queries = 20;
	static Random rand = new Random(Util.seed);
	static int top_k = 10;
	static int k = 10;
	
	static int num_buckets = 10;
	
	/**
	 * computes for each line of the matrix (i.e., query) the top_k results
	 */
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
		
		out_box_plott("SeDA^1", m_seda, m_bert);
		out_box_plott("Jaccard^1", m_jaccard, m_bert);
		out_box_plott("Fast Text^1", m_fast_text, m_bert);
		
		// todo materialize matrices at ./results/wiki_correctness/
		to_file(m_bert,"m_bert");
		to_file(m_seda,"m_seda");
		to_file(m_jaccard,"m_jaccard");
		to_file(m_fast_text,"m_fast_text");
	}

	public static void to_file(double[][] matrix, String approac_name) {
		final String directory_path = "./results/wiki_correctness/";
		System.out.println("Writin matrix of size "+matrix.length+" x "+matrix[0].length+" to "+directory_path+approac_name+".tsv");
		File directory = new File(String.valueOf(directory_path));

		if (!directory.exists()) {
			new File(directory_path).mkdir();
			directory.mkdir();
		}
		
		try(BufferedWriter writer = new BufferedWriter(new FileWriter(directory_path+approac_name+".tsv"))){
			for(double[] arr : matrix) {
				writer.write(Util.outTSV(arr).trim());
				writer.newLine();
			}
		} catch (IOException e) {
			e.printStackTrace();
		}		
	}
	
	public static double[][] from_file(String approac_name) {
		final String directory_path = "./results/wiki_correctness/";
		System.out.print("Reading matrix from "+directory_path+approac_name+".tsv ");
		File directory = new File(String.valueOf(directory_path));

		if (!directory.exists()) {
			System.err.println("Directory does not exist "+directory);
			return null;
		}
		
		try(BufferedReader reader = new BufferedReader(new java.io.FileReader(directory_path+approac_name+".tsv"))){
			String line;
			ArrayList<Double> values = new ArrayList<Double>(100000);
			int counter = 0;
			double start = System.currentTimeMillis();
			
			while((line = reader.readLine())!=null) {
				String[] tokens = line.split("\t");
				for(String s : tokens) {
					double val = Double.parseDouble(s);
					values.add(val);
				}
				if(counter%1000==0 ) {
					System.out.print(counter+" ");
				}
				counter++;
			}
			double stop = System.currentTimeMillis();
			
			System.out.println(" in "+(stop-start)+" ms.");
			
			double[] temp = Util.toPrimitive(values);
			double[][] ret = {temp};
			return ret;
		} catch (IOException e) {
			e.printStackTrace();
		}		
		return null;
	}

	/**
	 * Outputs raw information to plot boxplott. To this end, we first partition the data according to the Bert similarity value which is in [0,1] into <code>num_buckets</code> buckets
	 * @param name
	 * @param m_bert
	 * @param other_matrix
	 */
	private static void out_box_plott(String name, double[][] m_bert, double[][] other_matrix) {
		System.out.println(name);
		final int size = m_bert.length;//asserts quadratic matrix 
		
		ArrayList<Double>[] values = new ArrayList[num_buckets+1];
		ArrayList<double[]>[] pairs_by_bucket = new ArrayList[num_buckets+1];
		
		for(int i=0;i<values.length;i++) {
			values[i] = new ArrayList<Double>();
			pairs_by_bucket[i] = new ArrayList<double[]>();
		}
		
		for(int line=0;line<size;line++) {
			for(int column=start_colum(line);column<size;column++) {//contains sim(i,i) = 1
				double d = m_bert[line][column];
				double value = other_matrix[line][column];
				double[] temp = {d,value};				
				d*=num_buckets;
				
				int bucket = (int) d;//The idea is that d is a value in [0,1], we have 100 buckets.
				pairs_by_bucket[bucket].add(temp);
				values[bucket].add(value);
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
		for(int bucket=0;bucket<pairs_by_bucket.length;bucket++) {
			double corr = correlation(pairs_by_bucket[bucket]);
			System.out.println(bucket+" "+corr);
		}
	}

	private static double correlation(ArrayList<double[]> list) {
		if(list.size()<2) {
			System.err.println("list.size()<2");
			return -2;
		}
		
		double[] vals_1 = new double[list.size()];
		double[] vals_2 = new double[list.size()];
		
		for(int i=0;i<list.size();i++) {
			double[] temp = list.get(i);
			vals_1[i] = temp[0];
			vals_2[i] = temp[1];
		}
		double corr = new PearsonsCorrelation().correlation(vals_1, vals_2);
		return corr;
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

	public static double[][] get_bert_matrix(ArrayList<double[]> vectors){
		System.out.print("get_bert_matrix() ");
		double start = System.currentTimeMillis(); 
		
		final int size = vectors.size(); 
		final double[][] matrix = new double[size][size];
		
		for(int line=0;line<size;line++) {
			final double[] line_vector = vectors.get(line);
			for(int column=line;column<size;column++) {//contains sim(i,i) = 1
				final double[] column_vector = vectors.get(column);
				final double my_score = Solutions.cosine_similarity(line_vector, column_vector);
				matrix[line][column] = my_score;
				matrix[column][line] = my_score;
			}
		}
		System.out.println("[Done] in "+(System.currentTimeMillis()-start)+" ms");
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
		//out_box_plott();
		//run();
		regression_task();
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
		
		double[] max_fast_text 	 = {0.5795806341930356,		0.7118758341016336,		0.774290702219472,	0.8259615239617432,		0.861835145321691,	0.9213943973152826,	0.9618542156295424,	0.9664883601993478,	1.0	};
		double[] upper_fast_text = {0.45295709831002173,	0.46424992754836636,	0.4922299335929125,	0.524983227352854,		0.5827490041743206,	0.6623702426110856,	0.7492570355272347,	0.8609590336369519,	0.9687048972225987};
		double[] median_fast_text= {0.4049729987038117,		0.41215902369381635,	0.4382924504492823,	0.4685629173280756,		0.5246630697322859,	0.6072148903407533,	0.6913312623206933,	0.8076968926979297,	0.9428558035388772};
		double[] lower_fast_text = {0.3404071617505871,		0.35758667136002353,	0.3820022085942308,	0.40880033867421567,	0.46170781869235555,0.5487748879130424,	0.6322526062737539,	0.7499938441922283,	0.9264710039016439};
		double[] min_fast_text 	 = {0.11948063799104948,	0.08249687691050613,	0.04842558450858241,0.049150281305637145,	0.08338380302143165,0.13698123428420386,0.36041187101166305,0.48316543051421224,	0.8437113579710914};
		
		double[] max_oph  	= {0.11000000000000001,	0.29999999999999993,	0.4800000000000001,	0.82,	0.8400000000000001,	0.8400000000000001,	0.8800000000000002,	0.8800000000000002,	0.8500000000000002};	
		double[] upper_oph	= {0.0,	0.0,	0.0,	0.09999999999999999,	0.09999999999999999,	0.15000000000000005,	0.27999999999999997,	0.5900000000000001,	0.8400000000000001};	
		double[] median_oph	= {0.0,	0.0,	0.0,	0.0,	0.04,	0.09999999999999999,	0.18000000000000005,	0.39,	0.8100000000000002};	
		double[] lower_oph	= {0.0,	0.0,	0.0,	0.0,	0.0,	0.06999999999999999,	0.09999999999999999,	0.27,	0.74};	
		double[] min_oph	= {0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.0,	0.32999999999999996};	
		
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
			//OPH
			out("\\addplot+[");
			out("boxplot prepared={");
			out("upper whisker ="+max_oph[i]+",");
			out("upper quartile="+upper_oph[i]+",");
			out("median        ="+median_oph[i]+",");
			out("lower quartile="+lower_oph[i]+",");
			out("lower whisker ="+min_oph[i]+"");
			out("},");
			out(",black,fill=gray,solid] coordinates {};");
		}
	}
	static void out(String s) {
		System.out.println("\t"+s);
	}
	
	/**
	 * Contains the code of the entire OPH wiki experiment
	 * @return
	 */
	public static double[][] get_oph_matrix() {		
		SentenceEmbedding bert_embedding = SentenceEmbedding.load_wikipedia_emebddings();
		double[][] m_bert = get_bert_matrix(bert_embedding.vectors);
		
		WikiDataLoader wdl = new WikiDataLoader();
		wdl.RESULTS_TO_FILE = false;
		wdl.threshold = 0.0;
		Config.wiki_k_s = Util.to_array(k);
		wdl.intput_sequence_length = Util.to_array(4000);//TODO length parameter not hard coded
		wdl.use_entire_doc = true;
		
		String line = wdl.load_file(WikiDataLoader.test_file);
		ArrayList<String> tokens = wdl.tokenize_txt_align(line);
		ArrayList<String> input = wdl.shorten_to_length(tokens, wdl.intput_sequence_length[0]);
		wdl.prepare_solution(input);
		
		double[] thresholds = {0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95};
		int sketch_size = OPH.sketch_size;
		OPH src_index = new OPH(wdl.raw_paragraphs_b1.get(0), sketch_size);//same sketch size as always, only one long paragraph 
		
		long[] min_hashes =  OPH.my_min_hasher.h(wdl.raw_paragraphs_b1.get(0),0,wdl.raw_paragraphs_b1.get(0).length);
		long[][] queries = OPH.create_hashed_windows(min_hashes, k);
		long[][] query_oph_vectors = MinHash.get_oph_vectors(queries);
		
		double[][] my_matrix = src_index.wiki_correctness_experiment(query_oph_vectors, k);
		
		/*
		double[][] my_matrix = new double[queries.length][];
		
		for(int i=0;i<queries.length;i++) {
			double[] token_similarity = src_index.query(i, queries, thresholds);
			double[] window_similarity = to_window_similarity(token_similarity, k, queries.length);
			my_matrix[i] = window_similarity;
			
		}*/
		if(m_bert.length!=my_matrix.length) {
			System.err.println("m_bert.length!=my_matrix.length "+m_bert.length+" "+my_matrix.length);
		}
		if(m_bert[0].length!=my_matrix[0].length) {
			System.err.println("m_bert[0].length!=my_matrix[0].length "+m_bert[0].length+" "+my_matrix[0].length);
		}
		
		System.out.println("OPH");
		System.out.println(Util.outTSV(get_histogram(my_matrix)));
		
		out_box_plott("OPH", m_bert, my_matrix);
		
		out_box_plott("OPH^1", my_matrix, m_bert);
		
		to_file(my_matrix,"m_oph");
		
		Regression reg = Regression.fit(m_bert, my_matrix);
		System.out.println("Regression.fit(m_bert, my_matrix)");
		System.out.println(reg);
		
		reg = Regression.fit(my_matrix, m_bert);
		System.out.println("Regression.fit(my_matrix, m_bert)");
		System.out.println(reg);
		
		return my_matrix;
	}

	private static double[] to_window_similarity(double[] token_similarity, int k, int length) {
		double[] window_similarity = new double[length];
		for(int w=0;w<length;w++) {//for each window
			double avg = 0;
			for(int i=0;i<k;i++) {
				avg += token_similarity[w+i];
			}
			avg /= (double) k;
			window_similarity[w] = avg;
		}
		return window_similarity;
	}
	
	public static void regression_task() {
		//(1) load all matrices
		double[][] m_bert 		= from_file("m_bert");
		double[][] m_fast_text 	= from_file("m_fast_text");
		double[][] m_jaccard 	= from_file("m_jaccard");
		double[][] m_seda 		= from_file("m_seda");
		double[][] m_oph 		= from_file("m_oph");
		
		System.out.println("m_bert");
		Regression reg_m_bert = Regression.fit(m_bert, m_bert);
		System.out.println(reg_m_bert);
		
		System.out.println("m_fast_text");
		Regression reg_m_fast_text = Regression.fit(m_fast_text, m_bert);
		System.out.println(reg_m_fast_text);
		
		System.out.println("m_jaccard");
		Regression reg_m_jaccard= Regression.fit(m_jaccard, m_bert);
		System.out.println(reg_m_jaccard);
		
		System.out.println("m_seda");
		Regression reg_m_seda= Regression.fit(m_seda, m_bert);
		System.out.println(reg_m_seda);
		
		System.out.println("m_oph");
		Regression reg_m_oph= Regression.fit(m_oph, m_bert);
		System.out.println(reg_m_oph);
		
		/*
		//Imagine, we use the approaches as filter what value
		double[] bert_min_thresholds = {0.5,0.6,0.7,0.8,0.9,0.95};
		System.out.println(Util.outTSV(bert_min_thresholds));
		
		{
			double[] fractions = reg_m_bert.filter_strength(bert_min_thresholds);
			System.out.println("m_bert\t"+Util.outTSV(fractions));
		}
		{
			double[] fractions = reg_m_fast_text.filter_strength(bert_min_thresholds);
			System.out.println("m_fast_text\t"+Util.outTSV(fractions));
		}
		{
			double[] fractions = reg_m_jaccard.filter_strength(bert_min_thresholds);
			System.out.println("m_jaccard\t"+Util.outTSV(fractions));
		}
		{
			double[] fractions = reg_m_seda.filter_strength(bert_min_thresholds);
			System.out.println("m_seda\t"+Util.outTSV(fractions));
		}
		{
			double[] fractions = reg_m_oph.filter_strength(bert_min_thresholds);
			System.out.println("m_oph\t"+Util.outTSV(fractions));
		}
		*/
		
		double[] bert_min_thresholds = {0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.97,1};
		System.out.println("thresholds\t"+Util.outTSV(bert_min_thresholds));
		double[] count_bert = count(m_bert, bert_min_thresholds);
		System.out.println("m_bert\t"+Util.outTSV(count_bert));
		
		double[][] count_fast_text = reg_m_fast_text.count(bert_min_thresholds);
		System.out.println("fast_text\t"+Util.outTSV(count_fast_text));
		System.out.println("reg_m_jaccard\t"+Util.outTSV(reg_m_jaccard.count(bert_min_thresholds)));
		System.out.println("reg_m_seda\t"+Util.outTSV(reg_m_seda.count(bert_min_thresholds)));
		System.out.println("reg_m_oph\t"+Util.outTSV(reg_m_oph.count(bert_min_thresholds)));
		
	}

	private static double[] count(double[][] m_bert, double[] bert_min_thresholds) {
		double[] counts = new double[bert_min_thresholds.length];
		for(double[] arr : m_bert) {
			for(double val : arr) {
				for(int i=0;i<bert_min_thresholds.length;i++) {
					if(val>=bert_min_thresholds[i]) {
						counts[i]++;
					}
				}
			}
		}
		return counts;
	}
}
