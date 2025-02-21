package oph;

import java.util.ArrayList;
import java.util.BitSet;

import boss.util.Util;

/**
 * Refers to one (src, susp) document pair of the PAN corpus.
 * @author b1074672
 *
 */
public class PanResult {
	final int pair;
	final ArrayList<Integer> k_s = new ArrayList<Integer>();
	final ArrayList<Double> run_times = new ArrayList<Double>();
	final ArrayList<BitSet> marked_src = new ArrayList<BitSet>();
	final ArrayList<BitSet> marked_sup = new ArrayList<BitSet>();
	
	final BitSet ground_truth_src;
	final BitSet ground_truth_susp;
	
	final double threshold;
	
	public PanResult(int pair, double threshold, BitSet ground_truth_src, BitSet ground_truth_susp){
		this.pair = pair;
		this.ground_truth_src = ground_truth_src;
		this.ground_truth_susp = ground_truth_susp;
		this.threshold = threshold;
	}
	
	void add(int k, BitSet marked_src, BitSet marked_sup, double run_time){
		BitSet temp = (BitSet) marked_src.clone();
		this.marked_src.add(temp);
		temp = (BitSet) marked_sup.clone();
		this.marked_sup.add(temp);
		this.run_times.add(run_time);
		this.k_s.add(k);
	}
	
	double[][] all_results;
	static String header = "k\tnum_true_positives\tfound_src\tfound_susp\ttotal_num_found\trecall\tprecision\trun_time";
	void analyze() {
		all_results = new double[k_s.size()][];
		double num_true_positives = ground_truth_src.cardinality()+ground_truth_susp.cardinality();
		System.out.println(header);
		
		for(int i=0;i<k_s.size();i++) {
			//determine found src
			BitSet buffer = new BitSet(marked_src.size());
			buffer.or(ground_truth_src);
			buffer.and(marked_src.get(i));
			double found_src = buffer.cardinality();
			
			buffer.clear();
			buffer.or(ground_truth_susp);
			buffer.and(marked_sup.get(i));
			double found_susp = buffer.cardinality();
			
			double recall = (found_src+found_susp) / num_true_positives;
			if(recall < 0) {
				System.err.println("recall < 0");
			}
			if(recall > 1) {
				System.err.println("recall > 1");
			}
			
			//precision
			double total_num_found = marked_src.get(i).cardinality()+marked_sup.get(i).cardinality();
			double precision = (total_num_found>0) ? (found_src+found_susp) / total_num_found : 0;
			if(precision<0) {
				System.err.println("precision < 0");
			}
			if(precision>1) {
				System.err.println("precision > 1");
			}
			
			double[] result = {k_s.get(i),num_true_positives,found_src,found_susp,total_num_found,recall,precision, run_times.get(i)};
			all_results[i] = result;
			
			System.out.println(Util.outTSV(result));
			//System.out.println(k_s.get(i)+"\t"+num_true_positives+"\t"+found_src+"\t"+found_susp+"\t"+total_num_found+"\t"+recall+"\t"+precision);
		}
		
		
	}
	
	public String toString() {
		String s = header+"\n";
		for(double[] arr : all_results) {
			s+= Util.outTSV(arr)+"\n";
		}
		return s;
	}

	static final int o_k 					= 0;
	static final int o_num_true_positives	= 1;
	static final int o_found_src 			= 2;
	static final int o_found_susp 			= 3;
	static final int o_total_num_found 		= 4;
	static final int o_recall 				= 5;
	static final int o_precision 			= 6;
	static final int o_run_time 			= 7;
	
	public static double[][] aggregate(PanResult[] all_results) {
		double num_file_pairs = all_results.length;
		int num_k_s = all_results[0].k_s.size();
		
		if(all_results[0]==null) {
			System.err.println("all_results[0]==null");
		}
		/**
		 * [k][measures]
		 */
		double[][] aggregate = new double[num_k_s][o_run_time+1];
		
		for(int i=0;i<all_results[0].k_s.size();i++) {
			aggregate[i][o_k] = all_results[0].all_results[i][o_k];//set the k values
		}
		
		for(PanResult pr : all_results) {//for each document pair
			
			for(int i=0;i<pr.k_s.size();i++) {
				if(aggregate[i][o_k] != pr.all_results[i][o_k]) {
					System.err.println("aggregate[i][o_k] != pr.all_results[i][o_k]");
				}
				
				aggregate[i][o_num_true_positives] 	+= pr.all_results[i][o_num_true_positives];
				aggregate[i][o_found_src] 			+= pr.all_results[i][o_found_src];
				aggregate[i][o_found_susp] 			+= pr.all_results[i][o_found_susp];
				aggregate[i][o_total_num_found]		+= pr.all_results[i][o_total_num_found];
				aggregate[i][o_recall] 				+= pr.all_results[i][o_recall];
				aggregate[i][o_precision] 			+= pr.all_results[i][o_precision];
				aggregate[i][o_run_time] 			+= pr.all_results[i][o_run_time];
			}
		}
		
		System.out.println("Aggregated results "+all_results[0].threshold);
		normalize(aggregate, num_file_pairs);
		System.out.println(header);
		for(double[] arr : aggregate) {
			System.out.println(Util.outTSV(arr));
		}
		return aggregate;
	}

	private static void normalize(double[][] matrix, double normalize_by) {
		for(double[] arr : matrix) {
			for(int i=1;i<arr.length;i++) {//first index is k
				arr[i] /= normalize_by;
			}
		}
		
	}
}
