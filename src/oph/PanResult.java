package oph;

import java.util.ArrayList;
import java.util.BitSet;

import boss.util.Util;

public class PanResult {
	final int pair;
	final ArrayList<Integer> k_s = new ArrayList<Integer>();
	final ArrayList<Double> run_times = new ArrayList<Double>();
	final ArrayList<BitSet> marked_src = new ArrayList<BitSet>();
	final ArrayList<BitSet> marked_sup = new ArrayList<BitSet>();
	
	final BitSet ground_truth_src;
	final BitSet ground_truth_susp;
	
	public PanResult(int pair, BitSet ground_truth_src, BitSet ground_truth_susp){
		this.pair = pair;
		this.ground_truth_src = ground_truth_src;
		this.ground_truth_susp = ground_truth_susp;
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
	String header = "k\tnum_true_positives\tfound_src\tfound_susp\ttotal_num_found\trecall\tprecision\trun_time";
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
			
			double total_num_found = marked_src.get(i).cardinality()+marked_sup.get(i).cardinality();
			double precision = num_true_positives / total_num_found;
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
}
