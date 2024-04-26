package pan;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;


public class PanResult {
	private static ArrayList<PanResult> all_results = new ArrayList<PanResult>(1000);

	public static double connectivity_threshold;
	
	final String name;
	final int k;
	final double core_threshold;
	final double precision;
	final double recall;
	final double granularity;
	
	final double found_true_positives;
	final double all_true_positives;
	final double retrieved_elements;
	double granularity_local;
	
	public PanResult(String name, int k, double core_threshold
			, double precision, double recall, double granularity
			, double all_true_positives, double found_true_positives, double retrieved_elements){
		this.name 			= name;
		this.k 				= k;
		this.precision 		= precision;
		this.recall 		= recall;
		this.granularity	= granularity;
		this.core_threshold = core_threshold;
		
		this.found_true_positives = found_true_positives;
		this.all_true_positives   = all_true_positives;
		this.retrieved_elements   = retrieved_elements;
		all_results.add(this);
	}
	
	public String toString() {
		return name+"\t"+k+"\t"+core_threshold+"\t->"+"\tp="+precision+"\tr="+recall+"\tg="+granularity+"\ttp="+found_true_positives+"\tatp="+all_true_positives+"\tret="+retrieved_elements;
	}
	
	public static void clear() {
		all_results.clear();
	}
	
	public static ArrayList<PanResult> get_results(final String name) {
		ArrayList<PanResult> ret = new ArrayList<PanResult>();
		for(PanResult pr : PanResult.all_results) {
			if(pr.name.equals(name)) {
				ret.add(pr);
			}
		}
		return ret;
	}
	
	public static ArrayList<PanResult> get_results(final int k, final double threshold) {
		ArrayList<PanResult> ret = new ArrayList<PanResult>();
		for(PanResult pr : PanResult.all_results) {
			if(pr.k == k && pr.core_threshold == threshold) {
				ret.add(pr);
			}
		}
		return ret;
	}
	
	public static ArrayList<String> get_all_names(){
		HashSet<String> temp = new HashSet<String>(all_results.size());
		
		for(PanResult pr : PanResult.all_results) {
			temp.add(pr.name);
		}
		ArrayList<String> ret = new ArrayList<String>();
		for(String s : temp) {
			ret.add(s);
		}
		return ret;
	}
	public static ArrayList<Integer> get_all_k_values(){
		HashSet<Integer> temp = new HashSet<Integer>(all_results.size());
		
		for(PanResult pr : PanResult.all_results) {
			temp.add(pr.k);
		}
		ArrayList<Integer> ret = new ArrayList<Integer>();
		for(Integer i : temp) {
			ret.add(i);
		}
		Collections.sort(ret);
		return ret;
	}
	public static ArrayList<Double> get_all_thresholds(){
		HashSet<Double> temp = new HashSet<Double>(all_results.size());
		
		for(PanResult pr : PanResult.all_results) {
			temp.add(pr.core_threshold);
		}
		ArrayList<Double> ret = new ArrayList<Double>();
		for(Double d : temp) {
			ret.add(d);
		}
		Collections.sort(ret);
		return ret;
	}
}
