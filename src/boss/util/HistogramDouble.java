package boss.util;

import java.util.ArrayList;
import java.util.Collections;

public class HistogramDouble {
	final ArrayList<Double> raw_data;
	double[] bins_counts;
	double[] bins_min;
	double bin_width;
	
	public HistogramDouble(ArrayList<Double> raw_data){
		this(raw_data, 20);
	}
	
	public HistogramDouble(ArrayList<Double> raw_data, int num_bins){
		this.raw_data = raw_data;
		Collections.sort(raw_data);
		
		bins_counts = new double[num_bins];
		bins_min    = new double[num_bins];
		bin_width   = 0.05;
		double min_value = 0.05;
		for(int i=1;i<bins_min.length;i++) {
			bins_min[i] = min_value;
			min_value += bin_width;
		}
		for(double value : raw_data){
			int bin = get_bin(value);
			bins_counts[bin]++;
		}
	}
	
	private int get_bin(double value) {
		for(int i=bins_min.length-1;i>=0;i--) {
			if(bins_min[i]<=value) {
				return i;
			}
		}
		return 0;
	}

	public int size() {
		return raw_data.size();
	}
	
	public double sum() {
		double sum = 0;
		for(double i : raw_data) {
			sum+=i;
		}
		return sum;
	}
	
	public double get_min() {
		return raw_data.get(0);
	}
	public double get_max() {
		return raw_data.get(size()-1);
	}
	public String getStatistics() {
		String header = "#Pairs\t#Distance\tavg(computations)";
		String data   = size()+"\t"+sum()+"\t"+(double)sum()/(double)size();
		return header+"\n"+data;
	}
	public String toString() {
		String header = "";
		String data   = "";
		for(int bin=0;bin<bins_counts.length;bin++) {
			header+="<="+bins_min[bin]+"\t";
			data  +=bins_counts[bin]+"\t";
		}
		
		return header+"\n"+data;
	}
}
