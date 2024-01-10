package pan;

import java.util.ArrayList;
import java.util.BitSet;

import plus.data.Book;

/**
 * Das Sollte S zu allen R sein.
 * Wir machen das über bit sets
 * 
 * @author b1074672
 *
 */
public class FoundPlagiat {
	final int pair_id;
	final double threshold;
	
	ArrayList<Location> locs = new ArrayList<FoundPlagiat.Location>();
	final Book susp;
	final Book src;
	
	public FoundPlagiat(int pair_id, double threshold) {
		this.pair_id = pair_id;
		this.threshold = threshold;
		
		ArrayList<Book> temp = Data.load_entire_documents(pair_id);
		susp = temp.get(0);
		src  = temp.get(1);
	}
	
	public void add_found_location(int from_susp, int to_susp, int from_src, int to_src) {
		this.new Location(from_susp, to_susp, from_src, to_src);
	}
	
	private int[] get_ground_truth_location() {
		return Data.offsets_in_tokens(pair_id);
	}
	
	class Location implements Comparable<Location>{
		final int from_susp;
		final int to_susp;//inclusively
		
		final int from_src;
		final int to_src;//inclusively
		
		Location(int from_susp, int to_susp, int from_src, int to_src){
			this.from_susp = from_susp;
			this.to_susp = to_susp;
			
			this.from_src = from_src;
			this.to_src = to_src;
			
			locs.add(this);
		}

		@Override
		public int compareTo(Location arg0) {
			return Integer.compare(this.from_susp, arg0.from_susp);
		}
	}
	
	/**
	 * Metrics according to https://aclanthology.org/C10-2115.pdf
	 * @return
	 */
	/**
	 * Metrics according to https://aclanthology.org/C10-2115.pdf
	 * @param pan
	 * @param threshold
	 * @return
	 */
	public double precision() {
		final int size = susp.size();
		BitSet ground_truth_susp = new BitSet(size);
		ground_truth_susp.set(get_ground_truth_location()[0], get_ground_truth_location()[1]);//Note the indices //XXX das sind die character indices, nicht die der tokens
		BitSet found_in_susp = new BitSet(size);
		for(Location l : locs) {
			found_in_susp.set(l.from_susp, l.to_susp+1);//XXX end index inclusively
		}
		
		ground_truth_susp.and(found_in_susp);//|union s \in S(s union r)|
		
		double nominator = ground_truth_susp.cardinality();
		double denominator = get_ground_truth_location()[1] - get_ground_truth_location()[0];
		return nominator/denominator;
	}
	
	public double recall(PanResult pan, double threshold) {
		final int size = src.size();//XXX Check whether it is TokenizedParagraph
		BitSet ground_truth_src = new BitSet(size);
		ground_truth_src.set(get_ground_truth_location()[2], get_ground_truth_location()[3]);//Note the indices
		BitSet found_in_src = new BitSet(size);
		for(Location l : locs) {
			found_in_src.set(l.from_src, l.to_src+1);//XXX end index inclusively
		}
		
		ground_truth_src.and(found_in_src);
		
		double nominator = ground_truth_src.cardinality();
		double denominator = get_ground_truth_location()[3] - get_ground_truth_location()[2];
		return nominator/denominator;
	}
	/**
	 * Fragments detecting a plagiat - we want a 1:1 mapping 
	 * 
	 * S_R is for us always 1. R_s not.
	 * 
	 * @param pan
	 * @return
	 */
	public double gran(PanResult pan) {
		//compute R_s
		final int size = susp.size();
		BitSet ground_truth_susp = new BitSet(size);
		int plag_start = get_ground_truth_location()[0];
		int plag_stop= get_ground_truth_location()[1];
		
		ground_truth_susp.set(plag_start, plag_stop);
		BitSet found_in_susp = new BitSet(size);
		for(Location l : locs) {
			found_in_susp.set(l.from_susp, l.to_susp+1);//XXX end index inclusively
		}
		
		ground_truth_susp.and(found_in_susp);//|union s \in S(s union r)|
		int num_fragments = 0;
		for(int i=plag_start;i<plag_stop;i++) {
			if(ground_truth_susp.get(i)) {//first token of new detected fragment
				num_fragments++;
				while(i<plag_stop && ground_truth_susp.get(i)) {
					i++;
				}
			}
		}
		
		return num_fragments; 
	}
	public static plagdet(PanResult pan, double threshold) {
		return F(precision(pan, threshold), recall(pan, threshold)) / log_2(1+gran(pan));
	}

	/**
	 * Harmonic mean
	 * @return 
	 * @return
	 */
	private static double F(double val_1, double val_2) {
		float sum = 0; 
	    sum += 1.0d / val_1;
	    sum += 1.0d / val_2;
	    return 2.0d / sum; 
	}

	private double log_2(double d) {
		return Math.log(d) / Math.log(2.0d);
	}
}
