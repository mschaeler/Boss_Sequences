package oph;

import java.util.Arrays;
import java.util.HashSet;

public class TxtAlign {
	final int[] sequence;
	final long[] min_hashes;
	
	public TxtAlign(int[] sequence, long[] min_hashes){
		this.sequence = sequence;
		this.min_hashes = min_hashes;
		if(sequence.length!=min_hashes.length) {
			System.err.println("sequence.length!=min_hashes.length");
		}
	}
	
	long[] get_bottom_k_sketch(int k) {
		long[] buffer = new long[min_hashes.length];
		System.arraycopy(min_hashes, 0, buffer, 0, min_hashes.length);
		Arrays.sort(buffer);
		long[] bottom_k_sketch = new long[k];
		for(int i=0;i<k;i++) {
			bottom_k_sketch[i] = buffer[i];
		}
		return bottom_k_sketch;
	}
	
	class LocalMininmun{
		final int k;
		final HashSet<Integer> positions = new HashSet<Integer>();
		final long[] bottom_k_sketch;
		//final long my_min;
		final long my_max;
		
		final int from, to;
		
		LocalMininmun(long[] bottom_k_sketch){
			this(bottom_k_sketch, 0, sequence.length-1);
		}
		
		LocalMininmun(long[] bottom_k_sketch, final int from, int to){
			this.k = bottom_k_sketch.length;
			this.bottom_k_sketch = bottom_k_sketch;
			Arrays.sort(this.bottom_k_sketch);//TODO check for duplicates?
			 
			this.to = to;
			this.from = from;
			
			//find the position that have the bottom_k_sketch hash values
			for(int i=from;i<=to;i++) {
				for(int j=0;j<k;j++) {
					if(bottom_k_sketch[j] == min_hashes[i]) {
						positions.add(i);
					}
				}
			}
			
			
			my_max = bottom_k_sketch[k-1];
			check();
		}

		private boolean check() {
			boolean is_correct = true;
			
			for(int i=from;i<=to;i++) {
				if(!positions.contains(i)) {
					if(min_hashes[i]<my_max) {
						System.err.println("min_hashes[i]<my_max");
						return false;
					}
				}
			}
			
			return is_correct;
		}
	}
	
	private static final boolean not_is_in(final int pos, final int[] positions) {
		for(int p : positions) {
			if(p == pos) {
				return true;
			}
		}
		return false;
	}
	
	public static void main(String[] args) {
		int[] i = {1,2,3,4};
		int[] j = {14,15,16};
		int[] sequence = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
		long[] min_hashes = {68,57,53,13,28,66,17,35,44,20,77,31,49,10,61,38};
		
		TxtAlign ta = new TxtAlign(sequence, min_hashes);
		long[] bottom_k_sketch = ta.get_bottom_k_sketch(3);
		
		LocalMininmun lm = ta.new LocalMininmun(bottom_k_sketch);
	}
}
