package oph;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class MinHash {
	static int[] T = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
	static int[] H = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19};
	
	static long[] large_primes = {294001, 505447, 584141, 604171, 971767, 1062599, 1282529, 1524181, 2017963, 2474431, 2690201, 3085553, 3326489, 4393139};
	
	/**
	 * min hashes of T
	 */
	static long[] h_T = {82, 59, 22, 57, 90, 39, 94, 42, 32, 64, 91, 48, 99, 73, 53};
	/**
	 * min hashes of S
	 */
	static long[] h_S = {90, 64, 39, 30, 66, 42, 22, 63, 28, 56, 91, 11, 96, 99, 53, 61, 88, 73, 31};
	
	/**
	 * Indicates that an OPH partition is empty. Needs to be larger than any other min hash value.
	 */
	public static final long EMPTY = Integer.MAX_VALUE;
	/**
	 * Number of Hash functions
	 */
	final int k;
	final long[] a;
	final long[] b;
	/**
	 * Large prime
	 * @param value
	 * @return
	 */
	final long[] p;
	
	static int num_oph_bins = 10;
	public static final int default_num_hash_functions = 1;//For OPh one is sufficient
	
	public MinHash(final int num_hash_functions) {
		this.k = num_hash_functions;
		this.a = new long[k];
		this.b = new long[k];
		this.p = new long[k];
		
		java.util.Random rand = new java.util.Random(1234567);
		
		if(this.k==1) {
			p[0] = 998244353;//The same prime as in OPH paper
			a[0] = Math.abs(rand.nextLong())%p[0];
			b[0] = Math.abs(rand.nextLong())%p[0];
			
		}
		//TODO dice the values init()
	}
	
	/**
	 * We have a forest of hash function.
	 * @param sequence
	 * @param from
	 * @param to
	 * @return
	 */
	long[] get_min_hash_vector(int[] sequence, int from, int to) {
		long[] min_hash_vector = new long[this.k];
		for(int i=0;i<min_hash_vector.length;i++) {
			min_hash_vector[i] = get_min_hash(sequence, from, to, i);
		}
		return min_hash_vector;
	}
	
	long get_min_hash(int[] sequence, int from, int to, int hash_func_index) {
		long min_hash = Long.MAX_VALUE;
		
		for(int i=from;i<to;i++) {
			long h = h(sequence[i],hash_func_index);
			if(h<min_hash){
				min_hash=h;
			}
		}
		return min_hash;
	}
	
	/**
	 * Returns the hash values for all tokens in [from,to] in 0-based array 
	 * @param sequence
	 * @param from
	 * @param to
	 * @return
	 */
	public long[] h(int[] sequence, int from, int to) {
		long[] vector = new long[to-from];
		
		for(int i=from;i<to;i++) {
			long h = h(sequence[i],0);//By definition we take only the first hash function, not the entire forest
			vector[from+i] = h;
		}
		
		return vector;
	}
	
	long h(long value, int hash_func_index) {
		long hash = a[hash_func_index]*value+b[hash_func_index];
		hash = hash % p[hash_func_index];
		return hash;
	}
	
	static final double estimate_jaccard_k_min_sketch(final long[] min_hash_vec_1, final long[] min_hash_vec_2) {
		//asserts min_hash_vec_1.length = min_hash_vec_2.length
		double sim = 0;
		
		for(int i=0;i<min_hash_vec_1.length;i++) {
			sim+= (min_hash_vec_1[i]==min_hash_vec_2[i]) ? 1 : 0;//increase the similarity by one if the min hashes are the same. 
		}
		
		sim /= (double) min_hash_vec_1.length;
		return sim;
	}
	
	static long[] get_oph_vector(long[] min_hash_vector){
		long[] oph_vector = new long[num_oph_bins];
		for(int i=0;i<oph_vector.length;i++) {
			oph_vector[i] = EMPTY;
		}
		
		for(long value : min_hash_vector) {
			int my_bin = get_bin(value);
			if(oph_vector[my_bin]>value) {//found new min hash for this partition
				oph_vector[my_bin]=value;
			}
		}
		return oph_vector;
	}
	
	long[] get_oph_vector(int[] sequence, int from, int to){
		long[] min_hash_vector = h(sequence, from, to);
		return get_oph_vector(min_hash_vector);
	}
	
	static final double estimate_jaccard_oph(final long[] min_hash_vec_1, final long[] min_hash_vec_2){
		//asserts min_hash_vec_1.length = min_hash_vec_2.length
		/**
		 * Number of same OPH partition min values. I think the abbreviation means matching.
		 */
		double n_mat = 0;
		/**
		 * Number of of jointly empty partitions
		 */
		double n_emp = 0;
		
		for(int i=0;i<min_hash_vec_1.length;i++) {
			 if(min_hash_vec_1[i]==EMPTY && min_hash_vec_2[i]==EMPTY) {
				 n_emp++;
			 }else if(min_hash_vec_1[i]==min_hash_vec_2[i]){
				 n_mat++;
			 }
		}
		
		double sim = n_mat / (((double)num_oph_bins)-n_emp);
		return sim;
	}
	
	public static void main(String[] args) {
		System.out.println(Arrays.toString(get_oph_vector(h_T)));
		System.out.println(Arrays.toString(get_oph_vector(h_S)));
		System.out.println("OPH sim()="+estimate_jaccard_oph(get_oph_vector(h_T), get_oph_vector(h_S)));
		CompactWindow.create_compact_window(T,h_T,9,6-1,13-1);//-1 because implementation is zero based
		ArrayList<CompactWindow> list = CompactWindow.create_all_compact_window_bin_alternate(T, h_T, 9);
		
		for(CompactWindow w : list) {
			System.out.println(w);
		}
		System.out.println("************ Empyt compact windows");
		ArrayList<ArrayList<CompactWindow>> all_lists = CompactWindow.create_all_compact_window(T, h_T);
		for(ArrayList<CompactWindow> l : all_lists) {
			for(CompactWindow w : l) {
				System.out.println(w);
			}
		}
		System.out.println("************ Non Empyt compact windows");
		ArrayList<ArrayList<NonEmptyCompactWindow>> all_non_empty = NonEmptyCompactWindow.create_all_compact_window(T, h_T);
		for(ArrayList<NonEmptyCompactWindow> l : all_non_empty) {
			for(NonEmptyCompactWindow w : l) {
				System.out.println(w);
			}
		}
	}

	public static int get_bin(long hash_value) {
		int my_bin = (int) (hash_value % (long)num_oph_bins);
		return my_bin;
	}
}
