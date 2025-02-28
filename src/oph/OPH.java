package oph;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;

public class OPH {
	static boolean debug = false;
	static {
		if (debug)
			System.err.println("DEBUG");
	}
	
	static final MinHash my_min_hasher = new MinHash(MinHash.default_num_hash_functions);
	final long[] my_min_hashes;
	final long[] my_oph_vector;
	final ArrayList<ArrayList<CompactWindow>> empty_windows;
	final ArrayList<ArrayList<NonEmptyCompactWindow>> non_empty_windows;
	public static int sketch_size = 8;
	private double run_time;
	
	final BitSet marked_src;
	BitSet marked_susp;
	
	public OPH(int[] text, int sketch_size) {
		this(text, my_min_hasher.h(text,0,text.length), sketch_size);
	}
	public OPH(int[] text, long[] min_hashes, int sketch_size) {
		my_min_hashes = min_hashes;
		MinHash.num_oph_bins = sketch_size;
		my_oph_vector = MinHash.get_oph_vector(my_min_hashes);
		OPH.sketch_size = sketch_size;
		marked_src = new BitSet(text.length);
		
		empty_windows = CompactWindow.create_all_compact_window(text, my_min_hashes);
		non_empty_windows = NonEmptyCompactWindow.create_all_compact_window(text, my_min_hashes);
	}
	
	/**
	 * 
	 * @param query_sequence - usually the suspicious document
	 * @param threshold
	 * @param k window size
	 * @return one array of overlapping intervals per window
	 */
	public void query(int[] query_sequence, double threshold, int k) {
		System.out.println("OPH.query(int[],t="+threshold+",k="+k+")");
		this.marked_src.clear();
				
		double start = System.currentTimeMillis();
		
		long[] hashes = my_min_hasher.h(query_sequence,0,query_sequence.length);
		long[][] hashed_windows = create_hashed_windows(hashes, k);
		this.marked_susp = new BitSet(hashed_windows.length);
		
		for(int query_window=0;query_window<hashed_windows.length;query_window++) {
			boolean found_overlap = query(hashed_windows[query_window], threshold);
			if(found_overlap) {
				marked_susp.set(query_window, query_window+k);
			}
			if(query_window%300==0) {
				System.out.print("["+query_window+" of "+hashed_windows.length+"] ");
			}
		}
		System.out.println();
		
		double stop = System.currentTimeMillis();
		this.run_time = (stop-start);
		System.out.println("query(int[] query_sequence, double threshold, int k) done in "+(stop-start)+" ms");
	}
	
	/**
	 * 
	 * @param raw_paragraphs all the paragraphs
	 * @param k - window size
	 * @return
	 */
	private static long[][] create_hashed_windows(long[] min_hashes, final int k) {	
		long[][] windows; 
		if(min_hashes.length-k+1<0) {
			System.err.println("Solutions.create_windows(): raw_paragraph.length-k+1<0");
			windows = new long[1][];
			windows[0] = min_hashes.clone();
		}else{
			windows = new long[min_hashes.length-k+1][k];//pre-allocate the storage space for the
			for(int i=0;i<windows.length;i++){
				//create one window
				for(int j=0;j<k;j++) {
					windows[i][j] = min_hashes[i+j];
				}
			}
		}
		return windows;
	}
	
	public boolean query(int[] query_sequence, double threshold) {
		this.marked_src.clear();
		long[] hashes = my_min_hasher.h(query_sequence,0,query_sequence.length);
		return query(hashes, threshold);
	}
	boolean query(int[] query_sequence, int from, int to, double threshold) {
		this.marked_src.clear();
		long[] hashes = my_min_hasher.h(query_sequence, from, to);
		return query(hashes, threshold);
	}

	boolean query(long[] query_min_hashes, double threshold) {
		long[] oph_vector = MinHash.get_oph_vector(query_min_hashes);
		//get colliding empty compact windows
		ArrayList<CompactWindow> colliding_emtpy_windows = get_colliding_emtpy_windows(oph_vector,empty_windows);
		//get colliding non empty compact windows
		ArrayList<NonEmptyCompactWindow> colliding_non_emtpy_windows = get_non_colliding_non_emtpy_windows(oph_vector,non_empty_windows);
		
		return oph_interval_scan(this.sketch_size, threshold, colliding_non_emtpy_windows, colliding_emtpy_windows);
	}

	private ArrayList<NonEmptyCompactWindow> get_non_colliding_non_emtpy_windows(long[] oph_vector,
			ArrayList<ArrayList<NonEmptyCompactWindow>> windows) {
		ArrayList<NonEmptyCompactWindow> ret = new ArrayList<NonEmptyCompactWindow>();
		
		if(debug) {
			//XXX mimics Example 4
			ret.add(new NonEmptyCompactWindow(null, null, 1, 1, 3, 9));
			ret.add(new NonEmptyCompactWindow(null, null, 1, 4, 8, 13));
		}else{
			for(int bin=0;bin<oph_vector.length;bin++) {
				long bin_hash_value = oph_vector[bin];
				for(NonEmptyCompactWindow w : this.non_empty_windows.get(bin)) {
					long window_hash_value = this.my_min_hashes[w.index_token_min_hash];
					
					if(bin_hash_value==window_hash_value) {
						ret.add(w);
					}
				}
			}
		}

		return ret;
	}

	private ArrayList<CompactWindow> get_colliding_emtpy_windows(long[] oph_vector,
			ArrayList<ArrayList<CompactWindow>> windows) {
		ArrayList<CompactWindow> ret = new ArrayList<CompactWindow>();
		
		if(debug) {
			// XXX mimics Example 4
			ret.add(new CompactWindow(null, null, 2, 6, 10));
		}else{
			for(int bin=0;bin<oph_vector.length;bin++) {
				if(oph_vector[bin]==MinHash.EMPTY) {
					for(CompactWindow w : this.empty_windows.get(bin)) {
						ret.add(w);
					}
				}
			}
		}
		
		return ret;
	}	

	/**
	 * 
	 * @param k - sketch size, i.e., number of hash oph bin
	 * @param theta - Jaccard threshold in [0,1]
	 * @param C - the set of collided non-empty OPH compact windows
	 * @param C_e - the set of collided empty OPH compact windows
	 * @return true if there was at least one overlap. The overlaps itself are in <code>marked_src</code>
	 */
	private boolean oph_interval_scan(double k, double theta, ArrayList<NonEmptyCompactWindow> C, ArrayList<CompactWindow> C_e) {
		//System.out.println("oph_interval_scan() |C|="+C.size()+" |C_e|="+C_e.size());
		//double start = System.currentTimeMillis();
		//ArrayList<Integer> solution_intervals = new ArrayList<Integer>();//usually we do not need the intervals directly
		ArrayList<Endpoint> endpoints = new ArrayList<Endpoint>();
		
		final double LOWER_HALF_NON_EMPTY_WINODW = 1;
		final double UPPER_HALF_NON_EMPTY_WINODW = -1;
		final double LOWER_HALF_EMPTY_WINODW = theta;
		final double UPPER_HALF_EMPTY_WINODW = -theta;
		
		/**
		 * The threshold
		 */
		double k_theta = k*theta;
		
		if(((double)C.size())+k_theta*((double)C_e.size())<k_theta) {
			return false;
		}
		/**
		 * Kind of contains the number of open windows
		 */
		double cnt = 0;
		
		for(int d=0;d<C.size();d++) {
			NonEmptyCompactWindow w = C.get(d);
			endpoints.add(new Endpoint(w.l, LOWER_HALF_NON_EMPTY_WINODW, d));
			endpoints.add(new Endpoint(w.index_token_min_hash+1, UPPER_HALF_NON_EMPTY_WINODW, d));
		}
		
		for(int d=0;d<C_e.size();d++) {
			CompactWindow w = C_e.get(d);
			endpoints.add(new Endpoint(w.l, LOWER_HALF_EMPTY_WINODW, d));
			endpoints.add(new Endpoint(w.r+1, UPPER_HALF_EMPTY_WINODW, d));
		}
		
		Collections.sort(endpoints);
		
		
		//int[] all_u_x = get_all_distinct_values_sortetd(endpoints);
		//HashMap<Integer, ArrayList<Endpoint>> enpoints_grouped_by_u = get_enpoints_grouped_by_u(all_u_x, endpoints);//TODO optimize this such to avoid doubling this stuff
		
		//TODO use this
		HashMap<Integer, ArrayList<Endpoint>> enpoints_grouped_by_u = new HashMap<Integer, ArrayList<Endpoint>>(100);
		int[] all_u_x = get_enpoints_grouped_by_u(endpoints, enpoints_grouped_by_u);
		
		//HashMap<Integer,ArrayList<NonEmptyCompactWindow>> C_prime = new HashMap<Integer,ArrayList<NonEmptyCompactWindow>>(C.size());
		//HashMap<Integer,CompactWindow> C_e_prime = new HashMap<Integer,CompactWindow>(C_e.size());
		
		HashSet<Integer> C_prime = new HashSet<Integer>();
		HashSet<Integer> C_e_prime = new HashSet<Integer>();
		
		boolean found_overlap = false;
		
		for(int i=0;i<all_u_x.length;i++) {
			int u_x = all_u_x[i];
					
			for(Endpoint e : enpoints_grouped_by_u.get(u_x)) {//Lines 8-11
				/*if(e.u!=u_x) {
					System.err.println("e.u!=u_x");
				}*/
				
				cnt += e.w;
				if(e.w==LOWER_HALF_NON_EMPTY_WINODW) {
					C_prime.add(e.d);
				}else if(e.w==UPPER_HALF_NON_EMPTY_WINODW) {
					C_prime.remove(e.d);
				}else if(e.w==LOWER_HALF_EMPTY_WINODW) {
					C_e_prime.add(e.d);
				}else if(e.w==UPPER_HALF_EMPTY_WINODW) {
					C_e_prime.remove(e.d);
				}else{
					//Should never happen
				}
			}
			
			if(cnt>k_theta) {
				double cnt_prime = 0;
				ArrayList<Endpoint> endpoints_prime = new ArrayList<Endpoint>();
				
				for(int id : C_prime) {
					NonEmptyCompactWindow w = C.get(id);
					endpoints_prime.add(new Endpoint(w.index_token_min_hash, LOWER_HALF_NON_EMPTY_WINODW, id));
					endpoints_prime.add(new Endpoint(w.r+1, UPPER_HALF_NON_EMPTY_WINODW, id));
				}
				
				for(int id : C_e_prime) {
					CompactWindow w = C_e.get(id);
					endpoints_prime.add(new Endpoint(w.l, LOWER_HALF_EMPTY_WINODW, id));
					endpoints_prime.add(new Endpoint(w.r+1, UPPER_HALF_EMPTY_WINODW, id));
				}
				Collections.sort(endpoints_prime);
				
				//TODO optimize me
				//int[] all_v_x = get_all_distinct_values_sortetd(endpoints_prime);
				//HashMap<Integer, ArrayList<Endpoint>> enpoints_grouped_by_v_x = get_enpoints_grouped_by_u(all_v_x, endpoints_prime);
				HashMap<Integer, ArrayList<Endpoint>> enpoints_grouped_by_v_x = new HashMap<Integer, ArrayList<Endpoint>>(100);
				int[] all_v_x = get_enpoints_grouped_by_u(endpoints_prime, enpoints_grouped_by_v_x);
				
				//TODO early abort ???
				
				for(int j=0;j<all_v_x.length;j++) {
					int v_y = all_v_x[j];
					
					for(Endpoint e : enpoints_grouped_by_v_x.get(v_y)) {
						//if(e.u!=v_y) {
						//	System.err.println("e.u!=v_y");
						//}
						cnt_prime += e.w;
					}
					if(cnt_prime>=k_theta) {
						found_overlap = true;
						//This is a bit tricky. We a found overlap in two intervals, since we cut the compact window in half
						//solution_intervals.add(u_x);
						//int end = all_u_x[i+1]-1;
						//solution_intervals.add(end);
						marked_src.set(u_x, all_u_x[i+1]);//end exclusive
						//Second intervall
						//solution_intervals.add(v_y);
						//end = all_v_x[j+1]-1;
						//solution_intervals.add(end);
						marked_src.set(v_y, all_v_x[j+1]);//end exclusive
					}
				}
			}
		}
		//System.out.println("[DONE] in "+(System.currentTimeMillis()-start)+" ms found "+solution_intervals.size()+" intervals");
		return found_overlap;
	}
	
	private static int[] get_enpoints_grouped_by_u(ArrayList<Endpoint> endpoints, HashMap<Integer, ArrayList<Endpoint>> result) {
		HashSet<Integer> temp = new HashSet<Integer>();
		
		for(Endpoint e : endpoints) {
			int key = e.u;
			ArrayList<Endpoint> add_me_here = result.get(key);
			if(add_me_here==null) {
				add_me_here = new ArrayList<Endpoint>();
				result.put(key, add_me_here);
				temp.add(key);
			}
			add_me_here.add(e);
		}
		
		int[] all_u = new int[temp.size()];
		int i=0;
		for(int u : temp) {
			all_u[i++] = u;
		}
		Arrays.sort(all_u);
		return all_u;
	}

	@Deprecated
	private static HashMap<Integer, ArrayList<Endpoint>> get_enpoints_grouped_by_u(int[] all_u_x, ArrayList<Endpoint> endpoints) {
		HashMap<Integer, ArrayList<Endpoint>> result = new HashMap<Integer, ArrayList<Endpoint>>(all_u_x.length);
		
		//Create the entries
		for(int key : all_u_x) {
			result.put(key, new ArrayList<Endpoint>(100));
		}
		
		for(Endpoint e : endpoints) {
			int key = e.u;
			ArrayList<Endpoint> add_me_here = result.get(key);
			if(add_me_here==null) {
				System.err.println("add_me_here==null");
			}
			add_me_here.add(e);
		}
		
		return result;
	}

	@Deprecated
	private static int[] get_all_distinct_values_sortetd(ArrayList<Endpoint> endpoints) {
		HashSet<Integer> temp = new HashSet<Integer>();
		for(Endpoint e : endpoints) {
			temp.add(e.u);
		}
		int[] all_u = new int[temp.size()];
		int i=0;
		for(int u : temp) {
			all_u[i++] = u;
		}
		Arrays.sort(all_u);
		return all_u;
	}
	
	public static void main(String[] args) {
		OPH example_4 = new OPH(MinHash.T, 2);
		
		int[] query = {1,2,3,4,5};
		long[] hashes = {5,6,7,8};
		example_4.query(query, 0.8);
	}
	public Double get_runtime() {
		return run_time;
	}
	public BitSet marked_src() {
		return marked_src;
	}
	public BitSet marked_sup() {
		return this.marked_susp;
	}
	public int num_tokens() {
		return this.my_min_hashes.length;
	}
	public static boolean result_is_equal(BitSet result_org, BitSet result_seq) {
		System.out.println("result_is_equal(BitSet result_org, BitSet result_seq)");
		if(result_org.size()!=result_seq.size()) {
			System.err.println("result_org.size()!=result_seq.size()");
			return false;
		}
		for(int i=0;i<result_org.size();i++) {
			if(result_org.get(i)!=result_seq.get(i)) {
				System.err.println("result_org.get(i)!=result_seq.get(i)");
				System.out.println(result_org);
				System.out.println(result_seq);
				return false;
			}
		}
		System.out.println("result_is_equal(BitSet result_org, BitSet result_seq) return=true");
		return true;
	}

	public void query_exhaustive(int[] query_sequence, double threshold, int k) {
		System.out.println("OPH.query_exhaustive(int[],t="+threshold+",k="+k+")");
		this.marked_src.clear();
				
		double start = System.currentTimeMillis();
		
		long[] hashes = my_min_hasher.h(query_sequence,0,query_sequence.length);
		long[][] hashed_windows = create_hashed_windows(hashes, k);
		
		this.marked_susp = new BitSet(hashed_windows.length);
		
		for(int query_window=0;query_window<hashed_windows.length;query_window++) {
			boolean found_overlap = query_exhaustive(hashed_windows[query_window], threshold, k);
			if(found_overlap) {
				marked_susp.set(query_window, query_window+k);
			}
			if(query_window%300==0) {
				System.out.print("["+query_window+" of "+hashed_windows.length+"] ");
			}
		}
		System.out.println();
		
		double stop = System.currentTimeMillis();
		this.run_time = (stop-start);
		System.out.println("query_exhaustive(int[] query_sequence, double threshold, int k) done in "+(stop-start)+" ms");
	}
	
	/**
	 * 
	 * @param query_hashes
	 * @param threshold
	 * @param window_size
	 * @return
	 */
	private boolean query_exhaustive(final long[] query_window_hashes, final double threshold, final int window_size) {
		final long[] query_oph_vector = MinHash.get_oph_vector(query_window_hashes);
		long[] src_window_buffer = new long[window_size];
		boolean found_overlap = false;
		
		//for w = 0, create the first window of src
		int src_window=0;
		for(int i=0;i<window_size;i++) {
			src_window_buffer[i] = my_min_hashes[i];
		}
		long[] src_oph_vector = MinHash.get_oph_vector(src_window_buffer);//TODO buffer
		double sim = OPH.estimate_jaccard_sim(query_oph_vector, src_oph_vector);
		if(sim>threshold) {
			this.marked_src.set(src_window);
			found_overlap = true;
		}
		src_window++;
		
		for(;src_window<this.my_min_hashes.length-window_size+1;src_window++) {
			int offset = (src_window-1) % window_size;
			long new_value = my_min_hashes[src_window+window_size-1];
			src_window_buffer[offset] = new_value;
			MinHash.get_oph_vector(src_window_buffer);//TODO buffer
			sim = OPH.estimate_jaccard_sim(query_oph_vector, src_oph_vector);
			if(sim>threshold) {
				this.marked_src.set(src_window);
				found_overlap = true;
			}
		}
		
		return found_overlap;
	}
	
	private static double estimate_jaccard_sim(final long[] query_oph_vector, final long[] src_window_buffer) {
		/**
		 * Number of identical non empty OPH hash values
		 */
		double N_mat = 0;
		/**
		 * Number of identical empty OPH hash values
		 */
		double N_emp = 0;
		/**
		 * OPH sketch_size
		 */
		final double k = OPH.sketch_size;
		for(int i=0;i<query_oph_vector.length;i++) {
			if(query_oph_vector[i]==src_window_buffer[i]) {
				if(query_oph_vector[i]==MinHash.EMPTY) {
					N_emp++;
				}else{
					N_mat++;
				}
			}
		}
		
		return N_mat/(k-N_emp);//Eq. (3)
	}
}
