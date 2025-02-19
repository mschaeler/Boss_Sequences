package oph;

import java.util.ArrayList;
import java.util.Arrays;
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
	final int sketch_size;
	
	public OPH(int[] text, int sketch_size) {
		this(text, my_min_hasher.h(text,0,text.length), sketch_size);
	}
	public OPH(int[] text, long[] min_hashes, int sketch_size) {
		my_min_hashes = min_hashes;
		MinHash.num_oph_bins = sketch_size;
		my_oph_vector = MinHash.get_oph_vector(my_min_hashes);
		this.sketch_size = sketch_size;
		
		empty_windows = CompactWindow.create_all_compact_window(text, my_min_hashes);
		non_empty_windows = NonEmptyCompactWindow.create_all_compact_window(text, my_min_hashes);
	}
	
	public ArrayList<Integer> query(int[] query_sequence, double threshold) {
		long[] hashes = my_min_hasher.h(query_sequence,0,query_sequence.length);
		return query(hashes, threshold);
	}
	public ArrayList<Integer> query(int[] query_sequence, int from, int to, double threshold) {
		long[] hashes = my_min_hasher.h(query_sequence, from, to);
		return query(hashes, threshold);
	}

	ArrayList<Integer> query(long[] query_min_hashes, double threshold) {
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
	 * @return
	 */
	private static ArrayList<Integer> oph_interval_scan(double k, double theta, ArrayList<NonEmptyCompactWindow> C, ArrayList<CompactWindow> C_e) {
		ArrayList<Integer> solution_intervals = new ArrayList<Integer>();
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
			return solution_intervals;
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
		
		
		int[] all_u_x = get_all_distinct_values_sortetd(endpoints);
		HashMap<Integer, ArrayList<Endpoint>> enpoints_grouped_by_u = get_enpoints_grouped_by_u(all_u_x, endpoints);
		
		//HashMap<Integer,ArrayList<NonEmptyCompactWindow>> C_prime = new HashMap<Integer,ArrayList<NonEmptyCompactWindow>>(C.size());
		//HashMap<Integer,CompactWindow> C_e_prime = new HashMap<Integer,CompactWindow>(C_e.size());
		
		HashSet<Integer> C_prime = new HashSet<Integer>();
		HashSet<Integer> C_e_prime = new HashSet<Integer>();
		
		
		for(int i=0;i<all_u_x.length;i++) {
			int u_x = all_u_x[i];
			
			
			for(Endpoint e : enpoints_grouped_by_u.get(u_x)) {//Lines 8-11
				if(e.u!=u_x) {
					System.err.println("e.u!=u_x");
				}
				
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
				
				int[] all_v_x = get_all_distinct_values_sortetd(endpoints_prime);
				HashMap<Integer, ArrayList<Endpoint>> enpoints_grouped_by_v_x = get_enpoints_grouped_by_u(all_v_x, endpoints_prime);
				
				for(int j=0;j<all_v_x.length;j++) {
					int v_y = all_v_x[j];
					
					for(Endpoint e : enpoints_grouped_by_v_x.get(v_y)) {
						if(e.u!=v_y) {
							System.err.println("e.u!=v_y");
						}
						cnt_prime += e.w;
					}
					if(cnt_prime>=k_theta) {
						//This is a bit tricky. We a found overlap in two intervals, since we cut the compact window in half
						solution_intervals.add(u_x);
						int end = all_u_x[i+1]-1;
						solution_intervals.add(end);
						//Second intervall
						solution_intervals.add(v_y);
						end = all_v_x[j+1]-1;
						solution_intervals.add(end);
					}
				}
			}
		}

		return solution_intervals;
	}
	
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
}
