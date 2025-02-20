package oph;

import java.util.ArrayList;
import java.util.BitSet;

public class CompactWindow {
	/**
	 * The original Text as sequence of integers. T in the paper.
	 */
	final int[] sequence;
	/**
	 * The min hash values for all tokens in T. h in the paper. 
	 */
	final long[] hash_vector;
	/**
	 * The bin (i.e., partition) of the OPH. t in the paper. 
	 */
	final int bin;
	/**
	 * Left border of the window
	 */
	final int l;
	/**
	 * Right border of the window.
	 */
	final int r;
	
	boolean is_correct_window = true;
	
	/**
	 * Constructor for Empty Compact Windows
	 * @param T original sequence, i.e., the Text
	 * @param h - hash function results
	 * @param bin - the bin (i.e., partition) of the OPH. t in he paper
	 * @param l - left border
	 * @param r - right border
	 */
	public CompactWindow(int[] T, long[] h, int bin, int l, int r){
		this.sequence = T;
		this.hash_vector = h;
		this.bin = bin;
		this.l = l;
		this.r = r;
		is_correct_window = check_me();
	}

	/**
	 * Implements the check from page 200:6 Section 3.1
	 */
	private boolean check_me() {
		boolean is_correct = true;
		//(1)
		if(l<0) {
			System.err.println("l<0 "+this);
			is_correct = false;
		}
		if(r<l) {
			System.err.println("r<l "+this);
			is_correct = false;
		}
		if(r>=sequence.length){
			System.err.println("r>=T.length");
			is_correct = false;
		}
		//(2)
		if(bin<0) {
			System.err.println("t<0");
			is_correct = false;
		}
		if(bin>=MinHash.num_oph_bins) {
			System.err.println("t>=MinHash.num_oph_bins");
			is_correct = false;
		}
		//(3)
		for(int i=l;i<=r;i++) {//r included
			int my_bin = MinHash.get_bin(hash_vector[i]);
			if(my_bin == bin) {
				System.err.println("(3) my_bin == bin " + this);
				is_correct = false;
			}
		}
		//(4)
		if(l!=0) {//left border of the window != sequence start
			int my_bin = MinHash.get_bin(hash_vector[l-1]);
			if(my_bin!=bin) {
				System.err.println("(4) my_bin!=bin "+this);
				is_correct = false;
			}
		}
		//(5)
		if(r!=sequence.length-1) {//right border of the window != end of sequence
			int my_bin = MinHash.get_bin(hash_vector[r+1]);
			if(my_bin!=bin) {
				System.err.println("(5) my_bin!=bin "+this);
				is_correct = false;
			}
		}
		
		if(sequence.length!=hash_vector.length){
			System.err.println("sequence.length!=hash_vector.length");
			is_correct = false;
		}
		return is_correct;
	}
	
	public static ArrayList<ArrayList<CompactWindow>> create_all_compact_window(int[] sequence, long[] min_hashes) {
		//System.out.println("Create all Empty compact windows");
		ArrayList<ArrayList<CompactWindow>> all_empty_windows = new ArrayList<ArrayList<CompactWindow>>(MinHash.num_oph_bins);
		for(int bin=0;bin<MinHash.num_oph_bins;bin++) {
			ArrayList<CompactWindow> temp = create_all_compact_window_bin_alternate(sequence, min_hashes, bin);
			/*for(CompactWindow w : temp) {
				System.out.println(w);
			}*/
			all_empty_windows.add(temp);
		}
		return all_empty_windows;
	}
	
	public static ArrayList<CompactWindow> create_all_compact_window_bin_alternate(int[] sequence, long[] hashes, int bin) {
		BitSet is_empty = new BitSet(hashes.length);
		for(int i=0;i<hashes.length;i++) {
			if(MinHash.get_bin(hashes[i])!=bin) {
				is_empty.set(i);
			}
		}
		//System.out.println(is_empty);
		
		ArrayList<Integer> candidates_condensed = new ArrayList<Integer>(is_empty.size());
		//int q = 0;
		//boolean found_run = false;
		int start_alt=0, stop_alt=0;
		
		while((start_alt = is_empty.nextSetBit(start_alt))!=-1) {
			stop_alt = is_empty.nextClearBit(start_alt);
			candidates_condensed.add(start_alt);
			candidates_condensed.add(stop_alt-1);
			start_alt = stop_alt;
		}
		
		ArrayList<CompactWindow> ret = new ArrayList<CompactWindow>(candidates_condensed.size()/2);
		for(int w=0;w<candidates_condensed.size();w+=2) {//[from,to]. Thus +2.
			CompactWindow temp = new CompactWindow(sequence, hashes, bin, candidates_condensed.get(w), candidates_condensed.get(w+1));
			ret.add(temp);
		}
		
		return ret;
	}
	
	@Deprecated
	public static ArrayList<CompactWindow> create_all_compact_window_bin(int[] sequence, long[] hashes, int bin) {
		ArrayList<CompactWindow> result = new ArrayList<CompactWindow>();
		
		int from = 0;
		//int to = hashes.length-1;
		int i = 1;
		
		while(i<hashes.length) {
			if(MinHash.get_bin(hashes[i])!=bin) {
				//do nothing, expected path
			}else{
				CompactWindow temp = create_compact_window(sequence, hashes, bin, from, i);
				if(temp!=null) {
					if(!temp.has_errors())
					result.add(temp);	
				}
				
				from = i;
			}
			i++;
		}
		/** Wäre logisch, steht so aber nicht da
		if(MinHash.get_bin(hashes[hashes.length-1])!=bin) {//we need to create the last window manually
			OPH_CompactWindow temp = create_compact_window(sequence, hashes, bin, from, hashes.length-1);
			if(temp!=null) {
				if(!temp.has_errors())
				result.add(temp);	
			}
		}*/
		return result;
	}
	
	
	public boolean has_errors() {
		return !is_correct_window;
	}

	public static CompactWindow create_compact_window(int[] sequence, long[] hashes, int bin,  int from, int to) {
		int my_bin_from = MinHash.get_bin(hashes[from]);
		int my_bin_to = MinHash.get_bin(hashes[to]);
		if(from!=0 && my_bin_from!=bin){
			System.err.println("create_compact_window() my_bin_from!=bin");
			return null;
		}
		if(my_bin_to!=bin){
			System.err.println("create_compact_window() my_bin_from!=my_bin_to");
			return null;
		}
		int start_w = from+1;//one to the left
		if(from==0 && my_bin_from != bin) {
			start_w = from;
		}
		int stop_w = to-1;
		if(to==hashes.length-1) {//final window until end
			stop_w = to;
		}
		if(from>=to) {
			System.err.println("No window for ["+from+","+to+"] bin="+bin);//Error in Definition?
			return null;
		}
		if(start_w>stop_w) {
			System.err.println("No window for ["+from+","+to+"] bin="+bin);
			return null;
		}
		
		return new CompactWindow(sequence, hashes, bin, start_w, stop_w);
	}
	
	public String toString() {
		return "(bin="+bin+",l="+l+",r="+r+")";
	}
}
