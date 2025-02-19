package oph;

import java.util.ArrayList;

public class NonEmptyCompactWindow {
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
	 * index of the token defining the min hash of that bin, c in the paper
	 */
	final int index_token_min_hash;
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
	 * @param c - index of the token defining the min hash of that bin
	 * @param r - right border
	 */
	public NonEmptyCompactWindow(int[] T, long[] h, int bin, int l, int c, int r){
		this.sequence = T;
		this.hash_vector = h;
		this.bin = bin;
		this.l = l;
		this.index_token_min_hash = c;
		this.r = r;
		is_correct_window = check_me();
	}
	
	/**
	 * Implements the check from page 200:9 Section 3.2
	 */
	private boolean check_me() {
		boolean is_correct = true;
		//(1)
		if(l<0) {
			System.err.println("l<0 "+this);
			is_correct = false;
		}
		if(index_token_min_hash<l) {
			System.err.println("r<l "+this);
			is_correct = false;
		}
		if(r<index_token_min_hash) {
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
		if(MinHash.get_bin(hash_vector[index_token_min_hash])!=bin) {
			System.err.println("MinHash.get_bin(hash_vector[index_token_min_hash])!=bin");
			is_correct = false;
		}
		//(3)
		for(int i=l;i<=r;i++) {//r included
			if(i==index_token_min_hash) {
				continue;
			}
			int my_bin = MinHash.get_bin(hash_vector[i]);
			if(my_bin == bin) {
				if(hash_vector[i]<=hash_vector[index_token_min_hash]) {//XXX The paper says <= though < would make more sense
					System.err.println("(3) hash_vector[i]<=hash_vector[index_token_min_hash] " + this);
					is_correct = false;
				}//else it is k
				
			}
		}
		//(4)
		if(l!=0) {//left border of the window != sequence start
			int my_bin = MinHash.get_bin(hash_vector[l-1]);
			if(my_bin!=bin) {
				System.err.println("(4) my_bin!=bin "+this);
				is_correct = false;
			}else{
				if(!(hash_vector[l-1]<hash_vector[index_token_min_hash])) {
					System.err.println("(4) !(hash_vector[l-1]<hash_vector[index_token_min_hash]) "+this);
					is_correct = false;
				}
			}
		}
		//(5)
		if(r!=sequence.length-1) {//right border of the window != end of sequence
			int my_bin = MinHash.get_bin(hash_vector[r+1]);
			if(my_bin!=bin) {
				System.err.println("(5) my_bin!=bin "+this);
				is_correct = false;
			}else{
				if(!(hash_vector[r+1]<hash_vector[index_token_min_hash])) {
					System.err.println("(5) !(hash_vector[r+1]<hash_vector[index_token_min_hash]) "+this);
					is_correct = false;
				}
			}
		}
		
		if(sequence.length!=hash_vector.length){
			System.err.println("sequence.length!=hash_vector.length");
			is_correct = false;
		}
		return is_correct;
	}
	
	public static ArrayList<NonEmptyCompactWindow> create_windows(int[] sequence, long[] hashes, int bin) {
		ArrayList<Integer> bin_token_position = new ArrayList<Integer>(hashes.length);
		for(int i=0;i<hashes.length;i++) {
			if(MinHash.get_bin(hashes[i])==bin) {
				bin_token_position.add(i);
			}
		}
		System.out.println("Token position with hash value in bin="+bin+" :"+bin_token_position);
		
		ArrayList<Integer> windows = new ArrayList<Integer>(bin_token_position.size()*3);//index, start and end sequence the token dominates
		for(int i=0;i<bin_token_position.size();i++) {
			int token_position = bin_token_position.get(i);
			windows.add(token_position);
			long my_hash = hashes[token_position];
			
			int from = token_position;
			int to = token_position;
			{
				{
					//look for token to the left having a lower hash
					if(i==0) {
						from = 0;//Is first element
					}else{
						int down=i-1;
						for(;down>=0;down--) {
							int other_position =  bin_token_position.get(down);
							if(hashes[other_position]<my_hash) {
								from = other_position+1;
								break;
							}
						}
						if(down<0) {
							from = 0;//from the start
						}
					}
				}
				{
					//look for token to the right having a lower hash
					if(i==bin_token_position.size()-1) {
						to = hashes.length-1;//until end
					}else{
						int up=i+1;
						for(;up<bin_token_position.size();up++) {
							int other_position =  bin_token_position.get(up);
							if(hashes[other_position]<my_hash) {
								to = other_position-1;
								break;
							}
						}
						if(up==bin_token_position.size()) {
							to = hashes.length-1;//until end
						}
					}
				}
				
			}
			windows.add(from);
			windows.add(to);
		}
		
		ArrayList<NonEmptyCompactWindow> ret = new ArrayList<NonEmptyCompactWindow>(windows.size()/3);
		for(int w=0;w<windows.size();w+=3) {//[idx,from,to]. Thus +3.
			NonEmptyCompactWindow temp = new NonEmptyCompactWindow(sequence, hashes, bin, windows.get(w+1), windows.get(w), windows.get(w+2));
			ret.add(temp);
		}
		
		return ret;
	}
	
	public static ArrayList<ArrayList<NonEmptyCompactWindow>> create_all_compact_window(int[] sequence, long[] min_hashes) {
		System.out.println("Create all Non-Empty compact windows");
		ArrayList<ArrayList<NonEmptyCompactWindow>> all_empty_windows = new ArrayList<ArrayList<NonEmptyCompactWindow>>(MinHash.num_oph_bins);
		for(int bin=0;bin<MinHash.num_oph_bins;bin++) {
			ArrayList<NonEmptyCompactWindow> temp = create_windows(sequence, min_hashes, bin);
			for(NonEmptyCompactWindow w : temp) {
				System.out.println(w);
			}
			
			all_empty_windows.add(temp);
		}
		return all_empty_windows;
	}
	
	public String toString() {
		return "(bin="+bin+",l="+l+",c="+index_token_min_hash+",r="+r+")";
	}
}
