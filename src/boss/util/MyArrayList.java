package boss.util;


import java.util.Arrays;

//import elf.Elf;

public final class MyArrayList{
	public int[] ARRAY;
	static final int DEFAULT_INITIAL_SIZE = 16;//guessed
	private static final int EQUAL = 0;
	int capacity;
	public int writeHere = 0;
	
	public MyArrayList(){
		this(DEFAULT_INITIAL_SIZE);
	}
	
	public MyArrayList(int initialSize){
		ARRAY = new int[initialSize];
		capacity = initialSize;
	}
	public MyArrayList(int[] array){
		ARRAY = array;
		capacity = array.length;
		writeHere = capacity;
	}
	/**
	 * Copy constructor.
	 * 
	 * @param to_copy
	 */
	public MyArrayList(MyArrayList to_copy){
		ARRAY = new int[to_copy.ARRAY.length];
		System.arraycopy(to_copy.ARRAY, 0, ARRAY, 0, to_copy.ARRAY.length);
		capacity = to_copy.capacity;
		writeHere = to_copy.writeHere;
	}
	
	public void add(int element){
		/*if(element==16680) {
			System.err.println("add("+element+")");
		}*/
		if(writeHere==capacity){//esure capacity if full
			int[] newArray = new int[2*capacity];
			System.arraycopy(ARRAY, 0, newArray, 0, capacity);
			capacity *= 2;
			this.ARRAY = newArray;
		}
		ARRAY[writeHere++] = element;
	}
	public int get(int index){
		return ARRAY[index];
	}
	public int size(){
		return writeHere;
	}

	/** Virtual delete, i.e., does not free the associated memory.*/
	public void clear() {
		writeHere = 0;
	}

	public int[] toSortedArray() {
		int[] myArray = new int[this.size()];
		System.arraycopy(ARRAY, 0, myArray, 0, size());
		Arrays.sort(myArray);
		return myArray;
	}

	public void ensureCapacity(final int requiredCapacity) {
		if(requiredCapacity>this.capacity){
			int[] newArray = new int[2*requiredCapacity];
			System.arraycopy(ARRAY, 0, newArray, 0, capacity);
			capacity *= 2;
			this.ARRAY = newArray;
		}
	}

	public void sort(){
		if(writeHere==0){//empty
			return;
		}
		Arrays.sort(ARRAY,0,writeHere);
	}
	
	public void addAll(final int[] elf, final int start, final int stop) {
		final int length = stop-start;
		ensureCapacity(length);
		System.arraycopy(elf, start, this.ARRAY, 0, length);
		this.writeHere = length;
	}
	
	/**
	 * returns first difference
	 * @param otherList
	 * @return
	 */
	int compare(MyArrayList otherList){
		int[] l1 = this.toSortedArray();
		int[] l2 = otherList.toSortedArray();
		
		int minLength = Math.min(l1.length, l2.length);
		for(int i=0;i<minLength;i++){
			if(l1[i]!=l2[i]){
				System.out.println("First differnce @i="+i+" l1_i="+l1[i]+" l2_i="+l2[i]);
				return i;
			}
		}
		return EQUAL;
	}
	
	public void add(final MyArrayList toAdd){
		final int NEW_SIZE = this.size()+toAdd.size();
		ensureCapacity(NEW_SIZE);
		int[] from = toAdd.ARRAY;
		int[] to = this.ARRAY;
		System.arraycopy(from, 0, to, this.writeHere, toAdd.size());
		this.writeHere = NEW_SIZE;
	}
	
	public String toString(){
		return "s="+size()+" "+toString(size());
	}

	public String toString(int elmsToDisplay){
		String ret = "";
		for(int elem=0;elem<elmsToDisplay;elem++){
			ret+=" "+this.ARRAY[elem]+",";
		}
		return ret;
	}
	/*public boolean contains(final int value) {
		final int offset = Arrays.binarySearch(ARRAY, 0, writeHere, value);
		if(offset >= 0 && offset <writeHere){
			return true;
		}
		return false;
	}*/
	
	public void intersect(MyArrayList toIntersect){
		final int cand_length 		= this.size();
		final int to_merge_length 	= toIntersect.size();
		final int[] candidates 	= this.ARRAY;
		final int[] toMerge 	= toIntersect.ARRAY;
		
		this.clear();//the clear is only virtual, i.e., data is still ther, but writepointer is at front and can use it to write elements that are in both list at the currently next position. Recap, there are at most this.size() elems.
		
		int i = 0, j = 0;
		while (i < cand_length && j < to_merge_length){
			if(candidates[i] < toMerge[j]){//elem in candidates not found
				i++;
			}else if (toMerge[j] < candidates[i]){
				j++;
			}else{// if arr1[i] == arr2[j]
				candidates[writeHere++] = candidates[i];//this element remains
				j++;
				i++;
			}
		}
	}

	/** Physical delete, i.e., does free the associated memory.*/
	public void delete() {
		this.writeHere = -1;
		this.capacity = -1;
		this.ARRAY = null;//dear garbage collector, please really free the memory ...
	}
/*
	public static MyArrayList copy(MyArrayList synopsis) {
		if(TPC_H.out_operator){TPC_H.start = System.currentTimeMillis();}
		final int size = synopsis.size();
		final int[] array = new int[size];
		System.arraycopy(synopsis.ARRAY, 0, array, 0, size);
		if(TPC_H.out_operator){TPC_H.out_operator("copy synopsis",TPC_H.start);}
		if(TPC_H.out_cost){TPC_H.out_operator("copy synopsis",size,size);}
		return new MyArrayList(array);
	}
*/
	/**
	 * Returns a copy of the underlying array with its exact size
	 * @return
	 */
	public int[] getTrimmedArray() {
		int[] new_array = new int[this.writeHere];
		System.arraycopy(ARRAY, 0, new_array, 0, writeHere);
		return new_array;
	}
	
	/**
	 * O(n)
	 * @param value
	 * @return
	 */
	public final boolean isIn(final int value) {
		final int size = size();
		final int[] raw_array = this.ARRAY;
		for(int i=0;i<size;i++) {
			if(raw_array[i]==value) {
				return true;
			}
		}
		return false;
	}

	/**
	 * 
	 * @return the last inserted element or Integer.MIN_VALUE if empty
	 */
	public int last() {
		return (size()>0) ? this.ARRAY[writeHere-1] : Integer.MIN_VALUE;
	}

	public void override_last(int value) {
		this.ARRAY[writeHere-1] = value;
	}
}
