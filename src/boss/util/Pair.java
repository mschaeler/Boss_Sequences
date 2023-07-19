package boss.util;

public class Pair implements Comparable<Pair>{
	final int key;
	public final double similarity;
	
	public Pair(final int key, final double similarity) {
		this.key = key;
		this.similarity = similarity;
	}
	
	public String toString() {
		return key+"\t"+similarity;
	}

	@Override
	public int compareTo(Pair arg0) {
		int ret = Double.compare(similarity, arg0.similarity);
		if(ret==0) {//they are equal, sort by id
			return Integer.compare(key, arg0.key);
		}
		return ret;
	}
}
