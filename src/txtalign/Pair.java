package txtalign;

public class Pair implements Comparable<Pair>{
	int first;
	int second;
	
	public Pair(int i, int j) {
		this.first = i;
		this.second = j;
	}

	@Override
	public int compareTo(Pair arg0) {
		if(this.first != arg0.first) {
			return Integer.compare(first, arg0.first);
		}else {
			return Integer.compare(second, arg0.second);
		}
	}
}


