package bert;

public final class PairD implements Comparable<PairD>{
	final double d_1;
	final double d_2;
	public PairD(double first, double second) {
		this.d_1 = first;
		this.d_2 = second;
	} 
	@Override
	public int compareTo(PairD arg0) {
		int cmpr = Double.compare(d_1, arg0.d_1);
		return (cmpr!=0) ? cmpr : Double.compare(d_2, arg0.d_2);
	}
}
