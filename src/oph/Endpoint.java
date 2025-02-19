package oph;

public class Endpoint implements Comparable<Endpoint>{	
	/**
	 * Token position
	 */
	final int u;
	/**
	 * Flag: 1 lower half of non-empty window, -1 upper half. The same for empty windows: -theta and theta respectively 
	 */
	final double  w;
	/**
	 * No idea what this is
	 */
	final int d;
	
	/**
	 * TODO
	 * @param u
	 * @param w
	 * @param d
	 */
	public Endpoint(int u, double w, int d) {
		this.u = u;
		this.w = w;
		this.d = d;
	}
	
	@Override
	public int compareTo(Endpoint arg0) {
		// TODO Auto-generated method stub
		return Integer.compare(this.u, arg0.u);
	}

	public String toString(){
		return "(u="+u+" w="+w+" d="+d+")";
	}
}
