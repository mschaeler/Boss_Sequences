package pan;

public class Detection {
	final int susp_start;
	final int susp_stop;
	final int src_start;
	final int src_stop;
	
	/**
	 * Case Constructor
	 * @param array
	 */
	public Detection(int[] array) {
		susp_start = array[0];//Add the window length s.t. the last token in the window is also considered
		susp_stop  = array[1];
		src_start  = array[2];
		src_stop   = array[3];
	}
	public Detection(int line, int column, int k) {
		susp_start = line;//Add the window length s.t. the last token in the window is also considered
		susp_stop  = line+k-1;
		src_start  = column+k-1;
		src_stop   = column+k-1;
	}
	
	public boolean overlaps(Detection d) {
		if(overlap(this.susp_start, this.susp_stop, d.susp_start, d.susp_stop)==0) {
			return false;
		}
		if(overlap(this.src_start, this.src_stop, d.src_start, d.src_stop)==0) {
			return false;
		}
		return true;
	}
	public int overlap(Detection d) {
		if(!overlaps(d)) {
			return 0;
		}
		int overlap = overlap(this.susp_start, this.susp_stop, d.susp_start, d.susp_stop);
		overlap += overlap(this.src_start, this.src_stop, d.src_start, d.src_stop);
		return overlap;
	}

	private int overlap(int start_1, int stop_1, int start_2, int stop_2) {
		int min_start = Math.min(start_1, start_2);
		int max_stop  = Math.max(stop_1, stop_2);
		int overlap = ((stop_1 - start_1 + 1) + (stop_2 - start_2 + 1)) - (max_stop - min_start +1);//individual lengths - max length
		if(overlap<0) {
			overlap = 0;
		}
		return overlap;
	}
	public String toString() {
		return "["+susp_start+" "+susp_stop+"] ["+src_start+" "+src_stop+"]";
	}
}
