package pan;

public class Detection {
	public String susp;
	public String src;
	
	public int this_offset;
	public int this_length;
	
	public int source_offset;
	public int source_length;
	
	public int k = -1;
	public double theta = -1;
	
	public Detection() {
		
	}
	
	public Detection(String susp, String src, int this_offset, int this_length, int source_offset, int source_length) {
		this.susp = susp;
		this.src = src;
		
		this.this_offset = this_offset;
		this.this_length = this_length;
		
		this.source_offset = source_offset;
		this.source_length = source_length;
	}
	
	public String toString() {
		return "susp="+susp+" ["+this_offset+", "+this_length+"] src="+src+" ["+source_offset+", "+source_length+"]" ;
	}
}
