package txtalign;

public class windowPair {
	  public int docOffset;
	  public int docLen;
	  public int queryOffset;
	  public int queryLen;
	  
	  public windowPair(int docOffset, int docLen, int queryOffset, int queryLen) {
	    this.docOffset = docOffset;
	    this.docLen = docLen;
	    this.queryOffset = queryOffset;
	    this.queryLen = queryLen;
	  }
}
