package txtalign;

public class Node {
	public int prev;
	public int next;

	public Node(int p, int n) {
		this.prev = p;
		this.next = n;
	}

	public Node()
    {
        prev = -999;
        next = -999;
    }
	public String toString() {
		return "prev="+prev+" next="+next;
	}
}
