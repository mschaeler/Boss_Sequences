package txtalign;

import java.util.ArrayList;

public class CompositeWindow {
    public ArrayList<Integer> sketch = new ArrayList<Integer>(); // stores k hash values
    public ArrayList<Integer> positions = new ArrayList<Integer>(); //stores k positions to the hash values
    public Range beg_range = new Range();
    public Range end_range = new Range();
    
    public CompositeWindow() {}
    public CompositeWindow(int ll, int lr, int rl, int rr) {
        beg_range.l = ll;
        beg_range.r = lr;
        end_range.l = rl;
        end_range.r = rr;
    }
}
