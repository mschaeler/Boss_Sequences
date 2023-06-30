package boss.hungarian;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class HungarianAlgorithmWiki {
	
	public static ArrayList<Integer> hungarian(final int[][] costs) {
	    //const int J = (int)size(C), W = (int)size(C[0]);
	    final int J = costs.length, W = costs[0].length;
	    assert(J <= W);
	    // job[w] = job assigned to w-th worker, or -1 if no job assigned
	    // note: a W-th worker was added for convenience
	    final int[] job = new int[W + 1];
	    Arrays.fill(job, -1);
	    
	    final int[] ys = new int[J];// potentials
	    final int[] yt = new int[W+1];
	    // -yt[W] will equal the sum of all deltas
	    ArrayList<Integer> answers = new ArrayList<Integer>(W+1);
	    final int inf = Integer.MAX_VALUE;
	    	    
	    for (int j_cur = 0; j_cur < J; ++j_cur) {  // assign j_cur-th job
	        int w_cur = W;
	        job[w_cur] = j_cur;
	        // min reduced cost over edges from Z to worker w
	        int[] min_to  = new int[W + 1];
	        Arrays.fill(min_to, inf);
	        int[] prv  = new int[W + 1];// previous worker on alternating path
	        Arrays.fill(prv, -1);
	        boolean[] in_Z  = new boolean[W + 1]; // whether worker is in Z
	        
	        while (job[w_cur] != -1) {   // runs at most j_cur + 1 times
	            in_Z[w_cur] = true;
	            final int j = job[w_cur];
	            int delta = inf;
	            int w_next = 0;//@CheckMe set manually
	            for (int w = 0; w < W; ++w) {
	                if (!in_Z[w]) {
	                	int temp;
	                	if((temp=costs[j][w] - ys[j] - yt[w])<min_to[w]) {
	                		min_to[w] = temp;
	                		prv[w] = w_cur;
	                	}
	                	if(min_to[w]<delta) {
	                		delta=min_to[w];
	                		w_next = w;
	                	}
	                }
	            }
	            // delta will always be non-negative,
	            // except possibly during the first time this loop runs
	            // if any entries of C[j_cur] are negative
	            for (int w = 0; w <= W; ++w) {
	                if (in_Z[w]) { 
	                	ys[job[w]] += delta;
	                	yt[w] -= delta;
	                }
	                else min_to[w] -= delta;
	            }
	            w_cur = w_next;
	        }
	        // update assignments along alternating path
	        for (int w; w_cur != -1; w_cur = w) {
	        	w = prv[w_cur];
	        	if(w==-1) {
	        		break;
	        	}
	        	job[w_cur] = job[w];
	        }
	        answers.add(-yt[W]);
	    }
	    return answers;
	}
	static void sanity_check_hungarian() {
	    int[][] costs= {{8, 5, 9}, {4, 2, 4}, {7, 3, 8}};
	    int[] result = {5, 9, 15};
	    
		int[][] costs2= {
				{1, 3, 4, 2}
				, {1, 2, 4, 3}
				, {1, 4, 2, 3}
				, {2, 1, 4, 3}
			};
	    ArrayList<Integer> temp = hungarian(costs);
	    System.out.println("Done "+temp);
	    
	    temp = hungarian(costs2);
	    System.out.println("Done "+temp);
	}
	
	public static void main(String[] args) {
		sanity_check_hungarian();
	}
}
