package boss.hungarian;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class HungarianAlgorithmWiki extends Solver{
	final int k;
	/**
	 * potentials
	 */
    final double[] ys; 
    /**
     * -yt[W] will equal the sum of all deltas
     */
    final double[] yt;
    /**
     * job[w] = job assigned to w-th worker, or -1 if no job assigned. Note: a W-th worker was added for convenience
     */
    final int[] job;
    /**
     * min reduced cost over edges from Z to worker w
     */
    final double[] min_to;
    /**
     * previous worker on alternating path
     */
    final int[] prv;
    
	public HungarianAlgorithmWiki(final int k) {
		this.k = k;
		ys = new double[k];// potentials
	    yt = new double[k+1];
	    job = new int[k + 1];
	    min_to  = new double[k + 1];
	    prv  = new int[k + 1];// previous worker on alternating path
	}
	
	@Override
	public double solve(final double[][] costs, final double threshold) {
		final double cost_threshold = (double)k-(threshold*(double)k);
	    final int J = k, W = k;
	    Arrays.fill(job, -1);
	    //ArrayList<Double> answers = new ArrayList<Double>(W+1);
	    final double inf = Double.MAX_VALUE;
	    	    
	    for (int j_cur = 0; j_cur < J; ++j_cur) {  // assign j_cur-th job
	        int w_cur  = W;
	        job[w_cur] = j_cur;
	        
	        Arrays.fill(min_to, inf);
	        Arrays.fill(prv, -1);
	        boolean[] in_Z  = new boolean[W + 1]; // whether worker is in Z
	        
	        while (job[w_cur] != -1) {   // runs at most j_cur + 1 times
	            in_Z[w_cur] = true;
	            final int j = job[w_cur];
	            double delta = inf;
	            int w_next = 0;//@CheckMe set manually
	            for (int w = 0; w < W; ++w) {
	                if (!in_Z[w]) {
	                	double temp;
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
	        	if(w==-1) {//Error in Wiki code, works in CPP but results in memory access before job array
	        		break;
	        	}
	        	job[w_cur] = job[w];
	        }
	        /*
	        if(W<k && -yt[W]>=cost_threshold){//early termination
	        	return -yt[W];
	        }*/
	        //answers.add(-yt[W]);
	    }
	    return -yt[W];
	}
	@Override
	public String get_name() {
		return "HungarianAlgorithmWiki";
	}
}
