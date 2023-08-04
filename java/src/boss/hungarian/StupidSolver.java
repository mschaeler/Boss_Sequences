package boss.hungarian;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;

public class StupidSolver extends Solver{
	int[][] permutations;
	int dim;
	
	public StupidSolver(int dim){
		//if(dim!=3 && dim!=4) System.err.println("dim!=3");
		this.dim = dim;
		this.permutations = compute_permutations(dim);
	}
	
	private static int[][] compute_permutations(int n) {
		int[] elements = new int[n];
		for(int i=0;i<n;i++) {elements[i]=i;}
		ArrayList<int[]> permutations = new ArrayList<int[]>();
		get_permutations(n,elements,permutations);
		
	    Collections.sort(permutations, new Comparator<int[]>() {
			@Override
			public int compare(int[] arg0, int[] arg1) {
				for(int i=0;i<arg0.length;i++) {
					if(arg0[i]!=arg1[i]) {
						return Integer.compare(arg0[i],arg1[i]);
					}
				}
				return 0;
			}
		});
	    int[][] result = new int[permutations.size()][];
	    int i=0;
	    for(int[] perm : permutations) {
	    	//System.out.println(Arrays.toString(perm));
	    	result[i++] = perm;
	    }
	    
		return result;
	}
	
	private static void get_permutations(int n, int[] elements, ArrayList<int[]> permutations) { 	
	    if(n == 1) {
	        int[] temp = elements.clone();
	        permutations.add(temp);
	    } else {
	        for(int i = 0; i < n-1; i++) {
	            get_permutations(n - 1, elements, permutations);
	            if(n % 2 == 0) {
	                swap(elements, i, n-1);
	            } else {
	                swap(elements, 0, n-1);
	            }
	        }
	        get_permutations(n - 1, elements, permutations);
	    }
	}
	private static void swap(int[] elements, int a, int b) {
	    int tmp = elements[a];
	    elements[a] = elements[b];
	    elements[b] = tmp;
	}
	
	private int permutation = -1;
	public int[] get_assignment(){
		return this.permutations[permutation];
	}
	
	@Override
	public double solve(double[][] cost_matix, final double threshold) {
		double min_costs = Double.MAX_VALUE;
		int permutation  = -1;
		for(int i=0;i<permutations.length;i++){
			int[] my_permutaiton = permutations[i];
			double cost = 0;
			
			for(int mapping=0;mapping<dim;mapping++) {
				int mapped_to = my_permutaiton[mapping];
				final double mapping_cost = cost_matix[mapping][mapped_to]; 
				cost+=mapping_cost;
			}
			if(cost<min_costs){
				min_costs = cost;
				permutation = i;
			}
		}
		this.permutation = permutation;//used to get the job assignment
		return min_costs;
	}
	
	public static void main(String[] args) {
		int n = 5;
		compute_permutations(n);
		
		double[][] costs= {{8, 5, 9}, {4, 2, 4}, {7, 3, 8}};
		StupidSolver hs = new StupidSolver(3);
		System.out.println(hs.solve(costs));
		
		double[][] costs2= {
				{1, 1, 1, 2}
				, {3, 2, 4, 1}
				, {4, 4, 2, 4}
				, {2, 3, 3, 3}
			};
		hs = new StupidSolver(4);
		System.out.println(hs.solve(costs2));
	}

	@Override
	public String get_name() {
		return this.getClass().getCanonicalName();
	}
}
