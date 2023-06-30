package boss.hungarian;

import java.util.ArrayList;
import java.util.Arrays;

public class StupidSolver extends Solver{
	int[][] permutations;
	int dim;
	
	StupidSolver(int dim){
		if(dim!=3 && dim!=4) System.err.println("dim!=3");
		this.dim = dim;
		this.permutations = get_permutations(dim);
	}
	
	static final int[][] get_permutations(int dim){
		if(dim==3) {
			int[][] ret = {
					{0,1,2}
					,{0,2,1}
					,{1,0,2}
					,{1,2,0}
					,{2,0,1}
					,{2,1,0}
			};
			return ret;
		}
		if(dim==4) {
			int[][] ret = {
					{0,1,2,3}
					,{0,1,3,2}
					,{0,2,1,3}
					,{0,2,3,1}
					,{0,3,1,2}
					,{0,3,2,1}
					
					,{1,0,2,3}
					,{1,0,3,2}
					,{1,2,0,3}
					,{1,2,3,0}
					,{1,3,0,2}
					,{1,3,2,0}
					
					,{2,0,1,3}
					,{2,0,2,3}
					,{2,1,0,3}
					,{2,1,3,0}
					,{2,3,0,1}
					,{2,3,1,0}
					
					,{3,0,1,2}
					,{3,0,2,1}
					,{3,1,0,2}
					,{3,1,2,0}
					,{3,2,0,1}
					,{3,2,1,0}
			};
			return ret;
		}
		return null;
	}
	
	public static void printAllOrdered(int[] elements) {
			    Arrays.sort(elements);
			    boolean hasNext = true;

			    while(hasNext) {
			        System.out.println(Arrays.toString(elements));
			        int k = 0, l = 0;
			        hasNext = false;
			        for (int i = elements.length - 1; i > 0; i--) {
			            if (elements[i]>elements[i - 1]) {
			                k = i - 1;
			                hasNext = true;
			                break;
			            }
			        }

			        for (int i = elements.length - 1; i > k; i--) {
			            if (elements[i]>elements[k]) {
			                l = i;
			                break;
			            }
			        }

			        swap(elements, k, l);
			        java.util.Collections.reverse(Arrays.asList(elements).subList(k + 1, elements.length));
			    }
			}
	
	public static void printAllRecursive(int n, int[] elements) { 
		ArrayList<int[]> permutations = new ArrayList<int[]>();
	    if(n == 1) {
	        int[] temp = elements.clone();
	        permutations.add(temp);
	    } else {
	        for(int i = 0; i < n-1; i++) {
	            printAllRecursive(n - 1, elements);
	            if(n % 2 == 0) {
	                swap(elements, i, n-1);
	            } else {
	                swap(elements, 0, n-1);
	            }
	        }
	        printAllRecursive(n - 1, elements);
	    }
	    for(int[] perm : permutations) {
	    	System.out.println(Arrays.toString(perm));
	    }
	}
	private static void swap(int[] elements, int a, int b) {
	    int tmp = elements[a];
	    elements[a] = elements[b];
	    elements[b] = tmp;
	}
	
	int permutation = -1;
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
				if(mapping_cost<threshold) {
					cost+= mapping_cost;
				}else{
					cost+= HungarianExperiment.MAX_DIST;
				}
				
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
		int[] numbers = {0,1,2};
		//printAllOrdered(numbers);
		printAllRecursive(3, numbers);
		
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
}
