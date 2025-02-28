package boss.util;

public class Plotter {
	
static double[] thresholds = {0.6,	0.7,	0.73,	0.8,	0.9};

static double[][] values = {
{3,	0.11,	0.14,	0.17,	0.25,	0.50},
{4,	0.12,	0.20,	0.27,	0.40,	0.57},
{5,	0.13,	0.27,	0.40,	0.61,	0.59},
{6,	0.14,	0.39,	0.53,	0.72,	0.58},
{7,	0.15,	0.53,	0.66,	0.73,	0.58},
{8,	0.17,	0.62,	0.75,	0.75,	0.54},
{9,	0.18,	0.71,	0.76,	0.74,	0.52},
{10,0.20,	0.76,	0.83,	0.75,	0.51},
{11,0.21,	0.80,	0.80,	0.74,	0.51},
{12,	0.23,	0.83,	0.84,	0.75,	0.47},
{13,	0.24,	0.85,	0.86,	0.75,	0.47},
{13,	0.24,	0.85,	0.86,	0.75,	0.47},
{14,	0.26,	0.85,	0.85,	0.73,	0.45},
{15,	0.27,	0.88,	0.89,	0.72,	0.47}
};


/*


*/
	
	static void plot() {
		System.out.println("coordinates {");
		
		for(double[] line : values) {
			//(k,\theta, value)
			int k = (int) line[0];
			for(int i=0;i<thresholds.length;i++) {
				System.out.print("("+k+","+thresholds[i]+","+line[i+1]+")" );//line[0] == k
			}
			System.out.println();
			System.out.println();
		}
		
		System.out.println("};");
	}
	
	public static void main(String[] args) {
		plot();
	}
}
