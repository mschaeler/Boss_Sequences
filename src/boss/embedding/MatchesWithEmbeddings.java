package boss.embedding;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class MatchesWithEmbeddings extends Match{
	double[] vector;
	
	public MatchesWithEmbeddings(String s1, String s2, double score, double[] vector) {
		super(s1, s2, score);
		this.vector = vector;
	}

	public MatchesWithEmbeddings(Match match, double[] vector) {
		this(match.s1, match.s2, match.score, vector);
	}
	
	String toTSV() {
		String s = s1+"\t"+s2+"\t"+score;
		for(double d : vector) {
			s+="\t"+d;
		}
		return s;
	}

	public static void materialize(String string, ArrayList<MatchesWithEmbeddings> mew) {
		// Creates a FileWriter
	    FileWriter file;
		try {
			file = new FileWriter(string);
		    // Creates a BufferedWriter
		    BufferedWriter output = new BufferedWriter(file);

		    String header = "Embedding entry\tSyntactic Token\tscore\tvector";
		    output.write(header);
		    output.newLine();

		    for(MatchesWithEmbeddings m : mew) {
		    	output.write(m.toTSV());
		    	output.newLine();
		    }
		    
		    // Closes the writer
		    output.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}
}
