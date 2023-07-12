package boss.embedding;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

public class MatchesWithEmbeddings extends Match{
	public double[] vector;
	
	public MatchesWithEmbeddings(String s1, String s2, double score, double[] vector) {
		super(s1, s2, score);
		this.vector = vector;
	}

	public MatchesWithEmbeddings(Match match, double[] vector) {
		this(match.string_in_embedding, match.string_in_text, match.score, vector);
	}
	
	public static MatchesWithEmbeddings to_instance(String line) {
		String[] components = line.split("\t");
		String s1 = components[0];
		String s2 = components[1];
		Double score = Double.parseDouble(components[2]);
		double[] vector = new double[components.length-3];
		for(int i=0;i<vector.length;i++) {
			Double temp = Double.parseDouble(components[i+3]);
			vector[i] = temp.doubleValue();
		}
		return new MatchesWithEmbeddings(s1, s2, score, vector);
	}

	String toTSV() {
		String s = string_in_embedding+"\t"+string_in_text+"\t"+score;
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
	public static ArrayList<MatchesWithEmbeddings> load(String file_path) {
		// Creates a FileWriter
	    ArrayList<MatchesWithEmbeddings> mew = new ArrayList<MatchesWithEmbeddings>(1000);
		try {
		    // Creates a BufferedWriter
		    //BufferedReader input = new BufferedReader(new FileReader(file));
			Path path = Paths.get(file_path);
			BufferedReader input = Files.newBufferedReader(path,StandardCharsets.UTF_8);
		    String header = input.readLine();
		    System.out.println(header);
		    
			String line;
			int i = 0;
		
	        while ((line = input.readLine()) != null) {
	        	MatchesWithEmbeddings temp = to_instance(line);
	        	mew.add(temp);
	        	if(i%100==0) {
	        		System.out.println(line);	
	        	}
	        	i++;
	        }
		    
		    // Closes the reader
		    input.close();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		return mew;		
	}
}
