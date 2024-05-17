package pan;

import java.io.File;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import javax.xml.parsers.ParserConfigurationException;
import javax.xml.parsers.SAXParser;
import javax.xml.parsers.SAXParserFactory;
import org.xml.sax.SAXException;

public class PanXML_Parser {
	static String res_dir = "./results/xml";
	static String ground_truth_dir = "./data/pan11/01-manual-obfuscation-highjac/groud-truth";
	
    public static void main(String[] args) throws ParserConfigurationException, SAXException {
    	ArrayList<Detection> detections = run_detections();
    	ArrayList<Detection> ground_truths = run_ground_truth();
    	
    	PanMetrics.run_xml_detections(ground_truths, detections);
    }
    
    public static ArrayList<Detection> run_ground_truth() throws ParserConfigurationException, SAXException {
    	String dir = ground_truth_dir;
    	List<String> files = listFilesUsingFilesList(dir);
    	/*for(String f : files) {
    		System.out.println(f);
    	}*/
    	
    	SAXParserFactory factory = SAXParserFactory.newInstance();
        SAXParser saxParser = factory.newSAXParser();

        PanHandlerGroundTruth ph = new PanHandlerGroundTruth();
        //saxParser.parse("./results/xml/suspicious-document00228_10_0.300000.xml", ph);
        for(String f : files) {
        	try {
        		saxParser.parse(dir+"/"+f, ph);	
			} catch (Exception e) {
				System.out.println(e);
			}
        }
        
        return ph.detections; 
    }
    
    public static ArrayList<Detection> run_detections() throws ParserConfigurationException, SAXException {
    	String dir = res_dir;
    	List<String> files = listFilesUsingFilesList(dir);
    	/*for(String f : files) {
    		System.out.println(f);
    	}*/
    	
    	SAXParserFactory factory = SAXParserFactory.newInstance();
        SAXParser saxParser = factory.newSAXParser();

        PanHandlerDetections ph = new PanHandlerDetections();
        //saxParser.parse("./results/xml/suspicious-document00228_10_0.300000.xml", ph);
        for(String f : files) {
        	String[] tokens = f.split("_");//e.g., suspicious-document00228_3_0.300000.xml
        	
    		int k = Integer.parseInt(tokens[1]);
    		double theta = Double.parseDouble(tokens[2].substring(0,4));
    		System.out.println(f+" k="+k+" theta="+theta);
        	ph.k = k;
        	ph.theta = theta;
    		
        	try {
        		saxParser.parse(dir+"/"+f, ph);	
			} catch (Exception e) {
				System.out.println(e);
			}
        }
        
        return ph.detections; 
    }
    
	public static List<String> listFilesUsingFilesList(String dir) {
		File directory = new File(dir);
		if(!directory.exists()) {
			System.err.println(dir+" does not exist");
		}
		
	    try (Stream<Path> stream = Files.list(Paths.get(dir))) {
	        return stream
	          .filter(file -> !Files.isDirectory(file))
	          .map(Path::getFileName)
	          .map(Path::toString)
	          .collect(Collectors.toList());
	    }catch (Exception e) {
			System.err.println(e);
		}
	    return null;
	}
}
