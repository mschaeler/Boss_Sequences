package pan;

import java.util.ArrayList;

import org.xml.sax.Attributes;
import org.xml.sax.SAXException;
import org.xml.sax.helpers.DefaultHandler;

public class PanHandlerDetections extends DefaultHandler {
	    private static final String FEATURE = "feature";
	    private static final String DOCUMENT = "document";

	    public ArrayList<Detection> detections = new ArrayList<Detection>();
	    private Detection current;
	    /**
	     * Name of the suspicious file
	     */
	    private String reference = null;
	    private StringBuilder elementValue;
	    public int k;
	    public double theta;

	    @Override
	    public void characters(char[] ch, int start, int length) throws SAXException {
	        if (elementValue == null) {
	            elementValue = new StringBuilder();
	        } else {
	            elementValue.append(ch, start, length);
	        }
	    }

	    @Override
	    public void startDocument() throws SAXException {
	    	//detections = new ArrayList<Detection>();
	    }

	    @Override
	    public void startElement(String uri, String lName, String qName, Attributes attr) throws SAXException {
	        switch (qName) {
	            case FEATURE:
	            	/*System.out.print(attr.getQName(0)+" ");
	            	System.out.println(attr.getValue(0));
	            	
	            	System.out.print(attr.getQName(1)+" ");
	            	System.out.println(attr.getValue(1));
	            	
	            	System.out.print(attr.getQName(2)+" ");
	            	System.out.println(attr.getValue(2));
	            	
	            	System.out.print(attr.getQName(3)+" ");
	            	System.out.println(attr.getValue(3));
	            	
	            	System.out.print(attr.getQName(4)+" ");
	            	System.out.println(attr.getValue(4));
	            	
	            	System.out.print(attr.getQName(5)+" ");
	            	System.out.println(attr.getValue(5)+" ");*/
	            	
	            	
	            	String temp;
	            	this.current = new Detection();
	            	current.susp = reference;
	            	//current.this_offset = attr.getQName(0);
	            	temp = attr.getValue(1);
	            	current.this_offset   = Integer.parseInt(temp);
	            	current.this_length   = Integer.parseInt(attr.getValue(2));
	            	current.src			  = attr.getValue(3);
	            	current.source_offset = Integer.parseInt(attr.getValue(4));
	            	current.source_length = Integer.parseInt(attr.getValue(5));
	            	
	            	current.k = k;
	            	current.theta = theta;
	            	
	            	//System.out.println(current);
	            	detections.add(current);
	                break;
	            case DOCUMENT:
	            	if(attr.getQName(0).equals("reference")) {
	            		reference = attr.getValue(0);
	            		System.out.println("New file "+reference);
	            	}else {
	            		System.err.println("Wrong XML file?");
	            	}
	                break;
	        }
	    }

	    @Override
	    public void endElement(String uri, String localName, String qName) throws SAXException {

	    }
	}