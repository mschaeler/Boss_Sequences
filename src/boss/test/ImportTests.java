package boss.test;

import java.util.ArrayList;

import boss.load.*;
import plus.data.Book;

public class ImportTests {
	public static void main(String[] args) {
		Book b = ImporterAPI.get_book(Book.ELBERFELDER);
		System.out.println(b);
		
		ArrayList<Book> books;
		books = ImporterAPI.get_all_english_books();
		for(Book book : books) {
			System.out.println(book);
		}
		
		books = ImporterAPI.get_all_german_books();
		for(Book book : books) {
			System.out.println(book);
		}
		/*
		books = ImporterAPI.get_all_latin_books();
		for(Book book : books) {
			System.out.println(book);
		}
		
		books = ImporterAPI.get_all_greek_books();
		for(Book book : books) {
			System.out.println(book);
		}*/
	}
	
	
}
