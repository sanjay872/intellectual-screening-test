package com.example.demo.Entity;

import java.io.Serializable;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.GeneratedValue;
import javax.persistence.Id;

@Entity(name="iqdata")
public class IqDataEntity implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 7502977299584772379L;

	@Id
	@GeneratedValue
	private long id;
	
	@Column(nullable=false)
	private int mentalage;
	
	@Column(nullable=false)
	private int score;
	
	
	public int getMentalage() {
		return mentalage;
	}
	public void setMentalage(int mentalage) {
		this.mentalage = mentalage;
	}
	public int getScore() {
		return score;
	}
	public void setScore(int score) {
		this.score = score;
	}
	
}
