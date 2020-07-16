package com.example.demo.Entity;

import java.io.Serializable;
import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.Id;
import javax.persistence.Table;

@Entity()
@Table(name="iq")
public class IqEntity implements Serializable{

	/**
	 * 
	 */
	private static final long serialVersionUID = 6006804666333928787L;
	
	@Id
	@Column(nullable=false)
	private String userId;
	
	@Column(nullable=true)
	private int score;
	
	@Column(nullable=true)
	private int mentalage;
	
	@Column(nullable=true)
	private int phyage;
	
	public int getScore() {
		return score;
	}

	public void setScore(int score) {
		this.score = score;
	}

	public int getMentalage() {
		return mentalage;
	}

	public void setMentalage(int mentalage) {
		this.mentalage = mentalage;
	}

	public int getPhyage() {
		return phyage;
	}

	public void setPhyage(int phyage) {
		this.phyage = phyage;
	}

	public String getUserId() {
		return userId;
	}

	public void setUserId(String userId) {
		this.userId = userId;
	}

	
	
	
	
}
