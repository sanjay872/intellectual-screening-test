package com.example.demo.Entity;

import java.io.Serializable;

import javax.persistence.Column;
import javax.persistence.Entity;
import javax.persistence.Id;

@Entity(name="questions")
public class QuestionsEntity implements Serializable {
	/**
	 * 
	 */
	private static final long serialVersionUID = 7871976272932087520L;
	@Id
	@Column(nullable=false)
	private Long id;
	
	@Column()
	private String question;
	
	@Column()
	private String questionimg;
	
	@Column(nullable=false)
	private String option1;
	
	@Column(nullable=false)
	private String option2;
	
	@Column(nullable=false)
	private String option3;
	
	@Column(nullable=false)
	private String option4;
	
	@Column(nullable=false)
	private int solution;

	
	public Long getId() {
		return id;
	}

	public void setId(Long id) {
		this.id = id;
	}

	public String getQuestion() {
		return question;
	}

	public void setQuestion(String question) {
		this.question = question;
	}
	
	public String getQuestionimg() {
		return questionimg;
	}

	public void setQuestionimg(String questionimg) {
		this.questionimg = questionimg;
	}

	public String getOption1() {
		return option1;
	}

	public void setOption1(String option1) {
		this.option1 = option1;
	}

	public String getOption2() {
		return option2;
	}

	public void setOption2(String option2) {
		this.option2 = option2;
	}

	public String getOption3() {
		return option3;
	}

	public void setOption3(String option3) {
		this.option3 = option3;
	}

	public String getOption4() {
		return option4;
	}

	public void setOption4(String option4) {
		this.option4 = option4;
	}

	public int getSolution() {
		return solution;
	}

	public void setSolution(int solution) {
		this.solution = solution;
	}
	
}
