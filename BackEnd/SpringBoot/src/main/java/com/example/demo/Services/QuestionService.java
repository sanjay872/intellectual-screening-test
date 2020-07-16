package com.example.demo.Services;


import java.util.List;


import com.example.demo.Dto.QuestionDto;


public interface QuestionService {
	List<QuestionDto> getquestion();
	QuestionDto getquestionbyid(long id);
}
