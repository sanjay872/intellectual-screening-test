package com.example.demo.UserServiceImpl;

import java.util.ArrayList;
import java.util.List;
import java.util.Optional;

import org.modelmapper.ModelMapper;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.example.demo.Dto.QuestionDto;
import com.example.demo.Entity.QuestionsEntity;
import com.example.demo.Repository.QuestionsRepository;
import com.example.demo.Services.QuestionService;
@Service
public class QuestionServiceImpl implements QuestionService{
	@Autowired
	QuestionsRepository questionRepository;
	@Override
	public List<QuestionDto> getquestion() {
		List<QuestionDto> returnvalue=new ArrayList<>();
		ModelMapper modelMapper=new ModelMapper();
		Iterable<QuestionsEntity> questions=questionRepository.findAll();
		for (QuestionsEntity questionEntity:questions) {
			returnvalue.add(modelMapper.map(questionEntity,QuestionDto.class));
		}
		return returnvalue;
	}
	@Override
	public QuestionDto getquestionbyid(long id) {
		QuestionDto returnvalue=new QuestionDto();
		Optional<QuestionsEntity> questionEntity=questionRepository.findById(id);
        returnvalue = new ModelMapper().map(questionEntity, QuestionDto.class);
		return returnvalue;
	}

}
