package com.example.demo.controller;



import java.lang.reflect.Type;
import java.util.ArrayList;
import java.util.List;


import org.modelmapper.ModelMapper;
import org.modelmapper.TypeToken;
import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

import com.example.demo.Dto.QuestionDto;
import com.example.demo.Dto.UserDto;
import com.example.demo.Request.IqModel;
import com.example.demo.Request.RequestModel;
import com.example.demo.Response.QuestionResponseModel;
import com.example.demo.Response.UserResponseModel;
import com.example.demo.Services.IqService;
import com.example.demo.Services.QuestionService;
import com.example.demo.Services.UserServices;


@RestController
@RequestMapping("/users")
//@CrossOrigin(origins="*")
public class Controller {
	@Autowired
		UserServices userService;
	
	@Autowired
		QuestionService questionService;
	
	@Autowired
		IqService iqService;
	
	@PostMapping
	public UserResponseModel createUser(@RequestBody RequestModel req) {
		UserResponseModel returnvalue=new UserResponseModel();
		UserDto dto=new UserDto();
		BeanUtils.copyProperties(req, dto);
		UserDto createduser=userService.createUser(dto);
		BeanUtils.copyProperties(createduser, returnvalue);
		return returnvalue;
	}
	@GetMapping("/{id}")
	public List<QuestionResponseModel> createuser(@PathVariable String id ) {
		List<QuestionResponseModel> questionResponseModel=new ArrayList<>();
		List<QuestionDto> questionDto=questionService.getquestion();
		if(userService.checkuserid(id)) {
			if (questionDto != null && !questionDto.isEmpty()) {
				Type listType = new TypeToken<List<QuestionResponseModel>>() {
				}.getType();
				questionResponseModel = new ModelMapper().map(questionDto, listType);
		}else {
			throw new RuntimeException("Questions are not exist");
		}
		}else {
			throw new RuntimeException("Invalid User");
		}
		return questionResponseModel;
	}
	@PostMapping("/score/{id}")
	public void updatescore(@RequestBody IqModel iqmodel ,@PathVariable String id) {
	iqService.updatescore(iqmodel, id);
	}
	
}
