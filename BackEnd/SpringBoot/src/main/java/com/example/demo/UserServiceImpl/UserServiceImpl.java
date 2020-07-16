package com.example.demo.UserServiceImpl;
import java.util.ArrayList;

import org.springframework.beans.BeanUtils;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.security.core.userdetails.User;
import org.springframework.security.core.userdetails.UserDetails;
import org.springframework.security.core.userdetails.UsernameNotFoundException;
import org.springframework.security.crypto.bcrypt.BCryptPasswordEncoder;
import org.springframework.stereotype.Service;

import com.example.demo.Dto.UserDto;
import com.example.demo.Entity.UserEntity;
import com.example.demo.Repository.UserRepository;
import com.example.demo.Services.UserServices;
import com.example.demo.Shared.Utils;

@Service
public class UserServiceImpl implements UserServices{
	@Autowired
	UserRepository userRepository;
	
	@Autowired
	Utils utils;
	
	@Autowired
	BCryptPasswordEncoder  bCryptPasswordEncoder; 
	@Override
	public UserDto createUser(UserDto user) {
		UserEntity userEntity =new UserEntity();
		BeanUtils.copyProperties(user, userEntity);
		userEntity.setEncryptedPassword(bCryptPasswordEncoder.encode(user.getPassword()));
		userEntity.setUserId(utils.generateUserId(20));
		UserEntity storeduser=userRepository.findByEmail(user.getEmail());
		if(storeduser!=null) throw new RuntimeException("record already exist");
		UserEntity storeuserDetail=userRepository.save(userEntity);
		UserDto returnvalue=new UserDto();
		BeanUtils.copyProperties(storeuserDetail, returnvalue);
		return returnvalue;
	}
	@Override
	public UserDetails loadUserByUsername(String email) throws UsernameNotFoundException {
		UserEntity userEntity=userRepository.findByEmail(email);
		if(userEntity==null)throw new UsernameNotFoundException(email);
		return new User(userEntity.getEmail(),userEntity.getEncryptedPassword(),new ArrayList<>());
	}
	@Override
	public UserDto getUser(String email) {
		UserDto userDto=new UserDto();
		UserEntity storeduser=userRepository.findByEmail(email);
		if(storeduser==null)throw new UsernameNotFoundException(email);
		BeanUtils.copyProperties(storeduser, userDto);
		return userDto;
	}
	@Override
	public boolean checkuserid(String id) {
		UserEntity storeduser=userRepository.findByUserId(id);
		if(storeduser==null) {
			return false;
		}
		return true;
	}

}
