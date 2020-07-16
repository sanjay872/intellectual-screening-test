package com.example.demo.Services;

import org.springframework.security.core.userdetails.UserDetailsService;

import com.example.demo.Dto.UserDto;

public interface UserServices extends UserDetailsService {
UserDto createUser(UserDto user);
UserDto getUser(String email);
boolean checkuserid(String id);
}
