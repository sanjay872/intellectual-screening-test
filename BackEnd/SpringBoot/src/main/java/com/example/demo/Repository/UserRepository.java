package com.example.demo.Repository;

import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

import com.example.demo.Entity.UserEntity;

@Repository
public interface UserRepository extends CrudRepository<UserEntity,Long> {
	UserEntity findByEmail(String email);
	UserEntity findByUserId(String id);
}
