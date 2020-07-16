package com.example.demo.Repository;



import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

import com.example.demo.Entity.QuestionsEntity;

@Repository
public interface QuestionsRepository extends CrudRepository<QuestionsEntity,Long>{
}
