package com.example.demo.Repository;

import org.springframework.data.repository.CrudRepository;
import org.springframework.stereotype.Repository;

import com.example.demo.Entity.IqDataEntity;

@Repository
public interface IqDataRepository extends CrudRepository<IqDataEntity,Long> {

	IqDataEntity findByScore(int score);

}
