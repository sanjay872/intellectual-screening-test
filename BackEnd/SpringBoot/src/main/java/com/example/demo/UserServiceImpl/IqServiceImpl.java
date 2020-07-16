package com.example.demo.UserServiceImpl;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

import com.example.demo.Entity.IqDataEntity;
import com.example.demo.Entity.IqEntity;
import com.example.demo.Repository.IqDataRepository;
import com.example.demo.Repository.IqRepository;
import com.example.demo.Request.IqModel;
import com.example.demo.Services.IqService;

@Service
public class IqServiceImpl implements IqService {

	@Autowired
	IqRepository iqRepository;
	
	@Autowired
	IqDataRepository iqDataRepository;
	
	@Override
	public void updatescore(IqModel iqmodel, String id) {
		int mentalage=0;
		Iterable<IqDataEntity> iqDataEntity=iqDataRepository.findAll();
		for(IqDataEntity iqdataEntity:iqDataEntity) {
			if(iqdataEntity.getScore()>=iqmodel.getScore()) {
				mentalage=iqdataEntity.getMentalage();
				break;
			}
		}
		if(iqRepository.existsByUserId(id)) {
			IqEntity iqEntity=iqRepository.findByUserId(id);
			iqEntity.setMentalage(mentalage);
			iqEntity.setPhyage(iqmodel.getPhyage());
			iqEntity.setScore(iqmodel.getScore());
			iqRepository.save(iqEntity);
		}else {
			IqEntity iqEntity=new IqEntity();
			iqEntity.setUserId(id);
			iqEntity.setPhyage(iqmodel.getPhyage());
			iqEntity.setMentalage(mentalage);
			iqEntity.setScore(iqmodel.getScore());
			iqRepository.save(iqEntity);
		}
		

	}

}
