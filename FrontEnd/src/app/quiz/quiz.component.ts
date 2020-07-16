import { Component, OnInit } from '@angular/core';
import { authService } from '../shared/auth.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-quiz',
  templateUrl: './quiz.component.html',
  styleUrls: ['./quiz.component.css']
})
export class QuizComponent implements OnInit {
  isloading=true;
  constructor(private authservice:authService,private route:Router) {
  }
  i:number;
  ngOnInit() {
    setTimeout(()=>{
      this.isloading=false;
    },1000)
    this.authservice.quiz();
}
useranswer(index,opt){
  this.authservice.userans[index]=opt;
}
submitanswer(){
  this.route.navigate(['/result']);
}
}