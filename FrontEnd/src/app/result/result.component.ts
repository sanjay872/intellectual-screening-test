import { Component, OnInit } from '@angular/core';
import { authService } from '../shared/auth.service';

@Component({
  selector: 'app-result',
  templateUrl: './result.component.html',
  styleUrls: ['./result.component.css']
})
export class ResultComponent implements OnInit {
  solution:String[]=[];
  isloading=true
  i:number;
  iq;
  score:number=0;
  count:number=this.authservice.qns.length;
  constructor(private authservice:authService) {}
  ngOnInit() {
    this.authservice.qns.forEach(sol=>{
      this.solution.push(sol.solution);
    })
    for(this.i=0;this.i<=this.solution.length;this.i++){
      if(this.solution[this.i]!=this.authservice.userans[this.i+1]){
        this.count--;
      }
    }
    this.authservice.updatescore(this.count);
    this.score=this.count;
    setTimeout(()=>{
      this.isloading=false;
      this.authservice.getiq();
    },2000)
  }

}
