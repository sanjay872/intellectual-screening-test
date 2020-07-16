import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders} from '@angular/common/http';
import { Router } from '@angular/router';
import { questions } from './question.model';
import { User } from './user.model';
import { BehaviorSubject } from 'rxjs';
import { iq } from './iq.model';
import { iqres } from './iqres.model';
@Injectable({
    providedIn: 'root'
  })
export class authService {
  constructor(private http:HttpClient,private route:Router) {}
  user=new  BehaviorSubject<User>(null);
  userid:string=null
  token:string=null;
  qns:questions[]=[];
  errorstatus=false;
  userans:string[]=[];
  iq:iq;
  phyage:number;
  useriq;
  usersignup(userdata={username:'',email:'',password:''}){
    return this.http.post("http://localhost:8080/users",userdata);
  }
  usersignin(userlogin={email:'',password:''}){
    return this.http.post("http://localhost:8080/users/login",userlogin,{observe:'response'}).subscribe(
      data => {
        this.errorstatus=false;
        const user=new User(data.headers.get("USERID"),data.headers.get("Authorization"));
        this.user.next(user);
        this.route.navigate(['/instruction']);
      },error => {
         this.errorstatus=true;
         console.log(error);
        }
  );
  }
  quiz(){
    this.user.subscribe(data=>{
      this.token=data.token;
      this.userid=data.userid;
    })
     this.http.get<questions[]>("http://localhost:8080/users/"+this.userid
    ,{headers:new HttpHeaders({'Authorization':this.token})}).subscribe(
      (res:any)=>{
        this.qns=res;
      },error=>{
        console.log(error);
      });;
}
  updatescore(score:number){
    const iqdata=new iq(score,this.phyage);
    this.http.post("http://localhost:8080/users/score/"+this.userid,iqdata,{headers:new HttpHeaders({'Authorization':this.token})}).subscribe();
    //this.http.get("http://localhost:8080/users/result",{headers:new HttpHeaders({'Authorization':this.token})}).subscribe();
  }
logout(){
  this.user.next(null);
  this.iq=null;
  this.phyage=null;
  this.qns=null;
  this.token=null;
  this.userans=null;
  this.userid=null;
  this.useriq=null;
  this.route.navigate(['/signin']);
}

getiq(){
  this.http.get<iqres>("http://127.0.0.1:5000/linearresult/"+this.userid,{observe:'response'}).subscribe(res=>{
    this.useriq=res.body.Iq;
  })
}
}
