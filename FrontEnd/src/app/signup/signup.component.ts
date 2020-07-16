import { Component, OnInit } from '@angular/core';
import { NgForm } from '@angular/forms';
import { authService } from '../shared/auth.service';
import { Router } from '@angular/router';
import {tap} from 'rxjs/operators';
@Component({
  selector: 'app-signup',
  templateUrl: './signup.component.html',
  styleUrls: ['../shared/formsstyle.css','../shared/background.css']
})
export class SignupComponent implements OnInit {
  usernamepattern="[a-zA-Z0-9]{6,}";
  emailpattern="[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*";
  passwordpattern="[a-zA-Z0-9#$^+=!*()@%&]{7,}";
  userdata={username:'',email:'',password:''};
  errorstatus:boolean=false;
  image;
  imgstatus=false;
  constructor(private authservice:authService,private router:Router) { }
  ngOnInit() {
  }
  onSubmit(form:NgForm){
    if(!form.valid){
      return;
    }
    this.userdata.username=form.value.username;
    this.userdata.email=form.value.email;
    this.userdata.password=form.value.password;
    this.authservice.usersignup(this.userdata).pipe(tap(resdata=>{
    })).subscribe(
     data=>{
      this.errorstatus=false;
      this.router.navigate(['/signin']);
     },error=>{
       this.errorstatus=true;
     }
   );
  }

}
