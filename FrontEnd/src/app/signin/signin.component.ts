import { Component, OnInit } from '@angular/core';
import { NgForm } from '@angular/forms';
import { authService } from '../shared/auth.service';
import { Router } from '@angular/router';

@Component({
  selector: 'app-signin',
  templateUrl: './signin.component.html',
  styleUrls: ['../shared/formsstyle.css','../shared/background.css']
})
export class SigninComponent implements OnInit {
  emailpattern="[a-zA-Z0-9.!#$%&'*+/=?^_`{|}~-]+@[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*";
  passwordpattern="[a-zA-Z0-9#$^+=!*()@%&]{7,}";
  constructor(private authservice:authService,private router:Router) { }
  userdata={email:'',password:''};
  errorstatus:boolean=false;
  ngOnInit() {
  }
  onSubmit(form:NgForm){
    if(!form.valid){
      return;
    }
    this.userdata.email=form.value.email;
    this.userdata.password=form.value.password;
    this.authservice.usersignin(this.userdata);
  }

}
