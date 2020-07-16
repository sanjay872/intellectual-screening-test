import { Component, OnInit } from '@angular/core';
import { authService } from '../shared/auth.service';
import { Router } from '@angular/router';
import { NgForm } from '@angular/forms';

@Component({
  selector: 'app-instruction',
  templateUrl: './instruction.component.html',
  styleUrls: ['./instruction.component.css']
})
export class InstructionComponent implements OnInit {
  limit="([8-9]|1[0-8])"
  constructor(private authservice:authService,private router:Router) { }

  ngOnInit() {
  }
 onSubmit(form:NgForm){
    this.router.navigate(['/test']);
    this.authservice.phyage=form.value.age;
  }
}
