import { Component, OnInit } from '@angular/core';
import { authService } from '../shared/auth.service';

@Component({
  selector: 'app-navigation',
  templateUrl: './navigation.component.html',
  styleUrls: ['./navigation.component.css']
})
export class NavigationComponent implements OnInit {

  constructor(private authservice:authService) { }

  ngOnInit() {
  }
 logout(){
  this.authservice.logout();
  }
}
