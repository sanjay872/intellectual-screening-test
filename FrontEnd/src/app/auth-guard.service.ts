import { CanActivate, ActivatedRouteSnapshot, RouterStateSnapshot } from '@angular/router';
import { Observable } from 'rxjs';
import { authService } from './shared/auth.service';
import { map } from 'rxjs/operators';
import { Injectable } from '@angular/core';
@Injectable({providedIn:'root'})
export class authguard implements CanActivate{
  constructor(private authservice:authService){}
    canActivate(route:ActivatedRouteSnapshot,state:RouterStateSnapshot):Observable<boolean>|Promise<boolean>|boolean{
      return this.authservice.user.pipe(map(user=>{
        return !!user;
      }))
    }
}