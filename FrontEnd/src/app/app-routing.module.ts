import { NgModule } from '@angular/core';
import { Routes, RouterModule } from '@angular/router';
import { SignupComponent } from './signup/signup.component';
import { QuizComponent } from './quiz/quiz.component';
import { ResultComponent } from './result/result.component';
import { SigninComponent } from './signin/signin.component';
import { InstructionComponent } from './instruction/instruction.component';
import { authguard } from './auth-guard.service';
const routes: Routes = [
  {path:'signup',component:SignupComponent},
  {path:'signin',component:SigninComponent},
  {path:'test',component:QuizComponent,canActivate:[authguard]},
  {path:'result',component:ResultComponent,canActivate:[authguard]},
  {path:'instruction',component:InstructionComponent,canActivate:[authguard]},
  {path:'',redirectTo:'/signin',pathMatch:'full'}/*any other path will be redirected to the registeration */
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
