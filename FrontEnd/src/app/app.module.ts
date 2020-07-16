import { BrowserModule } from '@angular/platform-browser';
import { NgModule } from '@angular/core';
import {FormsModule} from'@angular/forms';
import { AppRoutingModule } from './app-routing.module';
import { AppComponent } from './app.component';
import { SignupComponent } from './signup/signup.component';
import { NavigationComponent } from './navigation/navigation.component';
import { QuizComponent } from './quiz/quiz.component';
import { ResultComponent } from './result/result.component';
import {HttpClientModule} from '@angular/common/http';
import { SigninComponent } from './signin/signin.component';
import { NgxPasswordToggleModule } from 'ngx-password-toggle';
import { InstructionComponent } from './instruction/instruction.component';
import { LoaderComponent } from './shared/loader/loader.component';

@NgModule({
  declarations: [
    AppComponent,
    SignupComponent,
    NavigationComponent,
    QuizComponent,
    ResultComponent,
    SigninComponent,
    InstructionComponent,
    LoaderComponent
  ],
  imports: [
    BrowserModule,
    AppRoutingModule,
    FormsModule,
    HttpClientModule,
    NgxPasswordToggleModule
  ],
  providers: [],
  bootstrap: [AppComponent]
})
export class AppModule { }
