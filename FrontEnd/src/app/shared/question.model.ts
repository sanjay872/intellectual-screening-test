export class questions{
    public id:String;
    public question:String;
    public questionsimg:String;
    public option1:String;
    public option2:String;
    public option3:String;
    public option4:String;
    public solution:String;
    constructor(id:String,question:String,quesimg:String,option1:String,option2:String,option3:String,option4:String,solution:String){
        this.id=id;
        this.question=question;
        this.questionsimg=quesimg;
        this.option1=option1;
        this.option2=option2;
        this.option3=option3;
        this.option4=option4;
        this.solution=solution;
    }
}