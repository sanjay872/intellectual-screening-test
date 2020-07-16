export class User{
    constructor(private _userid:string,private _token:string){}
    set token(token:string){
        this._token=token;
    }
    set userid(userid:string){
        this._userid=userid;
    }
    get token(){
        return this._token;
    }
    get userid(){
        return this._userid;
    }
    
   
}