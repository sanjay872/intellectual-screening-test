from flask import Flask,request,jsonify;
from flask_sqlalchemy import SQLAlchemy;
import numpy as np
import pickle
from flask_restful import  Api;
from flask_sqlalchemy import SQLAlchemy; #for database connection
from flask_marshmallow import Marshmallow;
from flask_cors import CORS

app=Flask(__name__)


app.config['CORS_HEADERS'] = 'Content-Type'
app.config['SQLALCHEMY_DATABASE_URI']='mysql://sanjay:sanjay@localhost/userdetails' #DB URI
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False #to avoid warning


db=SQLAlchemy(app) 
ma=Marshmallow(app) 
cors = CORS(app, resources={r"*": {"origins": "*"}})
api = Api(app)

class Iq(db.Model): #database structure of table post
    user_id=db.Column(db.String,primary_key=True)
    mentalage=db.Column(db.Integer)
    phyage=db.Column(db.Integer) 
    score=db.Column(db.Integer)
    def __init__(self,mentalage,phyage,score):
        self.mentalage=mentalage 
        self.phyage=phyage
        self.score=score

class IqSchema(ma.Schema):
    class Meta:
        fields=("mentalage","phyage")
iq_schema=IqSchema() 

def ValuePredictor(to_predict_list): 
    to_predict = np.array(to_predict_list).reshape(1,2) 
    loaded_model = pickle.load(open("linear_model.pkl", "rb")) 
    result = loaded_model.predict(to_predict) 
    return result[0] 
  
@app.route('/linearresult/<id>', methods = ['GET']) 
def result(id): 
    if request.method == 'GET': 
        to_predict_list =Iq.query.get(id)
        result=iq_schema.dump(to_predict_list)
        to_predict_list = list(result.values()) 
        to_predict_list = list(map(int, to_predict_list[:len(to_predict_list)])) 
        result = ValuePredictor(to_predict_list)  
        result=jsonify({"Iq":result});   
    return  result
   
    
if __name__ == "__main__":
    app.run(debug=True)