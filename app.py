from flask import Flask, flash, redirect, render_template, request, session, abort
import live_words
from flask_cors import CORS, cross_origin



app = Flask(__name__)
CORS(app)

 
@app.route("/")
def index():
    return "Index!"
 
@app.route("/hello")
def hello():
   return render_template('test.html',name=name)
 
@app.route("/members")
def members():
    return "Members"
 
@app.route("/members/<string:name>/")
def getMember(name):
    return name

@app.route("/data")
def data():
    nl_url = request.args.get('nlurl') 
    print(nl_url)
    l = live_words.main(nl_url)
    print "in route", l
    st = str(l)
    st = st.replace("'", "")
    return st

if __name__ == "__main__":
    app.run()