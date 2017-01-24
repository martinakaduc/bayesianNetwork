from app import app
import pgmpy
import pandas as pd
from pgmpy.models import BayesianModel
from flask import jsonify, render_template
from ast import literal_eval

nodes = pd.read_csv('../bayesnetapp/app/data/nodes.csv',index_col=0)
links = pd.read_csv('../bayesnetapp/app/data/links.csv',index_col=0)
cpt = {}
cpt['A'] = pd.read_csv('../bayesnetapp/app/data/cpt_A.csv',index_col=0)
cpt['B'] = pd.read_csv('../bayesnetapp/app/data/cpt_B.csv',index_col=0)
cpt['E'] = pd.read_csv('../bayesnetapp/app/data/cpt_E.csv',index_col=0)
cpt['JC'] = pd.read_csv('../bayesnetapp/app/data/cpt_JC.csv',index_col=0)
cpt['MC'] = pd.read_csv('../bayesnetapp/app/data/cpt_MC.csv',index_col=0)
dseplist = pd.read_csv('../bayesnetapp/app/data/dseplist.csv',index_col=0)
dseplist.separators = dseplist.separators.apply(literal_eval)

@app.route('/')
@app.route('/index/')
def index():
    return render_template("index.html")

@app.route('/nodedata/')
def nodedata():
    return nodes.to_json(orient='records')

@app.route('/linkdata/')
def linkdata():
    return links.to_json(orient='records')

@app.route('/cptdata/<variable>')
def cptdata(variable):
    return cpt[variable].to_json(orient='records')

@app.route('/dsepdata/<source>/<target>')
def dsepdata(source,target):
    return dseplist[(dseplist.source==source) & (dseplist.target==target)].to_json(orient='records')

@app.route('/navbartest/')
def navbartest():
    return render_template("navbartest.html")

@app.route('/networkdata/')
def networkdata():
    netdata = {"nodes":nodes.to_dict(orient='records'),"links":links.to_dict(orient='records')}
    return pd.json.dumps(netdata)
