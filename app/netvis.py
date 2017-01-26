from app import app
import pgmpy
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from flask import jsonify, render_template, request
from ast import literal_eval
import json

nodes = pd.read_csv('../bayesnetapp/app/data/nodes.csv',index_col=0)
links = pd.read_csv('../bayesnetapp/app/data/links.csv',index_col=0)
states = pd.read_csv('../bayesnetapp/app/data/states.csv',index_col=0)
states.states = states.states.apply(literal_eval)
cpt = {}
cpt['A'] = pd.read_csv('../bayesnetapp/app/data/cpt_A.csv',index_col=0)
cpt['B'] = pd.read_csv('../bayesnetapp/app/data/cpt_B.csv',index_col=0)
cpt['E'] = pd.read_csv('../bayesnetapp/app/data/cpt_E.csv',index_col=0)
cpt['JC'] = pd.read_csv('../bayesnetapp/app/data/cpt_JC.csv',index_col=0)
cpt['MC'] = pd.read_csv('../bayesnetapp/app/data/cpt_MC.csv',index_col=0)
dseplist = pd.read_csv('../bayesnetapp/app/data/dseplist.csv',index_col=0)
dseplist.separators = dseplist.separators.apply(literal_eval)

network = BayesianModel()
for index,row in links.iterrows():
    network.add_edge(row.source,row.target)
Bcpd = TabularCPD(variable='B',variable_card=2,values=[[cpt['B']['prob'][0],cpt['B']['prob'][1]]],
                  state_names={'B':['T','F']})
Ecpd = TabularCPD(variable='E',variable_card=2,values=[[cpt['E']['prob'][0],cpt['E']['prob'][1]]],
                  state_names={'E':['T','F']})
Acpd = TabularCPD(variable='A',variable_card=2,values=[[cpt['A']['prob'][0],cpt['A']['prob'][2],
                                                       cpt['A']['prob'][4],cpt['A']['prob'][6]],
                                                      [cpt['A']['prob'][1],cpt['A']['prob'][3],
                                                      cpt['A']['prob'][5],cpt['A']['prob'][7]]],
                 evidence=['B','E'],evidence_card=[2,2], state_names={'A':['T','F'],'B':['T','F'],
                                                                    'E':['T','F']})
JCcpd = TabularCPD(variable='JC',variable_card=2,values=[[cpt['JC']['prob'][0],cpt['JC']['prob'][2]],
                                                        [cpt['JC']['prob'][1],cpt['JC']['prob'][3]]],
                  evidence=['A'],evidence_card=[2],state_names={'JC':['T','F'],'A':['T','F']})
MCcpd = TabularCPD(variable='MC',variable_card=2,values=[[cpt['MC']['prob'][0],cpt['MC']['prob'][2]],
                                                        [cpt['MC']['prob'][1],cpt['MC']['prob'][3]]],
                  evidence=['A'],evidence_card=[2],state_names={'MC':['T','F'],'A':['T','F']})
network.add_cpds(Bcpd,Ecpd,Acpd,JCcpd,MCcpd)
assert network.check_model(), "Model CPTs are not consistent."
infer = VariableElimination(network,state_names={'B':['T','F'],'E':['T','F'],
                                                'A':['T','F'],'MC':['T','F'],
                                                'JC':['T','F']})
resdict = {}

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

@app.route('/cptdata/<variable>/')
def cptdata(variable):
    return cpt[variable].to_json(orient='records')

@app.route('/dsepdata/<source>/<target>/')
def dsepdata(source,target):
    return dseplist[(dseplist.source==source) & (dseplist.target==target)].to_json(orient='records')

@app.route('/navbartest/')
def navbartest():
    return render_template("navbartest.html")

@app.route('/networkdata/')
def networkdata():
    netdata = {"nodes":nodes.to_dict(orient='records'),"links":links.to_dict(orient='records')}
    return pd.json.dumps(netdata)

@app.route('/legend/')
def legend():
    nodelist = nodes.to_dict(orient='records')
    return render_template("legend.html", nodes=nodelist)

@app.route('/dseparation/')
def dseparation():
    nodelist = nodes.to_dict(orient='records')
    return render_template("dseparation.html", nodes=nodelist)

@app.route('/inference/')
def inference():
    statelist = states.to_dict(orient='records')
    return render_template("inference.html", states=statelist)

@app.route('/postmethod/', methods = ['POST','GET'])
def get_post_javascript_data():
    if request.method == 'POST':
        outp = str(request.json['output'])
        inp = str(request.json['input'])
        stat = str(request.json['state'])
        res = infer.query([outp], evidence={inp: stat})[outp]
        for i in range(0,len(res.values)):
            resdict[states.states[states.variable == outp].iloc[0][i]] = round(res.values[i],3)
    return json.dumps(resdict)
