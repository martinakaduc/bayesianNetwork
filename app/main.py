from app import app
import pandas as pd
from ProbabilityModel.models import BayesianModel
from ProbabilityModel.factors.discrete import TabularCPD
from ProbabilityModel.inference import VariableElimination
from ProbabilityModel.independencies import Independencies, IndependenceAssertion
from flask import jsonify, render_template, request, send_file, redirect, url_for
from ast import literal_eval
import json
import numpy as np
import os
import sys

nodes = {}
nodelist = []
links = []
statelist = []
cpt = {}
network = BayesianModel()
resdict = {}

class BayesNode():
    def __init__(self, name:str, parents, domain, shape, proba):
        self.target = name
        self.domain = domain
        self.parents = list(reversed(parents))
        self.children = []
        if isinstance(shape, int):
            self.shape = [shape]
        else:
            self.shape = list(reversed(shape))
        self.cpd = proba
        # print(self.cpd)

    def appendChild(self, node):
        self.children.append(node.target)

def __extract_model(line):
    parts = line.split(';')
    node = parts[0]
    if parts[1] == '':
        parents = []
    else:
        parents = parts[1].split(',')
    domain = parts[2].split(',')
    shape = eval(parts[3])
    probabilities = np.array(eval(parts[4])).reshape(shape)
    return node, parents, domain, shape, probabilities


def read_file(path='app/data', file='model01.txt'):
    global nodes
    f = open(path+'/'+file, 'r')
    N = int(f.readline())
    lines = f.readlines()
    for line in lines:
        node, parents, domain, shape, probabilities = __extract_model(line)
        nodes[node] = BayesNode(node, parents, domain, shape, probabilities)
        for p in parents:
            nodes[p].appendChild(nodes[node])
    f.close()

# dseplist = pd.read_csv('../bayesnetapp/app/data/dseplist.csv',index_col=0)
# dseplist.separators = dseplist.separators.apply(literal_eval)

def reset_network():
    global nodes
    global nodelist
    global links
    global statelist
    global cpt
    global network
    global independences

    nodes = {}
    nodelist = []
    links = []
    statelist = []
    cpt = {}
    independences = None
    network = BayesianModel()

def construct_network():
    global nodes
    global nodelist
    global links
    global statelist
    global cpt
    global network
    global infer
    global independences

    iter = 0
    for node, data in nodes.items():
        for p in data.children:
            network.add_edge(node, p)
            links.append({'source':node, 'target':p, 'value':1})

        nodelist.append({'group':iter, 'id':node, 'name':node})
        statelist.append({'states': data.domain, 'variable':node})

        prob_fla = data.cpd.ravel()
        tmp = []
        col_tmp = [node] + data.parents
        gap = 1
        for i in range(len(col_tmp)):
            column = nodes[col_tmp[i]].domain
            tmp.append(np.array([[x] * gap for x in column] * (int(prob_fla.size) // (gap * data.shape[i]))).flatten())
            gap *= data.shape[i]

        tmp.append(prob_fla.astype(np.object))
        tmp = np.column_stack(tmp)
        # print(tmp)

        # parents_count = tmp.shape[1] - 2
        # if parents_count > 0:
        #     for i in range(parents_count):
        #         # cpt[node]['parent'+str(i+1)] = [data.parents[i]] * tmp.shape[0]
        #         cpt[node].append(tmp[:,i+1].tolist())

        # cpt[node].append(tmp[:,0].tolist())
        # cpt[node].append(tmp[:,-1].tolist())
        cpt[node] = []
        cpt[node].append(col_tmp)
        cpt[node].append(tmp.tolist())

        nodes[node].cpd = np.moveaxis(data.cpd, -1, 0).reshape((len(data.domain), -1))
        iter += 1

    for node, data in nodes.items():
        col_tmp = [node] + data.parents
        tabular = TabularCPD(variable=data.target, variable_card=len(data.domain), values=nodes[node].cpd,
                            evidence=data.parents[::-1], evidence_card=[len(nodes[x].domain) for x in data.parents[::-1]],
                            state_names={x:nodes[x].domain for x in col_tmp[::-1]})
        network.add_cpds(tabular)

    independences = network.get_independencies()
    # ind = IndependenceAssertion('I', 'D')
    # sys.stdout = open("independ.txt", "w")
    # print(independences)
    # sys.stdout.close()

    assert network.check_model(), "Model CPTs are not consistent."
    infer = VariableElimination(network)

read_file()
construct_network()

@app.route('/')
@app.route('/index/')
def index():
    return render_template("index.html", nodes=nodelist)

@app.route('/nodedata/')
def nodedata():
    return nodelist

@app.route('/linkdata/')
def linkdata():
    return links

@app.route('/cptdata/<variable>/')
def cptdata(variable):
    # print(cpt[variable])
    return json.dumps(cpt[variable])

@app.route('/dsepdata/<source>/<target>/', methods=['POST'])
def dsepdata(source, target):
    evidences = request.json['evidences']
    ind = IndependenceAssertion(source, target, evidences)
    return json.dumps(ind in independences)

@app.route('/navbartest/')
def navbartest():
    return render_template("navbartest.html")

@app.route('/networkdata/')
def networkdata():
    netdata = {"nodes":nodelist,"links":links}
    return json.dumps(netdata)

@app.route('/legend/')
def legend():
    return render_template("legend.html", nodes=nodelist)

@app.route('/model/', methods = ['GET'])
def model():
    return render_template("model.html", nodes=nodelist)

@app.route('/model/', methods = ['POST'])
def upload_model():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file:
        filename = file.filename
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        reset_network()
        read_file(file=filename)
        construct_network()
        return redirect(url_for('index',
                                filename=filename))
@app.route('/dseparation/')
def dseparation():
    return render_template("dseparation.html", nodes=nodelist)

@app.route('/inference/')
def inference():
    return render_template("inference.html", states=statelist)

@app.route('/example_file')
def example_file():
    return send_file('data/model01.txt', as_attachment=True)

@app.route('/postmethod/', methods = ['POST'])
def get_post_javascript_data():
    queries = request.json['queries']
    evidences = request.json['evidences']
    evidence_states = request.json['evidence_states']

    # print(queries)
    # print({key:val for key,val in zip(evidences, evidence_states)})

    res = infer.query(queries, evidence={key:val for key,val in zip(evidences, evidence_states)}, joint=True)
    print(res)

    scope = res.scope()
    cardinality = res.get_cardinality(scope)
    assignment = res.assignment(list(range(np.product(list(cardinality.values())))))
    values = res.values.reshape((-1,)).tolist()
    # print(assignment)
    # print(values)

    res_row = []
    for i, row in enumerate(assignment):
        res_row.append([x[1] for x in row] + [round(values[i], 4)])

    return json.dumps({'scope': scope, 'value': res_row})
