"""
This script runs the application using a development server.
It contains the definition of routes and views for the application.
"""

from flask import Flask
from flask import render_template, request
import pandas as pd
import numpy as np
from PO import *
from datetime import datetime

app = Flask(__name__)

# Make the WSGI interface available at the top level so wfastcgi can get it.
wsgi_app = app.wsgi_app


@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        title='Home Page',
        year=datetime.now().year
    )

@app.route('/minrisk')
def minrisk():
    """Renders the home page."""
    return render_template(
        'minrisk.html',
        title='Minimum Risk Portfolio',
        year=datetime.now().year
    )
@app.route('/minrisk',methods=['POST','GET'])
def min_risk():
    if request.method == 'POST':
        inv = float(request.form['InvAmount'])
        msg = Min_Volatility_Portfolio(inv,clusters)
        msg1 =msg[0]
        msg2 =msg[1]
        msg3 =msg[2]
        msg4 =msg[3]
        df1 = msg[4].to_html()
        message = [msg1,msg2,msg3,msg4,df1]
        
    return render_template('minrisk.html',title='Minimum Risk Portfolio',message=message)

@app.route('/contact')
def contact():
    """Renders the contact page."""
    return render_template(
        'contact.html',
        title='Porfolios by Return %',
        year=datetime.now().year
    )
@app.route('/contact',methods=['POST','GET'])
def rtnper():
    if request.method == 'POST':
        inv = float(request.form['InvAmount'])
        rtn = int(request.form['rtn_per'])
        msg = Investor_Specified_Returns_Portfolio(inv,clusters,rtn)
        msg1 =msg[0]
        msg2 =msg[1]
        msg3 =msg[2]
        msg4 =msg[3]
        df1 = msg[4].to_html()
        message = [msg1,msg2,msg3,msg4,df1]
        
    return render_template('contact.html',title='Porfolios by Return %',message=message)

@app.route('/efficient')
def eff():
    """Renders the about page."""
    return render_template(
        'efficient.html',
        title='Efficient Portfolio',
        year=datetime.now().year
    )
@app.route('/efficient',methods=['POST','GET'])
def eff_por():
    if request.method == 'POST':
        inv = float(request.form['InvAmount'])
        msg = Max_Sharpe_Portfolio(inv,clusters)
        msg1 =msg[0]
        msg2 =msg[1]
        msg3 =msg[2]
        msg4 =msg[3]
        df1 = msg[4].to_html()
        message = [msg1,msg2,msg3,msg4,df1]
       
    return render_template('efficient.html',title='Efficient Portfolio',message=message)

@app.route('/about')
def about():
    """Renders the about page."""
    return render_template(
        'about.html',
        title='Maximum Return Portfolio',
        year=datetime.now().year
    )
@app.route('/about',methods=['POST','GET'])
def max_rtn():
    if request.method == 'POST':
        inv = float(request.form['InvAmount'])
        msg = Max_Returns_Portfolio(inv,clusters)
        msg1 =msg[0]
        msg2 =msg[1]
        msg3 =msg[2]
        msg4 =msg[3]
        df1 = msg[4].to_html()
        message = [msg1,msg2,msg3,msg4,df1]
        
    return render_template('about.html',title='Maximum Return Portfolio',message=message)

if __name__ == '__main__':
    app.run()
