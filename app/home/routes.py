# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

# Python modules
import os
from . import blueprint
# Flask modules
from flask import render_template, request, url_for, redirect, send_from_directory, jsonify
from flask_login import login_user, logout_user, current_user
from app import lm, db, bc
from app.models import User
from app.forms import LoginForm, RegisterForm

from googletrans import Translator
import plotly
import plotly.express as px
import plotly.graph_objects as go
from .topic_modeling import *

import pandas as pd
import numpy as np
import json

import spacy


word_corpus = []
dataset = ''
lang_dict = {'theguardian': 'en', 'bbc': 'en', 'dailymail': 'en', 'telegraph': 'en',
                 'globo': 'pt', 'estadao': 'pt', 'folha': 'pt', 'elmundo': 'es',
             'elpais':'es'}

euro_news = ['theguardian', 'dailymail', 'elmundo', 'elpais']


# App modules
#from app import app, lm
#from app.home.models import User
#from app.forms import LoginForm, RegisterForm


@blueprint.route('/')
def index2():
    return redirect(url_for('home_blueprint.index'))
    #return render_template('index.html')
# provide login manager with load_user callback

@lm.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Logout user
@blueprint.route('/home_blueprint.logout.html')
def logout():
    logout_user()
    return redirect(url_for('home_blueprint.login'))


# Register a new user
@blueprint.route('/register.html', methods=['GET', 'POST'])
def register():
    # declare the Registration Form
    form = RegisterForm(request.form)
    msg = None
    if request.method == 'GET':
        return render_template('layouts/auth-default.html',
                               content=render_template('pages/register.html', form=form, msg=msg))

    # check if both http method is POST and form is valid on submit
    if form.validate_on_submit():

        # assign form data to variables
        username = request.form.get('username', '', type=str)
        password = request.form.get('password', '', type=str)
        email = request.form.get('email', '', type=str)

        # filter User out of database through username
        user = User.query.filter_by(user=username).first()

        # filter User out of database through username
        user_by_email = User.query.filter_by(email=email).first()

        if user or user_by_email:
            msg = 'Error: User exists!'

        else:

            pw_hash = password  # bc.generate_password_hash(password)

            user = User(username, email, pw_hash)

            user.save()

            msg = 'User created, please <a href="' + url_for('home_blueprint.login') + '">login</a>'

    else:
        msg = 'Input error'

    return render_template('layouts/auth-default.html',
                           content=render_template('pages/register.html', form=form, msg=msg))


# Authenticate user
@blueprint.route('/login.html', methods=['GET', 'POST'])
def login():
    # Declare the login form
    form = LoginForm(request.form)

    # Flask message injected into the page, in case of any errors
    msg = None

    # check if both http method is POST and form is valid on submit
    if form.validate_on_submit():

        # assign form data to variables
        username = request.form.get('username', '', type=str)
        password = request.form.get('password', '', type=str)

        # filter User out of database through username
        user = User.query.filter_by(user=username).first()

        if user:

            # if bc.check_password_hash(user.password, password):
            if user.password == password:
                login_user(user)
                return redirect(url_for('home_blueprint.index'))
            else:
                msg = "Wrong password. Please try again."
        else:
            msg = "Unkkown user"

    return render_template('layouts/auth-default.html',
                           content=render_template('pages/login.html', form=form, msg=msg))


# App main route + generic routing
@blueprint.route('/index.html')
#@blueprint.route('/<path>')
def index():
    #if not current_user.is_authenticated:
     #   return redirect(url_for('home_blueprint.login'))

    #content = None

   # try:

        # try to match the pages defined in -> pages/<input file>
        return render_template('layouts/default.html',
                               content=render_template('pages/index.html'))
   # except:

    #    return render_template('layouts/auth-default.html',
     #                          content=render_template('pages/404.html'))


# Return sitemap
@blueprint.route('/sitemap.xml')
def sitemap():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'sitemap.xml')

@blueprint.route('/select', methods=['GET', 'POST'])
def upload_dataset_d3():
    global fd_corpus, dataset
    app.logger.info('This is a log message!')
    dataset = request.args['fdDataset']
    if dataset == 'FIFA':
        word_corpus = json.load(open('data/word_corpus_caio.json', 'r', encoding='utf-8'))
    elif dataset == 'earthquake':
        word_corpus = json.load(open('data/word_corpus_daniela.json', 'r', encoding='utf-8'))
    elif dataset == 'global':
        word_corpus = json.load(open('data/word_corpus_daniela.json', 'r', encoding='utf-8'))
    else:
        word_corpus = json.load(open('data/word_corpus_daniela.json', 'r', encoding='utf-8'))

    return jsonify(out=dataset)


@blueprint.route('/abdul', methods=['GET', 'POST'])
def abdul():
    return render_template("pages/indexD3.html")


@blueprint.route('/aboutus', methods=['GET', 'POST'])
def aboutus():
    return render_template("pages/aboutus.html")

@blueprint.route('/icons', methods=['GET', 'POST'])
def icons():
    return render_template("pages/icons.html")

@blueprint.route('/dashapp1')
def dashapp1():
    return redirect(url_for('DashExample_blueprint.app1_template'))

def create_plot(city_news, wd_list):
    translator = Translator()

    if dataset == 'olympics':
        ## Add which site, words and city you want
        site_city = city_news.split(',')
        words = wd_list.split(',')

        plot_data = []
        for i in range(len(site_city)):
            site, city = site_city[i].split('_')
            site = site.strip()
            city = city.strip()
            for wd in words:
                wd = wd.strip()
                qu_wd = wd if lang_dict[site] == 'en' else translator.translate(wd, src='en', dest='pt').text
                plot_values = []
                for year in range(2004, 2020):
                    if str(year) in word_corpus[site][city]:
                        norm_val = word_corpus[site][city][str(year)][qu_wd] \
                            if qu_wd in word_corpus[site][city][str(year)] else 0
                    else:
                        norm_val = 0
                    plot_values.append(round(norm_val, 4))

                plot_data.append(go.Scatter(x=list(range(2004, 2020)), y=plot_values,
                                            mode='lines', name='%s_%s_%s' % (site, city, wd)))
    else:
        ## Add which site, words and city you want
        sites = city_news.split(',')
        words = wd_list.split(',')

        plot_data = []
        for site in sites:
            site = site.strip()
            for wd in words:
                wd = wd.strip()
                qu_wd = wd if lang_dict[site] == 'en' else translator.translate(wd, src='en', dest='es').text
                plot_values = []
                for year in range(2004, 2020):
                    if str(year) in word_corpus[site]:
                        norm_val = word_corpus[site][str(year)][qu_wd] \
                            if qu_wd in word_corpus[site][str(year)] else 0
                    else:
                        norm_val = 0
                    plot_values.append(round(norm_val, 4))

                plot_data.append(go.Scatter(x=list(range(2004, 2020)), y=plot_values,
                                            mode='lines', name='%s_%s' % (site, wd)))

    graphJSON = json.dumps(plot_data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

@blueprint.route('/select', methods=['GET', 'POST'])
def upload_dataset():
    global word_cos, dataset

    dataset = request.args['dataset']
    if dataset == 'olympics':
        word_corpus = json.load(open('data/word_corpus_caio.json', 'r', encoding='utf-8'))
    else:
        word_corpus = json.load(open('data/word_corpus_daniela.json', 'r', encoding='utf-8'))

    return jsonify(out=dataset)

@blueprint.route('/line', methods=['GET', 'POST'])
def add_word_trends():
    city_news = request.args['city_news']
    wd_list = request.args['wd_list']
    graphJSON = create_plot(city_news, wd_list)

    return graphJSON


@blueprint.route('/topic', methods=['GET', 'POST'])
def add_topic_modeling():
    global dataset
    news_pub = request.args['news_pub']
    num_k = request.args['num_k'].strip()
    print(news_pub, num_k)

    data_all = []
    json_dict = {}
    df_stopwords = []
    site = ''
    if dataset == 'olympics':
        site, city = news_pub.split('_')
        site, city = site.strip(), city.strip()
        json_dict = json.load(open('data/%s_%s_allnews.json' % (city, site), 'r'))
        df_stopwords = get_stopwords_caio(lang_dict[site])
    elif dataset == 'euro':
        site = news_pub.strip()
        json_dict = json.load(open('data/%s_allnews.json' % (site), 'r'))
        df_stopwords = get_stopwords_daniela(lang_dict[site])

    for key in json_dict:
        data_all.append(json_dict[key]['text'])

    ## Change to True or False to have these in the modeling or not
    use_bigrams = True
    use_trigrams = True
    lemmatize = False

    if lang_dict[site] == 'en':
        nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    elif lang_dict[site] == 'es':
        nlp = spacy.load("es_core_news_md", disable=['parser', 'ner'])
    else:
        nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])


    if num_k == 'auto':
        topic_model, corpus, id2word, coherence_values, max_i = topic_model_lda_auto(data_all, lemmatize,
                                                                                use_bigrams, use_trigrams, df_stopwords,
                                                                                nlp)
    else:
        topic_model, corpus, id2word = topic_model_lda_k(data_all, lemmatize, use_bigrams, use_trigrams, df_stopwords, nlp, int(num_k))

    vis = pyLDAvis.gensim.prepare(topic_model, corpus, id2word)
    if dataset == 'olympics':
        pyLDAvis.save_html(vis, 'data/%s_%s_%s.html' % (site, city, num_k))
        lda_html = open('data/%s_%s_%s.html' % (site, city, num_k), 'r', encoding='utf-8').read()
    elif dataset == 'euro':
        pyLDAvis.save_html(vis, 'data/%s_%s.html' % (site, num_k))
        lda_html = open('data/%s_%s.html' % (site, num_k), 'r', encoding='utf-8').read()

    return jsonify({'out':lda_html})
