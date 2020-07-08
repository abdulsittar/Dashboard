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
from app import  lm, db, bc

from spacy import displacy
from googletrans import Translator
import plotly
import plotly.express as px
import plotly.graph_objects as go
from .topic_modeling import *
import pandas as pd
import numpy as np
import json
import spacy
from flask_table import Table, Col, LinkCol
import pickle


word_corpus = []
dataset = ''
lang_dict = {'theguardian': 'en', 'bbc': 'en', 'dailymail': 'en', 'telegraph': 'en',
                 'globo': 'pt', 'estadao': 'pt', 'folha': 'pt', 'elmundo': 'es',
             'elpais':'es'}

euro_news = ['theguardian', 'dailymail', 'elmundo', 'elpais']


@blueprint.route('/')
def dashboard():
    return render_template('layouts/default.html',
                           content=render_template('pages/index.html'))

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
@blueprint.route('/<path>')
def index(path):
    #if not current_user.is_authenticated:
     #   return redirect(url_for('home_blueprint.login'))

    content = None

    try:

        return render_template('layouts/default.html',
                               content=render_template('pages/' + path))

    except:

        return render_template('layouts/auth-default.html',
                               content=render_template('pages/404.html'))


# Return sitemap
@blueprint.route('/sitemap.xml')
def sitemap():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'sitemap.xml')


@blueprint.route('/forceDirected', methods=['GET', 'POST'])
def forceDirected():
    return render_template('layouts/default.html',
                               content=render_template("pages/forceDirected.html"))


@blueprint.route('/aboutus', methods=['GET', 'POST'])
def aboutus():
    return render_template("pages/aboutus.html")


@blueprint.route('/icons', methods=['GET', 'POST'])
def icons():
    return render_template("pages/wordTrends.html")


@blueprint.route('/dashapp1')
def dashapp1():
    return redirect(url_for('DashExample_blueprint.app1_template'))


@blueprint.route('/word_trends', methods=['GET', 'POST'])
def word_trends():
    return render_template('layouts/default.html',
                               content=render_template("pages/word_trends.html"))


@blueprint.route('/topic_mod', methods=['GET', 'POST'])
def topic_mod():
    return render_template('layouts/default.html',
                               content=render_template("pages/topic_mod.html"))


@blueprint.route('/sent_anly', methods=['GET', 'POST'])
def sent_anly():
    return render_template("pages/sent_anly.html")


@blueprint.route('/select', methods=['GET', 'POST'])
def upload_dataset():
    global word_corpus, dataset, inv_corpus, text_corpus

    dataset = request.args['dataset']
    if dataset == 'olympics':
        word_corpus = json.load(open('data/word_json/word_corpus_caio.json', 'r', encoding='utf-8'))
        inv_corpus = json.load(open('data/inverse_json/inv_doc_caio.json', 'r', encoding='utf-8'))
        for news_names in ['london_bbc', 'rio_bbc', 'london_dailymail', 'rio_dailymail', 'london_telegraph',
                           'rio_telegraph', 'london_theguardian', 'rio_theguardian', 'london_globo', 'rio_globo',
                           'london_estadao', 'rio_estadao', 'london_folha', 'rio_folha']:
            text_corpus[news_names] = json.load(
                open('data/news_json/%s_allnews.json' % (news_names), 'r', encoding='utf-8'))
    else:
        word_corpus = json.load(open('data/word_json/word_corpus_daniela.json', 'r', encoding='utf-8'))
        inv_corpus = json.load(open('data/inverse_json/inv_doc_daniela.json', 'r', encoding='utf-8'))
        for news_names in ['dailymail', 'theguardian', 'elmundo', 'elpais']:
            text_corpus[news_names] = json.load(
                open('data/news_json/%s_allnews.json' % (news_names), 'r', encoding='utf-8'))

    return jsonify(out=dataset)


@blueprint.route('/line', methods=['GET', 'POST'])
def add_word_trends():
    city_news = request.args['city_news']
    wd_list = request.args['wd_list']
    print(word_corpus)
    graphJSON = create_bar_plot(city_news, wd_list)

    return graphJSON


word_corpus = []
inv_corpus = {}
text_corpus = {}
dataset = ''
lang_dict = {'theguardian': 'en', 'bbc': 'en', 'dailymail': 'en', 'telegraph': 'en',
             'globo': 'pt', 'estadao': 'pt', 'folha': 'pt', 'elmundo': 'es',
             'elpais': 'es'}

euro_news = ['theguardian', 'dailymail', 'elmundo', 'elpais']


def create_bar_plot(city_news, wd_list):
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

                plot_data.append(go.Bar(name='%s_%s_%s' % (site, city, wd), x=list(range(2004, 2020)), y=plot_values))
    else:
        ## Add which site, words and city you want
        sites = city_news.split(',')
        words = wd_list.split(',')

        print(sites, words)

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

                plot_data.append(go.Bar(x=list(range(2004, 2020)), y=plot_values, name='%s_%s' % (site, wd)))

    graphJSON = json.dumps(plot_data, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


def create_line_plot(city_news, wd_list):
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

        print(sites, words)

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

#@blueprint.route('/doc_display/<news_site>/<rank>', methods=['GET', 'POST'])
@blueprint.route('/doc_display/<news_site>/<rank>', methods=['GET', 'POST'])
def get_doc_display(news_site, rank):
    if dataset == 'olympics':
        city, site = news_site.split('_')
        title = text_corpus[city + '_' + site]['rank_' + rank]['title']
        text = text_corpus[city + '_' + site]['rank_' + rank]['text']
    else:
        site = news_site
        title = text_corpus[site]['rank_' + rank]['title']
        text = text_corpus[site]['rank_' + rank]['text']

    if lang_dict[site] == 'en':
        nlp = spacy.load("en_core_web_md")
    elif lang_dict[site] == 'es':
        nlp = spacy.load("es_core_news_md")
    else:
        nlp = spacy.load('pt_core_news_md')

    ## Add spacy AnnotationW
    proc_text = nlp(text)

    spacy_html = displacy.render(proc_text, style="ent")

    return render_template('pages/doc_display.html', title=title, spacy_html=spacy_html)


@blueprint.route('/trendtable', methods=['GET', 'POST'])
def get_trend_docs():
    city_news = request.args['city_news']
    wd_list = request.args['wd_list']
    year = request.args['year']

    # Declare your table
    class ItemTable(Table):
        news_site = Col('News Site')
        rank = Col('Search Rank')
        word = Col('Words')
        title = LinkCol('Title', 'home_blueprint.get_doc_display', url_kwargs=dict(news_site='news_site', rank='rank'), attr='title')
        date = Col('Date')

    class Item(object):
        def __init__(self, news_site, rank, word, title, date):
            self.news_site = news_site
            self.rank = rank
            self.word = word
            self.title = title
            self.date = date

    translator = Translator()
    table_items = []
    table_dict = {}

    if dataset == 'olympics':
        ## Add which site, words and city you want
        site_city = city_news.split(',')
        words = wd_list.split(',')

        for i in range(len(site_city)):
            site, city = site_city[i].split('_')
            site = site.strip()
            city = city.strip()
            if year in inv_corpus[site][city]:
                for wd in words:
                    qu_wd = wd if lang_dict[site] == 'en' else translator.translate(wd, src='en', dest='es').text
                    if qu_wd in inv_corpus[site][city][year]:
                        for doc in inv_corpus[site][city][year][qu_wd]:
                            unq_colm = city + '_' + site + '_' + doc.split('_')[1]
                            if unq_colm not in table_dict:
                                table_dict[unq_colm] = {'title': text_corpus[city + '_' + site][doc]['title'],
                                                        'date': text_corpus[city + '_' + site][doc]['date'],
                                                        'words': wd}
                            else:
                                table_dict[unq_colm]['words'] += ', ' + wd
    else:
        ## Add which site, words and city you want
        sites = city_news.split(',')
        words = wd_list.split(',')

        for site in sites:
            site = site.strip()
            if year in inv_corpus[site]:
                for wd in words:
                    wd = wd.strip()
                    qu_wd = wd if lang_dict[site] == 'en' else translator.translate(wd, src='en', dest='es').text
                    if qu_wd in inv_corpus[site][year]:
                        for doc in inv_corpus[site][year][qu_wd]:
                            unq_colm = site + '_' + doc.split('_')[1]
                            if unq_colm not in table_dict:
                                table_dict[unq_colm] = {'title': text_corpus[site][doc]['title'],
                                                        'date': text_corpus[site][doc]['date'], 'words': wd}
                            else:
                                table_dict[unq_colm]['words'] += ', ' + wd

    for unq_colm in table_dict:
        table_items.append(
            Item(unq_colm.strip('_' + unq_colm.split('_')[-1]), unq_colm.split('_')[-1], table_dict[unq_colm]['words'],
                 table_dict[unq_colm]['title'], table_dict[unq_colm]['date']))

    table = ItemTable(table_items, border=True)

    return jsonify({'table': table})


@blueprint.route('/topic', methods=['GET', 'POST'])
def add_topic_modeling():
    news_pub = request.args['news_pub']
    num_k = request.args['num_k'].strip()

    data_all = []
    json_dict = {}
    df_stopwords = []
    site = ''
    if dataset == 'olympics':
        site, city = news_pub.split('_')
        site, city = site.strip(), city.strip()
        json_dict = text_corpus[city + '_' + site]
        df_stopwords = get_stopwords_caio(lang_dict[site])
    elif dataset == 'euro':
        site = news_pub.strip()
        json_dict = text_corpus[site]
        df_stopwords = get_stopwords_daniela(lang_dict[site])

    for key in json_dict:
        data_all.append(json_dict[key]['text'])

    ## Change to True or False to have these in the modeling or not
    use_bigrams = True
    use_trigrams = False
    lemmatize = False

    if dataset == 'olympics' and os.path.exists('data/topic_models/%s_%s_%s.html' % (site, city, num_k)):
        lda_html = open('data/topic_models/%s_%s_%s.html' % (site, city, num_k), 'r', encoding='utf-8').read()
        # pickle_mod = pickle.load(open('data/topic_models/%s_%s_%s.pkl' % (site, city, num_k), 'rb'))
        # topic_model, corpus = pickle_mod['model'], pickle_mod['corpus']

    elif dataset == 'euro' and os.path.exists('data/topic_models/%s_%s.html' % (site, num_k)):
        lda_html = open('data/topic_models/%s_%s.html' % (site, num_k), 'r', encoding='utf-8').read()
        # pickle_mod = pickle.load(open('data/topic_models/%s_%s.pkl' % (site, num_k), 'rb'))
        # topic_model, corpus = pickle_mod['model'], pickle_mod['corpus']

    else:
        if lang_dict[site] == 'en':
            nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
        elif lang_dict[site] == 'es':
            nlp = spacy.load("es_core_news_md", disable=['parser', 'ner'])
        else:
            nlp = spacy.load('pt_core_news_sm', disable=['parser', 'ner'])

        if num_k == 'auto':
            topic_model, corpus, id2word, _, max_i = topic_model_lda_auto(data_all, lemmatize,
                                                                          use_bigrams, use_trigrams, df_stopwords,
                                                                          nlp)
            num_k = max_i + 2
        else:
            topic_model, corpus, id2word = topic_model_lda_k(data_all, lemmatize, use_bigrams, use_trigrams,
                                                             df_stopwords, nlp, int(num_k))

        vis = pyLDAvis.gensim.prepare(topic_model, corpus, id2word)
        if dataset == 'olympics':
            pyLDAvis.save_html(vis, 'data/topic_models/%s_%s_%s.html' % (site, city, num_k))
            pickle.dump({'model': topic_model, 'corpus': corpus},
                        open('data/topic_models/%s_%s_%s.pkl' % (site, city, num_k), 'wb'))
            lda_html = open('data/topic_models/%s_%s_%s.html' % (site, city, num_k), 'r', encoding='utf-8').read()
        elif dataset == 'euro':
            pyLDAvis.save_html(vis, 'data/topic_models/%s_%s.html' % (site, num_k))
            pickle.dump({'model': topic_model, 'corpus': corpus},
                        open('data/topic_models/%s_%s.pkl' % (site, num_k), 'wb'))
            lda_html = open('data/topic_models/%s_%s.html' % (site, num_k), 'r', encoding='utf-8').read()

    return jsonify({'lda_html': lda_html})


@blueprint.route('/topicdist', methods=['GET', 'POST'])
def add_topic_dist():
    global df_topic_assign
    news_pub = request.args['news_pub']
    num_k = request.args['num_k'].strip()

    data_all = []
    json_dict = {}
    site = ''
    if dataset == 'olympics':
        site, city = news_pub.split('_')
        site, city = site.strip(), city.strip()
        json_dict = text_corpus[city + '_' + site]
    elif dataset == 'euro':
        site = news_pub.strip()
        json_dict = text_corpus[site]

    for key in json_dict:
        data_all.append(json_dict[key]['text'])

    if dataset == 'olympics':
        pickle_mod = pickle.load(open('data/topic_models/%s_%s_%s.pkl' % (site, city, num_k), 'rb'))
        topic_model, corpus = pickle_mod['model'], pickle_mod['corpus']

    elif dataset == 'euro':
        pickle_mod = pickle.load(open('data/topic_models/%s_%s.pkl' % (site, num_k), 'rb'))
        topic_model, corpus = pickle_mod['model'], pickle_mod['corpus']

    all_ranks = [key for key in json_dict]
    all_dates = [json_dict[key]['date'] for key in json_dict]
    all_titles = [json_dict[key]['title'] for key in json_dict]
    df_topic_assign = get_topic_assignments(topic_model, corpus, all_titles, all_ranks, all_dates)

    topic_bardata = get_topic_bargraph(df_topic_assign)

    graphJSON = json.dumps(topic_bardata, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON


@blueprint.route('/topicdocs', methods=['GET', 'POST'])
def add_topic_docs():
    news_pub = request.args['news_pub']
    year = request.args['year']

    if dataset == 'olympics':
        news_site = news_pub.split('_')[1] + '_' + news_pub.split('_')[0]
    else:
        news_site = news_pub

    # Declare your table
    class ItemTable(Table):
        news_site = Col('News Site')
        rank = Col('Search Rank')
        topic = Col('Topic')
        perc = Col('% Contribution')
        title = LinkCol('Title', 'home_blueprint.get_doc_display', url_kwargs=dict(news_site='news_site', rank='rank'), attr='title')
        date = Col('Date')

    class Item(object):
        def __init__(self, news_site, rank, topic, perc, title, date):
            self.news_site = news_site
            self.rank = rank
            self.topic = topic
            self.perc = perc
            self.title = title
            self.date = date

    table_items = []

    year_tpc_assign = df_topic_assign[df_topic_assign['date'].apply(lambda dt: dt.split('/')[2] == year)]

    for tpc in range(0, int(max(year_tpc_assign['topic'])) + 1):
        sub_assign_tpc = year_tpc_assign[year_tpc_assign['topic'].apply(lambda tp: int(tp) == tpc)]
        print(sub_assign_tpc)
        for i, row in sub_assign_tpc.iterrows():
            table_items.append(
                Item(news_site, row[3].split('_')[1], int(tpc) + 1, round(float(row[1]), 4), row[2], row[4]))

    table = ItemTable(table_items, border=True)

    return jsonify({'table': table})

@blueprint.route('/sentiment', methods=['GET', 'POST'])
def add_sentiment():
    news_pub = request.args['news_pub']
    year = request.args['year']

    if dataset == 'olympics':
        news_site = news_pub.split('_')[1] + '_' + news_pub.split('_')[0]
    else:
        news_site = news_pub

    # Declare your table
    class ItemTable(Table):
        news_site = Col('News Site')
        rank = Col('Search Rank')
        topic = Col('Topic')
        title = LinkCol('Title', 'home_blueprint.get_doc_display', url_kwargs=dict(news_site='news_site', rank='rank'), attr='title')
        date = Col('Perc Contribution')

    class Item(object):
        def __init__(self, news_site, rank, topic, title, date):
            self.news_site = news_site
            self.rank = rank
            self.topic = topic
            self.title = title
            self.date = date

    table_items = []

    year_tpc_assign = df_topic_assign[df_topic_assign['date'].apply(lambda dt: dt.split('/')[2] == year)]

    for tpc in range(0, int(max(year_tpc_assign['topic'])) + 1):
        sub_assign_tpc = year_tpc_assign[year_tpc_assign['topic'].apply(lambda tp: int(tp) == tpc)]
        print(sub_assign_tpc)
        for i, row in sub_assign_tpc.iterrows():
            table_items.append(Item(news_site, row[3].split('_')[1], int(tpc) + 1, row[2], row[4]))

    table = ItemTable(table_items, border=True)

    return jsonify({'table': table})