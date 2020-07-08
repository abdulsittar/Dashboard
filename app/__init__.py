# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_bcrypt import Bcrypt
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from importlib import import_module
# Import routing, models and Start the App
# from app import views, models
from DashApp import Dash_App1

from flask_bcrypt import Bcrypt

# Grabs the folder where the script runs.
basedir = os.path.abspath(os.path.dirname(__file__))
lm = LoginManager()
db = SQLAlchemy()
bc = Bcrypt()


# app = Flask(__name__, static_folder='static')
# app.config.from_object('app.configuration.Config')
# app = Flask(__name__, static_folder='base/static')
# for module_name in ():


def apply_themes(app):

    """
    Add support for themes.

    If DEFAULT_THEME is set then all calls to
      url_for('static', filename='')
      will modfify the url to include the theme name

    The theme parameter can be set directly in url_for as well:
      ex. url_for('static', filename='', theme='')

    If the file cannot be found in the /static/<theme>/ lcation then
      the url will not be modified and the file is expected to be
      in the default /static/ location
    """

    @app.context_processor
    def _generate_url_for_theme(endpoint, **values):
        if endpoint.endswith('static'):
            themename = values.get('theme', None) or \
                        app.config.get('DEFAULT_THEME', None)
            if themename:
                theme_file = "{}/{}".format(themename, values.get('filename', ''))
                if path.isfile(path.join(app.static_folder, theme_file)):
                    values['filename'] = theme_file
        return url_for(endpoint, **values)


def register_blueprints(app):
    for module_name in ('DashExample', 'home'):
        module = import_module('app.{}.routes'.format(module_name))
        app.register_blueprint(module.blueprint)


def create_app(config, selenium=False):
    app = Flask(__name__, static_folder='home/static')
    print(['%s' % rule for rule in app.url_map.iter_rules()]);
    app.config.from_object(config)
    db = SQLAlchemy(app)  # flask-sqlalchemy
    bc = Bcrypt(app)  # flask-bcrypt
    # flask-loginmanager
    lm.init_app(app)  # init the login manager

    if selenium:
        app.config['LOGIN_DISABLED'] = True
    register_blueprints(app)
    # apply_themes(app)
    app = Dash_App1.add_dash(app)
    return app


