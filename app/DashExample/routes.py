from . import blueprint
from flask import render_template
from flask_login import login_required
from DashApp import Dash_App1


@blueprint.route('/app1')
def app1_template():
    return render_template('layouts/default.html', content=render_template('app1.html', dash_url=Dash_App1.url_base))
