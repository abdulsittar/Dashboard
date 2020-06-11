# -*- encoding: utf-8 -*-
"""
License: MIT
Copyright (c) 2019 - present AppSeed.us
"""

from flask_migrate import Migrate
from configs.config import config_dict
import os
import sys
from app import create_app, db

get_config_mode = os.environ.get('GENTELELLA_CONFIG_MODE', 'Debug')

try:
    config_mode = config_dict[get_config_mode.capitalize()]
except KeyError:
    sys.exit('Error: Invalid GENTELELLA_CONFIG_MODE environment variable entry.')

app = create_app(config_mode)
Migrate(app, db)

if __name__ == "__main__":
    app.run(debug = True)
