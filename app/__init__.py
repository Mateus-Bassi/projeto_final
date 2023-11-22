from flask import Flask

app = Flask(__name__, template_folder='../templates', static_folder='../static')
app.config['SECRET_KEY'] = '123456789'

from app import routes
