import os

from celery import Celery
from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from keras_timeseries.web.milk_timeseries_stateless_predictor import MilkStateless
from keras_timeseries.web import MilkStateful

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])


def make_celery(app):
    celery = Celery(app.import_name, backend=app.config['CELERY_RESULT_BACKEND'],
                    broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    return celery


app = Flask(__name__)  # create the application instance :)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379',
    CELERY_RESULT_BACKEND='redis://localhost:6379'
)
celery = make_celery(app)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

milkStateful = MilkStateful()
milkStateless = MilkStateless()

# milkStateful.test_run()
milkStateless.test_run()


@app.route('/')
def classifiers():
    return render_template('home.html')


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def store_uploaded_file(action):
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    # if user does not select file, browser also
    # submit a empty part without filename
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        return redirect(url_for(action,
                                filename=filename))


@app.route('/about', methods=['GET'])
def about():
    return 'about us'


@app.route('/milk_timeseries_stateless', methods=['GET'])
def milk_timeseries_stateless():
    return render_template('milk_timeseries_stateless.html', output=milkStateless.test_run())



@app.route('/milk_timeseries_stateful')
def milk_timeseries_stateful():
    return render_template('milk_timeseries_stateful.html', output=milkStateless.test_run())




if __name__ == '__main__':
    app.run(debug=True)
