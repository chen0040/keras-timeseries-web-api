import os

from flask import Flask, request, redirect, url_for, render_template, flash
from werkzeug.utils import secure_filename
from keras_timeseries.library.recurrent import StatefulLSTM, StatelessLSTM
import pandas as pd

UPLOAD_FOLDER = './uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)  # create the application instance :)
app.config.from_object(__name__)  # load config from this file , flaskr.py

# Load default config and override config from an environment variable
app.config.from_envvar('FLASKR_SETTINGS', silent=True)
app.config.update(
    CELERY_BROKER_URL='redis://localhost:6379',
    CELERY_RESULT_BACKEND='redis://localhost:6379'
)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

milkStateful = StatefulLSTM()
milkStateless = StatelessLSTM()


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


def main():
    data_dir_path = '../demo/data'
    model_dir_path = '../demo/models/monthly-milk-production'
    data_file_path = os.path.join(data_dir_path, 'monthly-milk-production-pounds-p.csv')
    dataframe = pd.read_csv(filepath_or_buffer=data_file_path, sep=',')

    milkStateful.load_model(model_dir_path)
    milkStateless.load_model(model_dir_path)

    timeseries = dataframe.as_matrix(['MilkProduction']).T[0][0:43]  # 36 is the multiple of the batch size
    # milkStateful.test_run()
    milkStateless.test_run(timeseries)
    app.run(debug=True, use_reloader=False)


if __name__ == '__main__':
    main()
