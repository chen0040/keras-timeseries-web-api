from setuptools import setup

setup(
    name='keras_timeseries',
    packages=['keras_timeseries'],
    include_package_data=True,
    install_requires=[
        'flask',
        'celery',
        'keras',
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)