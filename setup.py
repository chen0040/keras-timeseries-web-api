from setuptools import setup

setup(
    name='keras_timeseries_web',
    packages=['keras_timeseries_web'],
    include_package_data=True,
    install_requires=[
        'flask',
        'keras',
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)