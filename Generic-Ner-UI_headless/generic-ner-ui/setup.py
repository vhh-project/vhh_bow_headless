from setuptools import find_packages, setup

setup(
    name='generic_ner_ui',
    version='1.0',
    packages=find_packages(exclude=('tests', 'docs')),
    scripts=['manage.py'],
    install_requires=[
        "django==3.1",
        "whitenoise==5.2.0",
        "loguru==0.5.2",
        "minio==6.0.0",
        "mysqlclient==2.0.1",
        "everett[yaml]==1.0.2",
        "social-auth-core==3.3.3",
        "social-auth-app-django==4.0.0",
        "django-decorator-include==3.0",
        "psycopg2-binary==2.8.6",
        "aio-pika==6.7.1",
        "channels==2.4.0",
        "gunicorn==20.0.4",
        "django-fixstaticurl==0.1.0",
        "django-middleware-global-request==0.1.2",
        "pillow==8.2.0",
    ],
    extras_require={
        "dev": [
            "django-livereload-server==0.3.2",
            "pytest",
            "pytest-cov",
            "pytest-flake8",
            "pytest-mypy",
            "pytest-xdist",
            "pytest-django",
            "flake8-import-order",
            "requests"
        ]
    }
)
