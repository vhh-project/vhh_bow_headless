"""
Django settings for generic_ner_ui project.

Generated by 'django-admin startproject' using Django 2.1.4.

For more information on this file, see
https://docs.djangoproject.com/en/2.1/topics/settings/

For the full list of settings and their values, see
https://docs.djangoproject.com/en/2.1/ref/settings/
"""

import os
from typing import Sequence

from django.utils.log import DEFAULT_LOGGING

from generic_ner_ui.static import config

DEFAULT_LOGGING['handlers']['console']['filters'] = []
# Build paths inside the project like this: os.path.join(BASE_DIR, ...)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/2.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = '*962%+m+4usv+kvqyc-e%^@ymtbu75u7h&uc*!-kd6h)1^nfqi'

# SECURITY WARNING: don't run with debug turned on in production!
# DEBUG = True
DEBUG = os.environ.get("DEBUG", "TRUE") == "TRUE"

ALLOWED_HOSTS = ['*']

# Application definition

INSTALLED_APPS = [
    #'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.sites',
    'django.contrib.staticfiles',
    'social_django',
    'generic_ner_ui',
    'django_global_request',
]
MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'whitenoise.middleware.WhiteNoiseMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'django.contrib.sites.middleware.CurrentSiteMiddleware',
    'django_global_request.middleware.GlobalRequestMiddleware',
]

if DEBUG:
    INSTALLED_APPS.append("livereload")
    MIDDLEWARE.append("livereload.middleware.LiveReloadScript")

SITE_ID = 1

ROOT_URLCONF = 'generic_ner_ui.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [os.path.join(BASE_DIR, "templates")],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'generic_ner_ui.wsgi.application'

AUTHENTICATION_BACKENDS = (
    'django.contrib.auth.backends.ModelBackend',
)

# Database
# https://docs.djangoproject.com/en/2.1/ref/settings/#databases

### DEV settings for DB connection
# DATABASES = {
#     'default': {
#         'OPTIONS': {
#             'charset': 'utf8mb4',
#         },
#         'ENGINE': 'django.db.backends.mysql',
#         'NAME': config.database.schema,
#         'USER': config.database.user,
#         'PASSWORD': 'root1',
#         'HOST': '127.0.0.1', # 'localhost',
#         'PORT': 3308,
#     }
# }

DATABASES = {
    'default': {
        'OPTIONS': {
            'charset': 'utf8mb4',
        },
        'ENGINE': 'django.db.backends.mysql',
        'NAME': config.database.schema,
        'USER': config.database.user,
        'PASSWORD': config.database.password,
        'HOST': config.database.host,
        'PORT': config.database.port,
    }
}

# Password validation
# https://docs.djangoproject.com/en/2.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS: Sequence[str] = []

# Internationalization
# https://docs.djangoproject.com/en/2.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_L10N = True

USE_TZ = True

# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/2.1/howto/static-files/

STATIC_URL = '/static/'

STATICFILES_DIRS = [
    os.path.join(BASE_DIR, "static"),
]

FORCE_SCRIPT_NAME = os.environ.get('SCRIPT_NAME', '')

LOGIN_URL = ''
LOGIN_REDIRECT_URL = '/'
LOGOUT_REDIRECT_URL = '/'

if FORCE_SCRIPT_NAME:
    LOGIN_URL = f'{FORCE_SCRIPT_NAME}{LOGIN_URL}'
    LOGIN_REDIRECT_URL = f'{FORCE_SCRIPT_NAME}{LOGIN_REDIRECT_URL}'
    LOGOUT_REDIRECT_URL = f'{FORCE_SCRIPT_NAME}{LOGOUT_REDIRECT_URL}'

STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')
STATICFILES_STORAGE = 'whitenoise.storage.CompressedStaticFilesStorage'

SOCIAL_AUTH_POSTGRES_JSONFIELD = True

protocol = ''
host = ''
access_host = ''
realm = ''
LOGOUT_URL = ''

if host != access_host:
    protocol = "http"

SOCIAL_AUTH_URL_NAMESPACE = 'social'

pipeline_names = list(map(lambda p: p.name, config.pipelines.pipelines))
assert len(pipeline_names) == len(set(pipeline_names))

for i, pipe in enumerate(config.pipelines.pipelines):
    if i == 0:
        assert len(pipe.depends_on) == 0
        assert pipe.return_pages
    else:
        assert len(pipe.depends_on) != 0
        assert not pipe.return_pages