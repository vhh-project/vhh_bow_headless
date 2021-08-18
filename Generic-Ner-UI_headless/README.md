# headless frontend app – BOW

### activate submodule 
```bash
git submodule init
git submodule update

cd Generic-Ner-UI_headless
git checkout master
git pull origin master
```

### run local (w/o docker)
```bash
pipenv run python manage.py makemigrations
pipenv run python manage.py migrate
pipenv run python manage.py runserver
```

### run with docker .yaml
```bash
docker-compose up --build generic-ner-ui_headless
```

### update submodule to latest commit (master)
https://stackoverflow.com/questions/5828324/update-git-submodule-to-latest-commit-on-origin

