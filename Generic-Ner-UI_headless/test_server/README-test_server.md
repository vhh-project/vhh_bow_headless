#### run setup
```console
### 1. start BOW
python manage.py collectstatic --noinput && python manage.py makemigrations generic_ner_ui && python manage.py migrate && gunicorn generic_ner_ui.wsgi -b 0.0.0.0:8000 --keep-alive 20

### 2. start target server
python test_server.py runserver 7000
http://127.0.0.1:7000/

### 3. upload file via BOW
curl -i -X POST \
    -F "file=@/Users/josefweber/Desktop/removefolder/ocr_testfile.png" \
    -F "name=ocr_testfile.png" \
    -F "language=ENG" \
    -F "bucket=''" \
    http://localhost:8000/main/upload

### 3.1 BOW response (immediate):
{"9d1f8620-613a-470f-8b92-83f02ade8d73": {"username": "", "file_name": "ocr_testfile.png", "language": "ENG"}}

### 4. response object sent via POST from BOW received by test_server @ bow_notification:
<QueryDict: {'task_id': ['9d1f8620-613a-470f-8b92-83f02ade8d73'], 'status': ['3']}>
attributes to be accessed eg. in django via:
request.POST["task_id"]
request.POST["status"]
```

#### test the notification api (curl & python)
```console
curl -X POST \
    -F "data=msg" \
    http://127.0.0.1:7000/bow_notification
```

```python
import requests
url = 'http://127.0.0.1:7000/bow_notification'
msg = {'task_id': run.task_id,
       'status': run.status}
res = requests.post(url, data=msg)
```


