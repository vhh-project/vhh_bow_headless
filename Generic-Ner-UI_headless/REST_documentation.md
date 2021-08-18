# Headless BOW REST Services

### GET api's
* http://localhost:8000/bow/main/health
* http://localhost:8000/bow/main/get_all_files
* http://localhost:8000/bow/main/get_json/id-1&id-2&id-3
* http://localhost:8000/bow/main/id-1/result.pdf
* http://localhost:8000/bow/main/id-1/original


### POST api's
* http://localhost:8000/bow/main/upload + post data
* http://localhost:8000/bow/main/ocr_data.zip + post data
* http://localhost:8000/bow/main/delete_paths + post data

-> add json examples/file response
-> workflow
    1. upload 2. receive id -> get json -> wait status to be == .FINISHED

    response notification when task_id finished
    -> requires interface on other host
    -> notify_from_BOW_2_mmsi service
    -> json ID - state finished

## cURL GET examples

#### BOW health check
```console
curl -X GET http://localhost:8000/health
```

```json
{"status": "up"}
```


#### get JSON file list
```console
curl -X GET http://localhost:8000/bow/main/get_all_files
```
```json
[
    {"task_id": "9d1f8620-613a-470f-8b92-83f02ade8d73", "user_id": "", "file_name": "ocr_testfile.png", "language": "ENG", "status": 3, "error": null, "upload_date": "2021-07-15T07:28:04.180Z"}, 
    {"task_id": "9bbcd3d3-eb77-4e13-a68e-8552261271fc", "user_id": "", "file_name": "ocr_testfile.png", "language": "ENG", "status": 3, "error": null, "upload_date": "2021-07-14T13:36:09.117Z"}
]
```

#### get JSON data for file-s (GET)
```console
curl -X GET http://localhost:8000/bow/main/get_json/e397385a-c9e3-4560-bb7f-3fe369aa0d27&f0ea12e0-4672-4004-aa89-d4048167023e&dd&6d0764c3-bcee-4a55-b583-8b98a63f0993
```
```json
{
    "9f7e2a9c-4b7a-4a8c-996b-7b9e0121670b": {
        "file_name": "ocr_testfile",
        "file_name_original": "ocr_testfile.png",
        "pages": [...],
        "entitystats": [...],
        "page_lengths": [
            3
        ]
    },
    "invalid_id-123456": {
        "error": "id not found"
    },
        "6d0764c3-bcee-4a55-b583-8b98a63f0993": {
        "file_name": "test_double",
        "file_name_original": "test_double.pdf",
        "pages": [...],
        "entitystats": [...],
        "page_lengths": [
            72,
            168
        ]
    }
}
```

*
#### get PDF file
```console
curl -i -X GET http://localhost:8000/bow/main/39ffea6e-b41f-4ad9-a752-48397e620162/result.pdf --output out.pdf
```
```console
Content-Type: application/pdf
```

#### get original image file
```console
curl http://localhost:8000/bow/main/52621f5e-17bf-432b-b62a-a1c49a3d2c84/original --output original.png
curl http://localhost:8000/bow/main/52621f5e-17bf-432b-b62a-a1c49a3d2c84/original > original.png
```
```json
Content-Type: image/png
```

#### cURL POST examples

#### upload file command 
```console
curl -i -X POST \
    -F "file=@/Users/josefweber/Desktop/removefolder/test_double.pdf" \
    -F "name=ocr_testfile.png" \
    -F "language=ENG" \
    -F "bucket=''" \
    http://localhost:8000/bow/main/upload
```
```json
{"f391e8fc-b76d-4746-9317-9a66713f246d": {"username": "", "file_name": "test_double.pdf", "language": "ENG"}}
```

#### get ZIP file-s data
```console
curl -X POST \
    -F "paths=52621f5e-17bf-432b-b62a-a1c49a3d2c84" \
    -F "paths=6d0764c3-bcee-4a55-b583-8b98a63f0993" \
    -L http://localhost:8000/bow/main/ocr_data.zip -o out.zip
```
```json
Content-Type: application/zip
```

#### remove file-s (run)
```console
curl -i -X POST \
    -F "paths=<id_1>" \
    -F "paths=<id_2>" \
    http://localhost:8000/bow/main/delete_paths
curl -i -X POST \
    -F "paths=9bbcd3d3-eb77-4e13-a68e-8552261271fc" \
    -F "paths=9ceec770-0b99-4324-bc15-172a2c1229ff" \
    -F "paths=invalid_id-123456" \
    http://localhost:8000/bow/main/delete_paths
```
```json
{
    "success": [
        {"9bbcd3d3-eb77-4e13-a68e-8552261271fc": true}, 
        {"9ceec770-0b99-4324-bc15-172a2c1229ff": true}, 
        {"invalid_id-123456": false}
    ]
}
```

#### URL patterns 
https://stackoverflow.com/questions/33555195/django-urlpattern-with-infinite-number-of-parameters