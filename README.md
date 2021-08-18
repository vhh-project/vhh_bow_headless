# Headless BOW

### Services (specified in [docker-compose.yaml](docker-compose.yml))
- database
- postgres
- rabbitmq
- minio
- ocr-pipeline
- generic-ner-ui_headless
  
### run headless BOW setup :
`docker-compose up rabbitmq postgres database minio ocr-pipeline generic-ner-ui_headless`