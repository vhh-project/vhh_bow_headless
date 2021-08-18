import asyncio
import json
import zipfile
import itertools
from io import BytesIO
from pathlib import Path

from PIL import Image
import requests
from loguru import logger

from django.http.response import HttpResponseForbidden
from asgiref.sync import async_to_sync, sync_to_async
from django.http import HttpResponse, HttpResponseBadRequest
from django.http import JsonResponse

from channels.db import database_sync_to_async
from django.template import loader
from django.views.decorators.cache import cache_page, never_cache
from django.views.decorators.csrf import csrf_exempt


from generic_ner_ui.models import Run, Status, RunResult
from generic_ner_ui.settings import LOGOUT_URL
from generic_ner_ui.static import config
from generic_ner_ui.utils import redirect
from main.processing import download_minio_file, start_processing, \
    get_files_to_download_any, download_file_to_stream, update_pdf, \
    download_minio_file_data
from django.views.decorators.csrf import csrf_protect

from main.amqp import myuuid


@async_to_sync
async def get_json(request, ids):  # GET
    ids = ids.split("&")
    runs = await get_runs_by_ids_data(request, ids, force_result=False)

    context = {}
    for run, id in zip(runs, ids):
        if run is not False:
            if run.status == Status.FINISHED:
                pages = run.result_data.data["pages"]
                ner_types = gather_entity_types(pages)
                entity_stats = gather_entity_stats(pages)
                context[run.task_id] = {
                    "file_name": Path(run.file_name).stem,
                    "file_name_original": run.file_name,
                    "pages": [
                        transform_page(page, ner_types=ner_types)
                        for page in pages
                    ],
                    "entitystats": entity_stats,
                    "page_lengths": [
                        len(page['ner']['entries'])
                        for i, page in enumerate(pages)
                    ],
                }
            else:
                context[run.task_id] = {"error": "status not finished"}
        else:
            context[id] = {"error": "id not found"}
    return JsonResponse(context, safe=False)


def get_all_files(request):  # GET
    runs = Run.objects.filter()
    runs = list(reversed(runs))
    run_task_ids = []
    for r in runs:
        run_task_ids.append({"task_id": r.task_id,
                             "user_id": r.user_id,
                             "file_name": r.file_name,
                             "language": r.lang,
                             "status": r.status,
                             "error": r.error,
                             "upload_date": r.upload_date})
    return JsonResponse(run_task_ids, safe=False)


@csrf_exempt
def upload(request):  # POST
    logger.debug(f"request 'upload' {request.method}")
    if request.method == 'POST':
        if request.FILES is None and "url" not in request.POST:
            return HttpResponseBadRequest('No files attached.')
        language = request.POST["language"]
        bucket = request.POST["bucket"]

        if request.FILES is not None:
            files = request.FILES.getlist('file')
            logger.info(f"File upload with {len(files)} files.")
            logger.debug(f"{request.FILES}")
            context = {}
            for f in files:
                task_id = myuuid()
                context[task_id] = {'username': request.user.username,
                                    'file_name': f.name,
                                    'language': language}
                start_processing(request.user.username, f, f.name, language, bucket, task_id)
            return JsonResponse(context, safe=False)

        if "url" in request.POST:
            urls = request.POST.getlist("url")
            names = request.POST.getlist("name")
            logger.info(f"File upload with {len(urls)} urls.")
            for url, name in zip(urls, names):
                logger.info(f"Downloading {name} from onedrive via {url}")
                r = requests.get(url)
                r.raise_for_status()
                logger.info(f"Downloaded File: {r.status_code} , {len(r.content)}")
                f = BytesIO(initial_bytes=r.content)
                f.seek(0)
                start_processing(request.user.username, f, name, language, bucket)

    return HttpResponse(status=200)


def delete(request, id):
    try:
        run = get_run(request, id, expect_result=False)
        if not run:
            return JsonResponse({"success": "false"}, safe=False)
        else:
            run.delete()
            return JsonResponse({"success": "true"}, safe=False)
    except Exception as e:
        logger.exception(f"Failed to delete {id}, {e}")
        return JsonResponse({"success": "false"}, safe=False)


@csrf_exempt
def delete_files(request):  # POST
    ids = request.POST.getlist("paths")
    res = []
    for task_id in ids:
        try:
            run = get_run(request, task_id, expect_result=False)
            if not run:
                res.append({task_id: False})
            else: 
                run.delete()
                res.append({task_id: True})
        except Exception as e:
            logger.exception(f"Failed to delete {task_id}, {e}")
            res.append({task_id: False})
    return JsonResponse({"success": res}, safe=False)


@csrf_exempt
@async_to_sync
async def download_all(request):
    ids = request.GET.getlist("paths")
    if not ids:
        ids = request.POST.getlist("paths")
    runs = await get_runs_by_ids_data(request, ids)
    response = HttpResponse(content_type='application/zip')

    with zipfile.ZipFile(response, "w", zipfile.ZIP_STORED, False) as zip_file:
        paths = await asyncio.gather(*[get_files(run) for run in runs])
        for path, stream in itertools.chain.from_iterable(paths):
            zip_file.writestr(path, stream)

    return response


@csrf_exempt
@cache_page(24 * 3600 * 7)
@async_to_sync
async def image(request, id, page):
    run = await database_sync_to_async(get_run)(request, id)
    minio_path = run.result_prep["pages"][page]["preprocessing"]["minio"]
    image_stream = await download_file_to_stream(minio_path)
    img = Image.open(image_stream).convert('RGB')

    img.thumbnail((config.image.width, config.image.height))

    response = HttpResponse(content_type="image/jpeg")
    img.save(response, format="jpeg", quality=100, optimize=True, progressive=True)

    return response


@csrf_exempt
@never_cache
def download_pdf(request, id):  # GET
    run = get_run(request, id)
    minio_path = run.result_prep["output"]
    return download_minio_file(minio_path, content_type="application/pdf")


@csrf_exempt
@cache_page(24 * 3600 * 7)
def download_original(request, id):
    run = get_run(request, id, expect_result=False)
    minio_path = run.minio_path
    return download_minio_file(minio_path)


###
#  helpers
###


def get_run(request, id, expect_result=True):
    try:
        run = Run.objects.get(task_id=id)
        assert run.user_id == request.user.username
        assert not expect_result or run.status == Status.FINISHED
        return run
    except Exception:
        return False


def get_run_forcedata(request, id, expect_result=True):
    try:
        run = Run.objects.get(task_id=id)
        _ = run.result_data
        assert run.user_id == request.user.username
        assert not expect_result or run.status == Status.FINISHED
        return run
    except Exception:
        return False


async def get_files(run):
    logger.info(run)
    request, paths = get_files_to_download_any(run.result_data.data)
    logger.info(paths)
    request_json = json.dumps(request, ensure_ascii=False)
    request_json_path = f"{run.task_id}/{Path(run.file_name).stem}_ocr.json"
    results = [(request_json_path, request_json)]
    paths, minio_paths = zip(*paths)

    streams = await asyncio.gather(*[
        download_file_to_stream(path)
        for path in minio_paths
    ])
    streams = [s.read() for s in streams]
    results.extend(zip(paths, streams))
    return results


@sync_to_async
def get_runs_by_ids(request, ids, force_result=True):
    return [get_run(request, task_id, expect_result=force_result)
            for task_id in ids]


@sync_to_async
def get_runs_by_ids_data(request, ids, force_result=True):
    return [get_run_forcedata(request, task_id, expect_result=force_result)
            for task_id in ids]


def transform_page(page, ner_types):
    entities = page["ner"]["entities"]
    entries = page["ner"]["entries"]
    text = page["ocr"]["text"]
    page_width = page["preprocessing"]["width"]
    page_height = page["preprocessing"]["height"]

    for entity in entities:
        entity["entries"] = []

    position = 0
    complete_text = ""
    for entry in entries:
        end_position = position + len(entry["text"])
        entry["left_norm"] = round(entry["left"] / page_width * 100, 2)
        entry["width_norm"] = round(entry["width"] / page_width * 100, 2)
        entry["top_norm"] = round(entry["top"] / page_height * 100, 2)
        entry["height_norm"] = round(entry["height"] / page_height * 100, 2)
        if entry["correction"] == "":
            entry["correction"] = []
        else:
            s = entry["correction"].replace("[", "").replace("]", "").replace("'", "")
            es = s.split(",")
            entry["correction"] = es

        for entity in entities:
            if position <= entity["end"] and end_position >= entity["start"]:
                entity["entries"].append(entry)
        position = end_position + 1
        complete_text += entry["text"] + " "

    if complete_text:
        complete_text = complete_text[:-1]
    try:
        # TODO: 
        assert complete_text == text
    except Exception as e:
        logger.info(f"assert err: {e}")

    entity_groups = [{
        "name": group.name,
        "entity_types": group.entities,
        "entities": []
    } for group in config.entitygroups.groups]

    entity_groups.append({
        "name": "Without group",
        "entity_types": [],
        "entities": []
    })

    for entity in entities:
        entity["color_id"] = ner_types.index(entity["type"]) % 9
        found = False

        for entity_group in entity_groups:
            if entity["type"] in entity_group["entity_types"]:
                entity_group["entities"].append(entity)
                found = True

        if not found:
            entity_groups[-1]["entities"].append(entity)

    if len(entity_groups) == 1:
        entity_groups[0]["name"] = None

    page["ner"]["entity_groups"] = entity_groups

    return page


def gather_entity_types(pages):
    ner_types = []

    for group in config.entitygroups.groups:
        for entity_type in group.entities:
            if entity_type not in ner_types:
                ner_types.append(entity_type)

    for page in pages:
        for entity in page["ner"]["entities"]:
            if entity["type"] not in ner_types:
                ner_types.append(entity["type"])
    return ner_types


def gather_entity_stats(pages):
    data = dict()

    for i, page in enumerate(pages):
        for entity in page["ner"]["entities"]:
            _type = entity["type"]
            text = entity["value"]
            if (_type, text) not in data:
                data[(_type, text)] = {
                    "text": text,
                    "type": _type,
                    "count": 0,
                    "pages": []
                }
            data[(_type, text)]["count"] += 1
            if i not in data[(_type, text)]["pages"]:
                data[(_type, text)]["pages"].append(i)
    return list(data.values())
