# Copyright Peter Gagarinov.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests
import boto3
import validators
import io
from pathlib import PurePath

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
}


def split_s3_url(s3_url):
    s3_url = s3_url.strip()
    assert s3_url.startswith("s3://")
    bucket_key = s3_url[len("s3://") :]
    bucket_name, key = bucket_key.split("/", 1)
    return bucket_name, key


def copy_urls_to_files(url_list, file_name_list, **kwargs):
    if not isinstance(url_list, list):
        url_list = [url_list]
    if not isinstance(file_name_list, list):
        file_name_list = [file_name_list]
    for url, file_name in zip(url_list, file_name_list):
        data_as_bytes = load_url_or_path_as_bytes(url, **kwargs)
        with open(file_name, "wb") as f:
            f.write(data_as_bytes.getbuffer())


def load_url_or_path_as_bytes(image_url_or_path, s3_resource=None):
    image_url_or_path = str(image_url_or_path)
    if s3_resource is None:
        s3_resource = boto3.Session().resource("s3")
    image_url = image_url_or_path.strip()
    if image_url.startswith("s3://"):
        bucket_name, key = split_s3_url(image_url)
        image_obj = s3_resource.Object(bucket_name=bucket_name, key=key)
        image_as_bytes = image_obj.get()["Body"].read()
    elif validators.url(image_url):
        image_as_bytes = requests.get(image_url, stream=True).content
    else:
        with open(image_url, "rb") as f:
            image_as_bytes = f.read()
    image_as_bytes = io.BytesIO(image_as_bytes)
    return image_as_bytes


def copy_file_to_s3(file, s3_url, s3_resource=None):
    if isinstance(file, str) or isinstance(file, PurePath):
        with open(file, "rb") as f:
            copy_fileobj_to_s3(f, s3_url, s3_resource=s3_resource)
    else:
        copy_fileobj_to_s3(file, s3_url, s3_resource=s3_resource)


def copy_fileobj_to_s3(file_like_obj, s3_url, s3_resource=None):
    bucket_name, key = split_s3_url(s3_url)
    if s3_resource is None:
        s3_resource = boto3.Session().resource("s3")
    s3_resource.Bucket(bucket_name).Object(key).upload_fileobj(file_like_obj)
