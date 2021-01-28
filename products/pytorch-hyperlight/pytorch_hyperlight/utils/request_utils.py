import requests

REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36"
}


def download_urls(url_list, file_name_list):
    for url, file_name in zip(url_list, file_name_list):
        result = requests.get(url, headers=REQUEST_HEADERS)
        with open(file_name, "wb") as f:
            f.write(result.content)
