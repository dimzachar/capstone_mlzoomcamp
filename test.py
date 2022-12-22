import requests

def main():
    url = "http://localhost:8080/2015-03-31/functions/function/invocations"
    # url = "use_invoke_url_aws_lambda/predict"
    data = {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/7/77/Pebbleswithquarzite.jpg/1280px-Pebbleswithquarzite.jpg"
    }

    result = requests.post(url, json=data).json()
    print(result)


if __name__ == "__main__":
    main()