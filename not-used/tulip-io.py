import requests

def test_proxy(proxy):
    try:
        response = requests.get("http://httpbin.org/ip", proxies={"http": proxy, "https": proxy}, timeout=5)
        if response.ok:
            print(f"Proxy {proxy} is working. Your IP: {response.json()['origin']}")
        else:
            print(f"Proxy {proxy} returned status code {response.status_code}")
    except Exception as e:
        print(f"Error occurred while testing proxy {proxy}: {e}")

def main():
    proxy = "http://TP32684532:qtpbAOTc@208.195.168.150:65095"
    test_proxy(proxy)

if __name__ == "__main__":
    main()

