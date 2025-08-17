import requests
proxyip = "http://smart-b0ibmkjy90uq_area-US_state-Northcarolina_life-15_session-0Ve35bhsUr:sU8CQmV8LDmh2mXj@proxy.smartproxy.net:3120"
url = "https://api.ip.cc"
proxies={
    'http':proxyip,
    'https':proxyip,
}
data = requests.get(url=url,proxies=proxies)
print(data.text)