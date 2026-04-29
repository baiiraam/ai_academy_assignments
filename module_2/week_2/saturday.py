# import requests
# from pprint import pprint

# url = "https://jsonplaceholder.typicode.com/posts"

# res = requests.get(url)

# pprint(res.json())

# if res.status_code == 200:
#     dataa = res.json()
#     print(dataa)
#     print(dataa[0]['title'])
# else:
#     print('oops')


# # post
# data = {
#     "a": 1,
#     "b": 2,
#     "c": 3
# }

# res = requests.post(url, json=data)
# print(res.status_code)
# print(res.json())

# # guid


# # query parameters
# s = {
#     " userId": 4
# }
# res = requests.get(url, params=s)
# print(res.json())


import requests

# url = "https://jsonplaceholder.typicode.com/users"

# res = requests.get(url)
# data = res.json()
# for record in data:
#     print(f"city: {record["address"]["city"]}, \twebsite: {record["website"]}, \tphone: {record["phone"]}")

from pprint import pprint

s = {"id": 4}


url = "https://jsonplaceholder.typicode.com/posts"

res = requests.get(url, s)
data = res.json()
# pprint(data)

if __name__ == "__main__":
    pprint(data[0]["body"])
