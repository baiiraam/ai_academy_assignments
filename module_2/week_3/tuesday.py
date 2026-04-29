# from bs4 import BeautifulSoup

# html = """
# <html>
#     <body>
#     <h1>Python</h1>
#     <p>two</p>
#     <p>three</p>
#     <p class="price">234</p>
#     </body>
# </html>
# """


# soup = BeautifulSoup(html, "html.parser")
# print(soup.find("h1").text)

# for link in soup.find_all("p"):
#     print(link.text)

# print(soup.find("p", class_="price").text)












# import requests

# url = "1news.az"

# res = requests.get(url)

# res.status_code

# soup = BeautifulSoup(res.text, "html.parser")

# links = soup.find_all("a")

# for link in links:
#     title = link.get_text(strip=True)
#     href = link.get("href")
#     print(title, href)












# from urllib.parse import urljoin


# headers = {
#     "user-agent": "Mozilla"
# }

# res
# soup

# all_links = soup.find_all("a")

# results = []

# for item in all_links:
#     title = item.get_text()
#     href = item.get("href")

#     if title and href:
#         full_link = urljoin(url, href)

#         results.append({"title": title, "url": full_link})

# for news in results[:15]:
#     print(news["title"])
#     print(news["url"])
#     print("-"*50)



# url = "books.toscrape.com/"
# res
# soup
# books = soup.find_all("article", class_="product_pod")
# for book in books:
#     title = book.find("h3").find("a")["title"]
#     price = book.find("p")
# # this code should be inside moodle IG
# # use api if available
# # web scraping if allowed (probably not)

# import requests
# from bs4 import BeautifulSoup

# url = "https://books.toscrape.com/"
# res = requests.get(url)
# soup = BeautifulSoup(res.text, "html.parser")
# books = soup.find_all("article", class_="product_pod")
# for book in books:
#     title = book.find('h3').find('a')['title']
#     price = book.find('p', class_='price_color').text
#     print(title, price)




# import threading
# import time

# def task():
#     print("task start")
#     time.sleep(2)
#     print("task finish")
# thread=threading.Thread(target=task)
# thread.start()
# print("main program")
# thread.join()


# import threading
# import requests
# import time
# from time import perf_counter
# t1 = perf_counter()
# websites = [
#  "https://google.com",
#  "https://github.com",
#  "https://example.com"
# ]
# def check(url):
#  try:
#     r= requests.get(url, timeout=5)
#     time.sleep(2)
#     print(url, r.status_code)
#  except requests.RequestException:
#     print(url, "işləmir")
# threads = []
# for url in websites:
#  t = threading.Thread(target=check, args=(url,))
#  threads.append(t)
#  t.start()
# for t in threads:
#  t.join()
# t2 = perf_counter()
# print(t2-t1)



# import threading
# counter = 0
# def increment():
#     global counter
#     for _ in range(100000):
#         counter += 1
# t1 = threading.Thread(target=increment)
# t2 = threading.Thread(target=increment)
# t1.start()
# t2.start()
# t1.join()
# t2.join()
# print(counter)