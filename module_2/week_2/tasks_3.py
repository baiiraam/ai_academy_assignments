# Getting emails

import requests
from bs4 import BeautifulSoup

comments_url = "https://jsonplaceholder.typicode.com/comments"

res = requests.get(comments_url)
data = res.json()

for row in data:
    print(row.get("email", "no email"))


# Getting user names, emails, cities

users_url = "https://jsonplaceholder.typicode.com/users"

res = requests.get(users_url)

data = res.json()

for row in data:
    print(row["username"].split(".")[0], " ", row["email"], " ", row["address"]["city"])


url = "https://books.toscrape.com"
res = requests.get(url)
soup = BeautifulSoup(res.content, "html.parser")

# Find all book articles
books = []
for article in soup.find_all("article", class_="product_pod"):
    # Get title from the 'title' attribute of the <a> tag inside <h3>
    title_tag = article.find("h3").find("a")
    title = title_tag.get("title")

    # Get price (remove the £ symbol and convert to float if needed)
    price_tag = article.find("p", class_="price_color")
    price = price_tag.text

    books.append({"title": title, "price": price})

# Display all books
for idx, book in enumerate(books, 1):
    print(f"{idx}. {book['title']}: {book['price']}")

print(f"\nTotal books found: {len(books)}")


# filter those less than 50

filtered_books = []
for article in soup.find_all("article", class_="product_pod"):
    # Get title from the 'title' attribute of the <a> tag inside <h3>
    title_tag = article.find("h3").find("a")
    title = title_tag.get("title")

    # Get price (remove the £ symbol and convert to float if needed)
    price_tag = article.find("p", class_="price_color")
    price = price_tag.text[1:]

    if float(price) < 50:
        filtered_books.append({"title": title, "price": price})

for filtered_book in filtered_books:
    print(filtered_book["title"], filtered_book["price"])
