import requests
from bs4 import BeautifulSoup
import time

def scrape_amazon_books(query):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    }
    url = f"https://www.amazon.com/s?k={query.replace(' ', '+')}"
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f"Failed to fetch Amazon page. Status code: {response.status_code}")
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    books = []
    results = soup.find_all("div", {"data-component-type": "s-search-result"})

    for item in results:
        try:
            title = item.find("span", class_="a-text-normal").text.strip()
            price_elem = item.find("span", class_="a-price-whole")
            price = f"${price_elem.text.strip()}" if price_elem else "Price not available"
            link = f"https://www.amazon.com{item.find('a', class_='a-link-normal')['href']}"
            image = item.find("img", class_="s-image")["src"]
            books.append({"title": title, "price": price, "link": link, "image": image})
        except Exception as e:
            print(f"Error parsing item: {e}")
            continue

    time.sleep(1)
    return books
