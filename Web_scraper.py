import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
from dotenv import load_dotenv

def proxy_request(url, proxy_url, username, password):
    proxies = {
        "http": proxy_url,
        "https": proxy_url,
    }
    auth = (username, password)
    response = requests.get(url, proxies=proxies, auth=auth)
    response.raise_for_status()  # Raise an HTTPError for bad responses
    return BeautifulSoup(response.text, 'html.parser')

def scrape_reviews(base_url, num_pages, proxy_url, username, password):
    data = []
    for page_num in range(num_pages):
        page_url = f'{url}?page={page_num}'
        try:
            soup = proxy_request(page_url, proxy_url, username, password)
        except requests.RequestException as e:
            print(f"Error fetching page {page_num + 1}: {e}")
            continue
        
        reviews = soup.find_all("article", class_="-pvs -hr _bet")
        for review in reviews:
            review_text = review.find("p", class_="-pvs")
            if review_text:
                data.append({'review': review_text.text.strip()})

    return data

def save_to_csv(data, filename):
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False)

if __name__ == "__main__":
    url = 'https://www.jumia.co.ke/catalog/productratingsreviews/sku/AQ014DR0PDU7GNAFAMZ/'
    num_pages = 5
    
    # Load environment variables from the .env file
    load_dotenv()

    # Load credentials from environment variables
    username = os.getenv('PROXY_USERNAME')
    password = os.getenv('PROXY_PASSWORD')
    country = 'DE'
    city = 'munich'
    
    # Construct the proxy URL
    proxy_url = f'http://customer-{username}-cc-{country}-city-{city}:{password}@pr.oxylabs.io:7777'
    
    # Scrape reviews and save to CSV
    reviews_data = scrape_reviews(url, num_pages, proxy_url, username, password)
    save_to_csv(reviews_data, "reviews.csv")
    print(f"Saved {len(reviews_data)} reviews to 'reviews.csv'")
