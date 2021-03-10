# import libraries
#https://appliedmachinelearning.blog/2017/08/28/topic-modelling-part-1-creating-article-corpus-from-simple-wikipedia-dump/
import pickle
import requests
from urllib.request import urlopen
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def get_url():
    """ Get the url to scrap

    """
    print("Enter the URL to scrape")
    url = input()
    return url


# crawling website
def get_weblinks(url):
    try:
        html_page = urlopen(url)
        pages_soup = BeautifulSoup(html_page.read(), features='html.parser')  # BeautifulSoup object
        total_pages = []

        for link in pages_soup.find_all('a', href=True):
            if link.get('href') not in total_pages:
                total_pages.append(link.get('href'))
    except:
        print("An exception occured")

    return total_pages


def extract_weblinks(url, links):
    # Extracted  5,885 feature articles links
    main_url = url
    parse_url = urlparse(main_url)
    # print(parse_url)
    base_url = parse_url[0] + '://' + parse_url[1]
    # print(base_url)
    specific_url = []
    total_links = links
    for url in total_links[78:5981]:  #
        if url not in ["NULL", "_blank", "None", None, "NoneType", '#', ':']:
            if url[0] == "/":
                url = url[1:]
            if base_url in url:
                if base_url == url:
                    pass
                if base_url != url and "https://" in url:
                    url = url[len(base_url) - 1:]

            if "http://" in url:
                specific_url = url
            elif "https://" in url:
                specific_url = url
            else:
                specific_url.append(base_url + url)

        else:
            pass

    with open('E:/Sharpest_Mind/WikipediaCitation/notebooks/specific_url.txt', 'wb') as fp:
        pickle.dump(specific_url, fp)
    return specific_url


def get_text(url_list):
    scrapped_text = []
    file_name = 'E:/Sharpest_Mind/WikipediaCitation/notebooks/all_text.txt'
    with open(file_name, 'w', encoding='utf-8') as outfile:
        for url in url_list:
            website = requests.get(url, timeout=5, allow_redirects=False)
            soup = BeautifulSoup(website.content)
            text = [''.join(s.findAll(text=True)) for s in soup.findAll('p')]
            scrapped_text = [x.replace('\n', '') for x in text]
        for item in scrapped_text:
            print(item, file=outfile)
    return scrapped_text


