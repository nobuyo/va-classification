import os
from bs4 import BeautifulSoup
import re
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2
import urllib.parse

# res = urllib2.urlopen("http://gensun.org/list_voiceart.html")
# html = res.read()
# soup = BeautifulSoup(html,"html.parser")
# base_url = "http://gensun.org/"
soup = BeautifulSoup(open('va.html'),"html.parser")
pages = soup.find("div", class_="pagenavi")


# page 1
for a in soup.find_all("a", class_='trimming'):
    act_url = a.get('href')
    # print(act_url)
    # act_url = base_url + a.get('href')
    res = urllib2.urlopen(act_url)
    act_page = BeautifulSoup(res.read(),"html.parser")
    # print(act_page.find_all("img"))
    for d in act_page.find_all('div', class_='imagebox'):
        print(d.a.img.get('src'))


# page 2 and more
for p in pages.find_all('a'):
    next_url = p.get('href')

    res = urllib2.urlopen(nexr_url)
    next_page = BeautifulSoup(res.read(),"html.parser")

    for a in next_page.find_all("a", class_='trimming'):
    act_url = a.get('href')
    res = urllib2.urlopen(act_url)
    act_page = BeautifulSoup(res.read(),"html.parser")
    for d in act_page.find_all('div', class_='imagebox'):
        print(d.a.img.get('src'))
