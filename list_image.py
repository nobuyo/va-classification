import os
from bs4 import BeautifulSoup
import re
try:
    import urllib.request as urllib2
except ImportError:
    import urllib2
import urllib.parse
import scrapelib
from joblib import Parallel, delayed
from time import time

scr = scrapelib.Scraper(retry_attempts=2, retry_wait_seconds=5)

def download(image_url, save_dir):
    image_name = os.path.basename(image_url)
    try:
        if os.path.exists(save_dir+'/'+image_name):
            print('Pass: ' + image_url)
        elif image_url.count('gazo.galman.jp')!=0:
            print('Skip: ' + image_url)
        else:
            scr.urlretrieve(image_url, filename=save_dir+'/'+image_name)
            print('Sucs: ' + image_url)
    except Exception as e:
        print("Fail: %s %s" %(e , image_url))


base_url = "http://gensun.org/"

for act in open('add_list', 'r'):
    query = urllib.parse.urlencode({'q': act})
    act_url = base_url + '?' + query
    print(act_url)
    res = urllib2.urlopen(act_url)
    act_page = BeautifulSoup(res.read(),"html.parser")
    save_dir = (act_page.find('div', id='pan').text.split('>').pop().strip())
    print("======= %s ========" % save_dir)

    try:
        os.mkdir(save_dir)
        print("Created: " + save_dir)
    except Exception as e:
        print(e)

    for d in act_page.find_all('div', class_='imagebox'):
        image_url = d.a.img.get('src')
        Parallel(n_jobs=-1)( [delayed(download)(image_url, save_dir)] )

