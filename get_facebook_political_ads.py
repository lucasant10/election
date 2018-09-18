import requests
import requests, zipfile, io
import time

import shutil
import zlib
import base64
import brotli
import json
import pprint
from random import randint
from time import sleep
import datetime
import os

folder_name = datetime.datetime.today().strftime('%Y-%m-%d')

if not os.path.isdir ('data/'+ folder_name):
    os.mkdir ('data/'+ folder_name)

page_number = 1

params = {
    '__a':1,
    '__be':1,
    '__dyn':'7xe6Fo4OQ5E5mWyUhxPLFwn84a2i5U4e1Fx-ewXwHxW1qwRzEeUhwmU2JwgEhw9-15w5VCK1awcG2y6U7m789U7W3a2W2y11xmczU4-0i2V8zway1aw',
    '__pc':'PHASED:DEFAULT',
    '__req':'1',
    '__rev':4222301,
    '__spin_b':'trunk',
    '__spin_r': 4222301,
    '__spin_t': int(time.time()),
    '__user': '100000447281094', # change this to your Facebook ID
    'fb_dtsg':'AQEZVXQAdXMT:AQHZOEcasGji',
    'jazoest':'2658169908688816510088778458658172907969999711571106105',
    'ph':'C3',
    'q' : [{"user":"100000447281094","page_id":"n6p2tz","posts":"gwlwW1sibG9nZ2VyOlBvbGl0aWNhbEFkQXJjaGl2ZUwFGfBMQ29uZmlnIix7ImVudGl0eSI6InBhZ2luYXRpb25fbmV4dF9idXR0b24iLCJldmVudCI6ImNsaWNrIn0sMTUzNDUzNzcwNDQ5NCwwXSz+bgCWbgAMNzMwNf5uAKJuABAxMDg0MwluTHRpbWVfc3BlbnRfYml0X2FycmF5ITlcdG9zX2lkIjoibjZwMnR6Iiwic3RhcnRfATAIIjoxKSUMNjQ3LAUqCTdIOlsxMDQ4NTc3LC04Mzg4NjA4XQkfGGxlbiI6NjQJDRBzZXEiOg0MHGN1bSI6MTd9NXYQMTEwNTIJmv7kAYrkARAxMzU5Of5uAKZuAAw2ODc0/m4Aom4ADDI4MTQNbixnazJfZXhwb3N1cmUh3ARpZGEX8HtmaWVyIjoiQVQ1ZmFIdVRYS3I5MGlRd1FLbDFTSFZDTGdlbGZkQWlCVENNMzZJUG1vMEdwenFJekNualpuTFIxWjNiS3NoY2tBaDNReWY1bkJLOFR4MjR3ZGFCeElkNzU5QTE4LXdwdmNIbXFZalJXanltekEiLCJoYXNoBXg0Nl9tZGRINXdWdDZrYmZ9eww2OTA5LZdBb+KfAgg3MTFJc1GfODQ5NjYzOSwzNzc0ODczNmKeAgA1CTdFngQzNFmeIDc1ODc0LDBdXQ==","snappy":'true',"trigger":"gk2_exposure","send_method":"ajax","snappy_ms":3}],
    'ts': int(time.time()) #new timestamp over new requests
}

header = {
    'Content-Type': 'application/x-www-form-urlencoded',
    'Origin': 'https://www.facebook.com',
    'Host': 'www.facebook.com',
    'Accept': '*/*',
    'Connection': 'keep-alive',
    'Accept-Language': 'en-US,en;q=0.5',
    'TE': 'Trailers',
    'Cookie': 'fr=0YUwvDU4oCJh2gRfi.AWWZqfJU3IZs3RGbV-06KTvy0_o.Ba0CZ8.RC.Ftu.0.0.BbdyAZ.AWVgPbxr; sb=eybQWlWo5B09VMeCyoAF3g6-; datr=libQWkQS2cjsSieuQrAmNQ9-; wd=1280x426; c_user=100000447281094; xs=44%3AQ8UOgSaGD18GQw%3A2%3A1532883055%3A20735%3A7694; pl=n; spin=r.4222301_b.trunk_t.1534533655_s.1_v.2_; act=1534534902960%2F9; pnl_data2=eyJhIjoiQmlnUGlwZS9pbml0IiwiYyI6IlhBZHNQb2xpdGljYWxBZEFyY2hpdmVDb250cm9sbGVyIiwiYiI6ZmFsc2UsImQiOiIvYWRzL2FyY2hpdmUvIiwiZSI6W119',
    'Accept-Encoding': 'gzip, deflate, br',
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_13_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/11.1.2 Safari/605.1.15',
    'Referer': 'https://www.facebook.com/ads/archive/?active_status=all&ad_type=political_and_issue_ads&country=BR&q=a',    
}

def get_ad_insights (ad_archive_id):
    api = "https://www.facebook.com/ads/archive/async/insights/?ad_archive_id=" + ad_archive_id + "&dpr=1"

    r = requests.post(api, headers=header, timeout=10, data=params)

    # data in brotli compression
    decompressed_data = (brotli.decompress(r.content)).decode ('utf-8')
    decompressed_data = decompressed_data.replace ('for (;;);', '')

    parsed = (json.loads(decompressed_data))

    print (parsed)

def get_ad_archive (next_page_token = None):
    global page_number
    global folder_name

    # get the next page
    if next_page_token:
        params ['page_token'] = next_page_token

    api = "https://www.facebook.com/ads/archive/async/search_ads/?q=a&count=30&active_status=all&type=political_and_issue_ads&country=BR&dpr=1"
    

    r = requests.post(api, headers=header, timeout=10, data=params)

    # data in brotli compression
    decompressed_data = (brotli.decompress(r.content)).decode ('utf-8')
    decompressed_data = decompressed_data.replace ('for (;;);', '')

    parsed = (json.loads(decompressed_data))
    
    # save json on file
    json_file = open ('data/'  + folder_name  + '/' + str(page_number) + '.json', 'w')
    json_file.write (json.dumps(parsed, indent=4, sort_keys=True))
    json_file.close()

    # wait for new request - 
    sleep(randint(10,100))

    print ('Crawling page ', page_number)
    page_number += 1
    if not parsed['payload']['isResultComplete']:
        get_ad_archive (parsed['payload']['nextPageToken'])



if __name__ == "__main__":
    get_ad_archive ()