from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
import random
import os
from time  import sleep
import numpy as np
import pandas as pd

def login(browser):
    # Mở website
    browser.get("https://mbasic.facebook.com/")
    sleep(random.randint(3,5))
    acc = browser.find_element(By.ID, "m_login_email")
    acc.send_keys("ptbao1505@gmail.com") 
    passw = browser.find_element(By.NAME, "pass")
    passw.send_keys("thuchanh12345*")
    passw.send_keys(Keys.ENTER)
    ok = browser.find_element(By.CLASS_NAME,"bo")
    ok.send_keys(Keys.ENTER)
    sleep(random.randint(3,5))

def get_link_post_in_page(browser):
    #Vào page KTX ĐHQG Confessions
    browser.get("https://mbasic.facebook.com/KTXDHQGConfessions?v=timeline&lst=100091813722701%3A100064552577559%3A1681549024&\
            eav=AfYkO3gxpz4E5Ut5b50h5cTSzphkcN5XrO8mqijuRmGrQJrzkI8a0klc0JSh0qAZfnM&paipv=0")
    elems = browser.find_elements(By.CSS_SELECTOR,"footer > div:nth-child(2) > a:nth-child(7)")
    result = [elem.get_attribute("href") for elem in elems]
    return result

def crawl_content_cmt(browser,post_link):
    elems = browser.find_elements(By.CSS_SELECTOR,"div > p") #Content trong post
    content = []
    if (elems):
        for elem in elems:
            content.append(elem.text)
    else:
        bgs = browser.find_element(By.CSS_SELECTOR,"div > div > div > div > div > div > span") #Content trong background
        content.append(bgs.text)

    list_cmt = []
    name = browser.find_elements(By.CSS_SELECTOR,"div > h3 > a")
    #if name:
        #cmt = browser.find_elements(By.CSS_SELECTOR,"div > div.dy")
            #cmt = browser.find_elements(By.CSS_SELECTOR,"div > div.dv")
        #cmt = browser.find_elements(By.CSS_SELECTOR,"div > div.dz")
    if (len(name)):
        pick_id = browser.find_elements(By.XPATH,'//a[contains(@href, "comment/replies")]') #tìm tất cả cmt
        cmt = []
        for id in pick_id:
            takeid= id.get_attribute('href').split('ctoken=')[1].split('&')[0] #lấy id của từng cmt
            text_cmt = browser.find_element(By.XPATH,('//*[@id="' + takeid.split('_')[1] + '"]/div/div[1]')) #lấy phần bình luận của cmt đó
            cmt.append(text_cmt.text)

    #Lưu kết quả vào crawler_fb_cmt.txt
    with open(r'D:\Desktop\CS232\Lab1_Crawler\result\crawler_fb_cmt.txt', "a", encoding='utf-16') as myfile:
        myfile.write(f"--Crawl Post_Link : {post_link}" + os.linesep)
        myfile.write(f"Content------------------"+ os.linesep)
        for i in content:
            myfile.write(i + os.linesep)
        myfile.write(f"Comment------------------"+ os.linesep)
        if name:
            for a,b in zip(name,cmt):
                myfile.write(f"{a.text} : {b}" + os.linesep)
        myfile.write("-----------Done-----------"+ os.linesep)

def crawl_post(post_link,browser):
    #Truy cập đến từng bài post trong page
    for post in post_link:
        browser.get(post)
        sleep(random.randint(3,5))
        crawl_content_cmt(browser,post)
    
    
# Khai báo browser
browser = webdriver.Chrome(service =Service(r'D:\Desktop\CS232\chromedriver.exe'))
login(browser)
post_link = get_link_post_in_page(browser)
#print(f'Total of Page : {len(post_link)}')
#print(post_link)
crawl_post(post_link,browser)
sleep(random.randint(3,5))
browser.close()