from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import random
from time  import sleep
import numpy as np
import pandas as pd

# Declare the browser
browser = webdriver.Chrome(service =Service(r'D:\Desktop\CS232\chromedriver.exe'))

# Open the website
browser.get("https://scholar.google.com/citations?user=VoTCbXYAAAAJ&hl=en&oi=sra")
sleep(5)

while browser.find_element(By.ID,'gsc_bpf_more').is_enabled():
    browser.find_element(By.ID,'gsc_bpf_more').click()
    sleep(2)

all_papers = browser.find_elements(By.CLASS_NAME, "gsc_a_tr")

title,author,year,linkdetail,cited = [],[],[],[],[]

for paper in all_papers:
    title.append(paper.find_element(By.CSS_SELECTOR,"td.gsc_a_t > a").text)
    author.append(paper.find_element(By.CLASS_NAME,"gs_gray").text)
    year.append(paper.find_element(By.CLASS_NAME,"gsc_a_h").text)
    cited.append(paper.find_element(By.CLASS_NAME,"gsc_a_ac").text)
    linkdetail.append(paper.find_element(By.CSS_SELECTOR,"td.gsc_a_t > a").get_attribute("href"))
    sleep(random.randint(1,3))


data ={'Tittle':title, 'Authors':author, 'Year':year,'Cited':cited,'Link_detail':linkdetail}
pd.DataFrame(data).to_csv(r'D:\Desktop\CS232\paper.csv', encoding="utf-8")

sleep(5)
browser.close()