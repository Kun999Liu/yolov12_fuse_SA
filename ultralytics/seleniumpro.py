# -*- coding: utf-8 -*-
# @Time    : 2025/9/8 17:58
# @Author  : Liu Kun
# @Email   : liukunjsj@163.com
# @File    : seleniumpro.py
# @Software: PyCharm

"""
Describe:
"""
# -------------------------- 导入库 --------------------------
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import json

# -------------------------- 配置 Chrome --------------------------
chrome_options = Options()
# chrome_options.add_argument("--headless")  # 调试时先注释掉
chrome_options.add_argument("--disable-gpu")
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--window-size=1920,1080")

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

# -------------------------- 打开页面 --------------------------
url = "https://caplos.aircas.ac.cn/#/ZZSJ"
driver.get(url)
wait = WebDriverWait(driver, 20)

time.sleep(3)  # 等待基础页面加载

# -------------------------- 定义抓取表格数据函数 --------------------------
def extract_table():
    """抓取当前页表格数据"""
    try:
        table_element = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, ".el-table__body")))
        rows = table_element.find_elements(By.CSS_SELECTOR, "tr")
        page_data = []
        for row in rows:
            cells = row.find_elements(By.CSS_SELECTOR, "td")
            row_data = [cell.text.strip() for cell in cells]
            if row_data:
                page_data.append(row_data)
        return page_data
    except:
        return []

# -------------------------- 自动滚动加载 --------------------------
for _ in range(5):  # 可根据数据量调整滚动次数
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
    time.sleep(2)

# -------------------------- 循环翻页抓取 --------------------------
all_data = []
header_saved = False

while True:
    page_data = extract_table()
    if page_data:
        # 保存表头（假设第一行是表头）
        if not header_saved:
            header = page_data[0]
            header_saved = True
            page_data = page_data[1:]
        all_data.extend(page_data)
    else:
        print("当前页未抓取到数据")

    # 尝试点击“下一页”按钮
    try:
        next_btn = driver.find_element(By.XPATH, "//button[contains(text(),'下一页')]")
        if "disabled" in next_btn.get_attribute("class"):
            print("已到最后一页")
            break
        next_btn.click()
        time.sleep(2)  # 等待下一页加载
    except:
        print("未找到下一页按钮或已到最后一页")
        break

# -------------------------- 保存为 CSV --------------------------
if all_data:
    df = pd.DataFrame(all_data, columns=header)
    df.to_csv("caplos_data.csv", index=False, encoding="utf-8-sig")
    print(f"已抓取 {len(all_data)} 行数据，保存为 caplos_data.csv")
else:
    print("未抓取到表格数据")

# -------------------------- 可选：抓取 JS JSON 数据 --------------------------
try:
    js_data = driver.execute_script("return window.__INITIAL_STATE__.data")
    if js_data:
        with open("caplos_data.json", "w", encoding="utf-8") as f:
            json.dump(js_data, f, ensure_ascii=False, indent=2)
        print("JSON 数据已保存为 caplos_data.json")
except:
    print("未找到页面 JS 数据对象")

# -------------------------- 关闭浏览器 --------------------------
driver.quit()

