import requests
from bs4 import BeautifulSoup
import os
import time
import csv

# UA伪装
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36',

}


def get_comment_content(comment):
    rate = 0
    for i in range(1, 6):
        temp = comment.find('span', class_='allstar' + str(10 * i) + ' rating')
        if temp is not None:
            rate = i
    if rate == 0:
        return None
    else:
        span = comment.find('span', class_='short')
        return span.get_text(), rate


def save_to_csv(comments):
    with open('./data/test.csv', 'a', newline='', encoding='utf-8-sig') as csvfile:
        writer = csv.writer(csvfile)
        for comment in comments:
            if get_comment_content(comment) is None:
                continue
            else:
                content, rate = get_comment_content(comment)
                print(content, rate)
                writer.writerow([content, rate])

# 主网页
url = 'https://movie.douban.com/chart'

# 设置代理,
proxies = {
    'http': "49.233.242.15:5000"
}

# 解决乱码问题
response = requests.get(url, headers=headers, proxies=proxies)
response.encoding = 'utf-8'

# 获得不同类型电影的url
response = response.text
main_soup = BeautifulSoup(response, 'html.parser')
dif_movies = main_soup.select('#content > div > div.aside > div:nth-child(1) > div')
movie_types = []  # 存储不同类型电影的页面url
urls = dif_movies[0].find_all('a')

# 获得不同类型电影的代号和名称
for url in urls:
    param = url['href']
    index1 = param.find("&type=")
    index2 = param.find("&interval_id=")
    type_num = param[index1 + 6:index2]  # 电影类型代号
    type_name = param[20:index1]  # 电影类型名称
    movie_types.append((type_num, type_name))

# 爬取不同类型电影中前15的电影评论
for type, movie_class in movie_types:
    print(f"{'*' * 20}开始爬取->{movie_class}<-类型的电影{'*' * 20}")

    movie_url = 'https://movie.douban.com/j/chart/top_list'
    param = {
        "start": "0",
        "limit": "20",
        "type": f"{type}",
        "interval_id": "100:90",
        "action": "",

    }
    response = requests.get(movie_url, headers=headers, params=param, proxies=proxies)
    time.sleep(1)  # 限制请求时间，防止被封ip
    response.encoding = 'utf-8'
    movie_data = response.json()

    # 存放这个类型下前二十的电影的网址和名称
    urls = []
    for data in movie_data:
        urls.append((data['url'], data['title']))

    # 爬取每一部电影前10页的电影评论，共计200条
    for url in urls:
        # 获得这部电影下前十页的影评网址
        comment_urls = []
        head_url = url[0] + 'comments?limit=20&status=P&sort=new_score'  # 影评网址首页
        comment_urls.append(head_url)
        for i in range(20, 201, 20):
            other_url = url[0] + f"comments?start={i}&limit=20&status=P&sort=new_score"
            comment_urls.append(other_url)  # 后续9个影评网页

        # 爬取所有的200个影评并保存到对应文件
        print(f"===开始爬取电影《{url[1]}》的200条影评===")
        for comment_url in comment_urls:
            response = requests.get(comment_url, headers=headers, proxies=proxies)
            time.sleep(1)  # 限制次数，防止被封ip
            response.encoding = 'utf-8'
            response = response.text

            # 定位到评论所在的地方
            soup = BeautifulSoup(response, 'html.parser')
            comments = soup.find_all('div', class_='comment-item')
            save_to_csv(comments)
            time.sleep(1)

        print(f"===电影《{url[1]}》的200条影评爬取完毕===")
        time.sleep(5)