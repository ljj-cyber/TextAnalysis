import requests
from bs4 import BeautifulSoup
import csv
import time


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


if __name__ == '__main__':
    url0 = 'https://movie.douban.com/subject/35700910/comments'
    url = 'https://movie.douban.com/subject/36283000/comments'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/117.0.0.0 Safari/537.36 Edg/117.0.2045.43,',
        'Cookie': 'bid=AAt90yNMAVM; ap_v=0,6.0; __utma=30149280.1187531932.1696056975.1696056975.1696056975.1;'
                  ' __utmz=30149280.1696056975.1.1.utmcsr=(direct)|utmccn=(direct)|utmcmd=(none); __utmt=1; __'
                  'utmc=30149280; __utmb=30149280.2.10.1696056975; dbcl2="217639176:0OYYjnLY8dE"; ck=FVf7; '
                  'push_noty_num=0; push_doumail_num=0'}

    while url:
        print('正在爬取：', url)
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, 'html.parser')
        comments = soup.find_all('div', class_='comment-item')
        save_to_csv(comments)
        time.sleep(1)
        try:
            #页面跳转的点击在div的id为paginator ，class为center中的a标签class为next
            pagination = soup.find('div', class_='center').find('a', class_='next')
            print(pagination)
            if pagination:
                next_page = pagination['href']
                url = url0 + next_page
                print('跳转到下一页：', url)
                print('找到标签跳转')
            else:
                url = None
                print('无跳转')
        except:
            url = None
            print('没有找到，结束')

    print('爬取完成！')