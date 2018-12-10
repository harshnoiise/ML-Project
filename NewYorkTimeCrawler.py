import urllib.request as request
import urllib.parse as parse
from pprint import pprint
import json
import numpy as np


class Crawler(object):
    def __init__(self, url, apikey, start_year, end_year):
        self.url = url
        self.apiKey = apikey
        self.startYear = int(start_year)
        self.endYear = int(end_year)

    def start(self):
        articles = []
        years = np.arange(self.startYear, self.endYear+1)

        for year in years:
            url = self.url + "/" + str(year) + "/" + "1.json"

            data = {}
            data['api-key'] = self.apiKey
            url_values = parse.urlencode(data)

            # add api key
            url = url + "?" + url_values
            with request.urlopen(url) as response:
                content = response.read()
                content = content.decode('utf-8')
                content = json.loads(content)
                for each in content['response']['docs']:
                    if "section_name" in each:
                        section = each['section_name']
                    else:
                        continue
                    if "print_headline" in each['headline']:
                        headline = each['headline']['print_headline']
                    else:
                        headline = None
                    keywords = each['keywords']
                    if "lead_paragraph" in each:
                        lead_paragraph = each['lead_paragraph']
                    else:
                        lead_paragraph = None
                    articles.append({
                        'section': section,
                        'headline': headline,
                        'keywords': keywords,
                        'lead_paragraph': lead_paragraph
                    })
        print(len(articles))
        with open('newyorktimes.jsonl', 'w') as f:
            for each in articles:
                json.dump(each, f)
                f.write('\n')

    def filter(self, path):
        filtered_articles = []
        with open(path, 'r') as f:
            for line in f:
                article = json.loads(line)
                if article['section'] is None:
                    continue
                else:
                    filtered_articles.append(article)

        print(len(filtered_articles))
        with open('newyorktimes_filtered.jsonl', 'w') as f:
            for each in filtered_articles:
                json.dump(each, f)
                f.write('\n')



if __name__ == '__main__':
    crawler = Crawler(
        url = "https://api.nytimes.com/svc/archive/v1",
        apikey='c01f49c89058448b8c3bf555db0ca4cd',
        start_year=2005,
        end_year=2018
    )

    crawler.start()
    crawler.filter('newyorktimes.jsonl')
