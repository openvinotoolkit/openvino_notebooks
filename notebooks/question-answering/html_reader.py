# code from https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/common/python/html_reader.py
import logging as log
import requests
from html.parser import HTMLParser


class HTMLDataExtractor(HTMLParser):
    def __init__(self, tags):
        super(HTMLDataExtractor, self).__init__()
        self.started_tags = {k: [] for k in tags}
        self.ended_tags = {k: [] for k in tags}

    def handle_starttag(self, tag, attrs):
        if tag in self.started_tags:
            self.started_tags[tag].append([])

    def handle_endtag(self, tag):
        if tag in self.ended_tags:
            txt = ''.join(self.started_tags[tag].pop())
            self.ended_tags[tag].append(txt)

    def handle_data(self, data):
        for tag, l in self.started_tags.items():
            for d in l:
                d.append(data)


# read html urls and list of all paragraphs data
def get_paragraphs(url_list):
    headers = {"User-agent": "Mozilla/5.0"}
    
    paragraphs_all = []
    for url in url_list:
        log.info("Get paragraphs from {}".format(url))
        response = requests.get(url=url, headers=headers)
        parser = HTMLDataExtractor(['title', 'p'])
        parser.feed(response.text)
        title = ' '.join(parser.ended_tags['title'])
        paragraphs = parser.ended_tags['p']
        log.info("Page '{}' has {} chars in {} paragraphs".format(title, sum(len(p) for p in paragraphs), len(paragraphs)))
        paragraphs_all.extend(paragraphs)

    return paragraphs_all
