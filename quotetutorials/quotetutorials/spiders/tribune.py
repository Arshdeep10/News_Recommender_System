import scrapy

from ..items import QuotetutorialsItem


class QuotesSpider(scrapy.Spider):
    name = 'tribune'

    start_urls = ['https://www.tribuneindia.com/news/special/entertainment']
    
    
    def parse2(self, response1):
        items = QuotetutorialsItem()
        # print("ardhdeep" )
        card2 = response1.css('div.story-desc')
        # print(card2)
        content = card2.css('p::text').extract()
        # print(content)
        time1 = response1.css('div.time-share')
        time = time1.css('ul>li>p>span::text').extract()
        title = response1.css('div.glb-heading>h1::text').extract()
        time_final = []
        for i in time:
            if i.strip() != '':
                time_final.append(i)
        content_final = []
        for i in content:
            if i.strip() != '':
                content_final.append(i)

        
        items['title'] = title
        items['time'] = time_final
        items['content'] = content_final
        yield items

    
    
    def parse(self, response):
        items = QuotetutorialsItem()
        cards = response.css('div.card')
        count = 0
        for headings in cards:
            # print(count)
            # title = headings.css('a.card-top-align::text').extract()
            # items['title'] = title
            # yield items
            # time = headings.css('span.text::text').extract()
            # print(title)

            next_page = headings.css('a.card-top-align::attr(href)')
            # print(next_page.extract())
            count = count +1
            string = 'https://www.tribuneindia.com'
            if next_page:
                url = string + next_page.extract()[0]
                # print(url)
                url = scrapy.Request(url, callback = self.parse2)
                yield url
                # print(type(url))
        print(items)
                