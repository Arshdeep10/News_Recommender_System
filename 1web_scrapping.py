import scrapy
class thetribune(scrapy.Spider):
    name 'TheTribne'
    start_urls = ['https://www.tribuneindia.com/news/world']
    

    def parse(self, response):
        