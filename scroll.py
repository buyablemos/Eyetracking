from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def init_driver():
    # Inicjalizacja przeglądarki
    options = webdriver.ChromeOptions()
    options.add_argument('--start-fullscreen')
    options.add_argument('--ignore-certificate-errors')
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    driver.get('https://csc.poc.us.comarch.com/selfcare/eshop')
    #driver.get('https://orange.pl')
    return driver


class BrowserTrackerApp:
    def __init__(self, driver):

        self.driver = driver
        self.page_height, self.page_width, self.viewport_height,self.viewport_width, self.scroll_top = 0, 0, 0, 0, 0
        self.update_position()

    def init_driver():
        # Inicjalizacja przeglądarki
        options = webdriver.ChromeOptions()
        options.add_argument('--start-fullscreen')
        options.add_argument('--ignore-certificate-errors')
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.get('https://csc.poc.us.comarch.com/selfcare/eshop')
        return driver

    def get_page_metrics(self):
        # Uruchamianie JavaScript w przeglądarce i pobieranie wymiarów strony i okna
        page_height = self.driver.execute_script('return document.body.scrollHeight;')
        page_width = self.driver.execute_script('return document.body.scrollWidth;')
        viewport_height = self.driver.execute_script('return window.innerHeight;')
        viewport_width = self.driver.execute_script('return window.innerWidth;')
        scroll_top = self.driver.execute_script('return window.pageYOffset;')
        return page_height, page_width, viewport_height,viewport_width, scroll_top

    def update_position(self):
        self.page_height, self.page_width, self.viewport_height,self.viewport_width, self.scroll_top = self.get_page_metrics()
        print(f'Page Height: {self.page_height},Page Width: {self.page_width}, Viewport Height: {self.viewport_height}, Scroll Top: {self.scroll_top}')

    def return_dimensions(self):
        return [self.page_height, self.page_width, self.viewport_height,self.viewport_width, self.scroll_top]
