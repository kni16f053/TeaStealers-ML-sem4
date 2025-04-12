from playwright.sync_api import sync_playwright
from time import sleep
import argparse
import requests

def handle_request(request):
    # Проверяем, является ли запрос GET-запросом
    if request.method == 'GET' and 'audio12' in request.url:
        print(f"Найдено аудио: {request.url}")

def parse_audios(word):
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()
        
        page.on('request', handle_request)

        # go to url
        page.goto(f"https://ru.forvo.com/word/{word}/#en", timeout=100000)
        # get HTML
        div_elements = page.query_selector_all('div.play.icon-size-xl')
        
        if len(div_elements) >= 4:
            div_elements = div_elements[:4]
        
        if div_elements:
            for element in div_elements:
                if type(element) != None:
                    element.click()  # Кликаем по кнопке воспроизведения записи для каждого найденного элемента
        else:
            print("Элементы не найдены.")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Слово из командной строки")
    parser.add_argument('word', type=str)
    
    args = parser.parse_args()
    parse_audios(args.word)