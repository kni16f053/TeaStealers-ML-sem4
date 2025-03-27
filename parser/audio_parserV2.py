#_________________________________________________ОПИСАНИЕ ПРОГРАММЫ_________________________________________________
#
# 
# Данный парсер выгружает из сайта forvo ПРЯМЫЕ ссылки на .mp3 файлы с произношениями.
# Выгрузка сама по себе очень долгая, поэтому в программе предлагается настройка
# параметров распределения задачи между разными вычислительными машинами для ускорения.
#
# Прежде всего это параметр 'startIdx' - индекс слова, с которого начинаем.
# Параметр 'wordsToProcess' отвечает за то, сколько всего слов за все сессии нужно обработать одной машине.
# Под сессией подразумевается непрерывная работа машины по добыче не более audiosForSession записей.
# Если машина за сессию обрабатывает всё кол-во слов 'wordsToProcess', то она завершает работу заранее.
# Сколько уж получилось добыть, из воздуха аудиозаписи не возьмёшь...
#
# Так как во время работы случайно может выключиться компьютер или программа зависнуть, то
# это не проблема, так как каждые 'audiosCountToDump' добытых  аудиозаписей сохраняется в файл
# во время работы программы.
#
# Так же если случайно отключился интернет или что-то случилось с сайтом forvo, то программа
# не будет продолжать работу, пока соединение не восстановится.
# В то же время она сообщит о проблеме соединения с сайтом.

# Выкачиваются только произношения с британским акцентом.

# Отдельно упомянем следующий параметр, от которого зависит настройка параметров распределения нагрузки между машинами:
#
# Параметр 'excludeList'* - список тех подстрок, с которыми слова удаляются и их произношение не рассматривается.
# *Именно то итоговое количество слов, которое получилось после удаления с использованием этого параметра, 
# рассматривается при настроке 'startIdx' и 'wordsToProcess'!

# Остальные параметры, которые не названы тут, описаны в комментариях к коду, который ниже:


from playwright.sync_api import sync_playwright
from time import sleep
import requests


startIdx = 61678 #стартовая точка диапазона слов, аудио которых начинаем парсить
wordsToProcess = 61678 #сколько слов всего нужно пройти на конкретной
excludeList = ['aaa', 'bbb', 'ccc', 'ddd', 'eee', 'fff', 'ggg', 'hhh', 'iii', 'jjj', 'kkk', 'lll', 'mmm',
               'nnn', 'ooo', 'ppp', 'qqq', 'rrr', 'sss', 'ttt', 'uuu', 'vvv', 'www', 'xxx', 'yyy', 'zzz']
audiosCountToDump = 50 # через сколько аудио выгружать в файл для сохранения,
                       # а то вдруг вырубят электричество. Лучше после с checkpointWord начать, чем заново

#session parameters
audiosForSession = 100000 #сколько записей нужно сохранить (не обязательно равно кол-ву слов!)
checkpointWord = '' #последнее слово, обработанное в прошлый раз
fileWithWordsName = 'C:/words_alpha.txt' # файл, откуда берутся слова, по которым ищутся произношения
fileToSaveDatasetName = 'C:/audio_urls_part2.txt' # файл, куда сохраняются спарсенные данные в формате:
                                                  # <произнесённое слово>: <пол произносившего>: <url на mp3 аудиозапись>

#служебные глобальные переменные
savedCount = 0
gender = ''

def CheckInternetConnection(url):
    try:
        requests.get(url, timeout=4)
        return True
    except:
        return False
def handle_request(request):
    global savedCount
    if request.method == 'GET' and 'audio12' in request.url:
        print(f"Найдено аудио: {request.url}")
        f.write(word + ': ' + gender + ': ' + request.url + '\n')
        savedCount += 1

def parse_audios(sample):
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False)
        context = browser.new_context(viewport={"width": 1920, "height": 1080})
        page = context.new_page()       
        page.on('request', handle_request)

        attemptCount = 0
        while not(CheckInternetConnection('https://ru.forvo.com/')):        
            sleep(5)
            attemptCount += 1
            print(f'Error Forvo connection! Attempt {attemptCount} to connect...')

        ##Так сделано для ускорения (низкий таймаут загрузки страницы),
        ##Потому что страница может загружаться долго, хотя все нужные элементы уже давно загружены.
        ##Тогда зачем лишний раз ждать?)
        try:
            page.goto(f"https://ru.forvo.com/word/{sample}/#en", timeout=5000)
        except:
            pass
        try:
            page.click('button:has-text("СОГЛАСЕН")', timeout=3000)
        except:
            pass    
        try:
            page.wait_for_selector('.pronunciations-list', timeout=1000)
        except:
            print("Элементы не найдены (404).")
            return
        
        try:
            page.wait_for_selector('.pronunciations-list.pronunciations-list-en_uk', timeout=1000)
        except:
            print(f'Для слова "{sample}" не найдено британских прозношений')
            return
        
        uk_pronunciations = page.query_selector('.pronunciations-list.pronunciations-list-en_uk')
        if not uk_pronunciations:
            print(f'Британский блок не найден для слова "{sample}"')
            return
        
        pronunciation_items = uk_pronunciations.query_selector_all('li.pronunciation.en_uk')
        if not pronunciation_items:
            print(f'Записи для слова "{sample}" не найдены!')
            return
        
        if len(pronunciation_items) > 4:
            pronunciation_items = pronunciation_items[:4]

        global gender
        for item in pronunciation_items:
            div_element = item.query_selector('div.play.icon-size-xl')
            span_element = item.query_selector('span.from')       
            if span_element:
                speakerInfo = span_element.inner_text()
                gender = speakerInfo.split(', ')[0][1:]
            else:
                gender = 'неизвестно'
            if type(div_element) != None:
                div_element.click()
            
if __name__ == '__main__':

    with open(fileWithWordsName, 'r') as EngWordsFile:
        engWordsList = EngWordsFile.read().splitlines()
        print('Initial english words count: ', len(engWordsList))

    f = open(fileToSaveDatasetName, 'a')

    margin = 0
    for idx in range(len(engWordsList)):
        if any(subStr in engWordsList[idx-margin] for subStr in excludeList):
            engWordsList.pop(idx-margin)
            margin += 1
    print('Finally english words count: ', len(engWordsList))

    if startIdx + wordsToProcess <= len(engWordsList):
        startSessionIdx = -1
        if checkpointWord == '':
            startSessionIdx = startIdx
        else:
            for idx in range(startIdx, startIdx + wordsToProcess - 1):
                if engWordsList[idx] == checkpointWord:
                    startSessionIdx = idx + 1
        if startSessionIdx > -1:
            dumpsCountPrev, dumpsCountNext = 0, 0
            for sample in engWordsList[startSessionIdx:startIdx + wordsToProcess]:
                word = sample
                parse_audios(sample)
                if savedCount >= audiosForSession:
                    print (f'Parsed {savedCount} audio urls.')
                    break
                dumpsCountNext = savedCount // audiosCountToDump
                if dumpsCountNext > dumpsCountPrev:
                    f.close()
                    sleep(1)
                    f = open(fileToSaveDatasetName, 'a')
                    dumpsCountPrev = dumpsCountNext
        else:
            print('Invalid Checkpoint Word!')
    else:
        print('Invalid start index or words count to process!')

    f.close()
