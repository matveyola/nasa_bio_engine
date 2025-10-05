import requests
import pandas as pd
import time
import re
import os
import json
from urllib.parse import urlparse
import xml.etree.ElementTree as ET
from io import StringIO
import html

class NCBIArticleExtractor:
    def __init__(self, email, max_retries=3, delay=0.34):
        self.email = email
        self.max_retries = max_retries
        self.delay = delay
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'restricted_skipped': 0,
            'failed': 0
        }
    
    def extract_pmc_id(self, url):
        """Витягнення PMC ID з URL"""
        if not url or pd.isna(url):
            return None
        
        url = str(url).strip()
        
        patterns = [
            r'PMC(\d+)',
            r'/articles/PMC(\d+)',
            r'pmc=PMC(\d+)',
            r'https?://[^/]+/pmc/articles/PMC(\d+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url, re.IGNORECASE)
            if match:
                pmc_id = f"PMC{match.group(1)}"
                return pmc_id
        
        return None
    
    def check_article_restrictions(self, xml_content):
        """Перевірка обмежень доступу до статті"""
        if "does not allow downloading of the full text" in xml_content:
            return True
        
        # Перевірка розміру контенту (дуже малий розмір може означати обмежений доступ)
        text_only = re.sub(r'<[^>]+>', ' ', xml_content)
        text_only = re.sub(r'\s+', ' ', text_only).strip()
        if len(text_only) < 5000:  # Дуже мало тексту для наукової статті
            return True
        
        return False
    
    def get_article_via_api(self, pmc_id):
        """Отримання статті через API"""
        for attempt in range(self.max_retries):
            try:
                params = {
                    'db': 'pmc',
                    'id': pmc_id.replace('PMC', ''),
                    'retmode': 'xml',
                    'rettype': 'full'
                }
                
                headers = {
                    'User-Agent': f'NCBI_API_Client/1.0 ({self.email})',
                    'Accept': 'application/xml'
                }
                
                response = requests.get(
                    f"{self.base_url}/efetch.fcgi",
                    params=params,
                    headers=headers,
                    timeout=30
                )
                
                if response.status_code == 200:
                    return response.text
                elif response.status_code == 429:
                    wait_time = (2 ** attempt) * self.delay
                    time.sleep(wait_time)
                    
            except Exception as e:
                print(f"   ⚠️  Помилка спроби {attempt + 1}: {e}")
            
            if attempt < self.max_retries - 1:
                time.sleep(self.delay)
        
        return None
    
    def get_article_metadata(self, pmc_id):
        """Отримання метаданих статті"""
        try:
            params = {
                'db': 'pmc',
                'id': pmc_id.replace('PMC', ''),
                'retmode': 'json'
            }
            
            response = requests.get(f"{self.base_url}/esummary.fcgi", params=params, timeout=15)
            if response.status_code == 200:
                data = response.json()
                result = data.get('result', {})
                article_info = result.get(pmc_id, {})
                
                return {
                    'title': article_info.get('title', ''),
                    'authors': ', '.join(article_info.get('authors', [])),
                    'journal': article_info.get('fulljournalname', ''),
                    'pub_date': article_info.get('pubdate', ''),
                    'doi': article_info.get('elocationid', ''),
                    'pmcid': pmc_id
                }
        except Exception:
            pass
        
        return {
            'title': '',
            'authors': '',
            'journal': '',
            'pub_date': '',
            'doi': '',
            'pmcid': pmc_id
        }
    
    def extract_element_text(self, element):
        """Рекурсивне витягнення тексту з XML елемента"""
        if element is None:
            return ""
        
        texts = []
        
        if element.text and element.text.strip():
            texts.append(element.text.strip())
        
        for child in element:
            child_text = self.extract_element_text(child)
            if child_text:
                texts.append(child_text)
            
            if child.tail and child.tail.strip():
                texts.append(child.tail.strip())
        
        return ' '.join(texts)
    
    def parse_article_content(self, xml_content, pmc_id):
        """Парсинг вмісту статті з XML"""
        try:
            root = ET.fromstring(xml_content)
            
            # Отримання заголовка
            title_elem = root.find('.//article-title')
            title = self.extract_element_text(title_elem) if title_elem is not None else ""
            
            # Отримання абстракту
            abstract_elems = root.findall('.//abstract')
            abstract_texts = []
            for abstract in abstract_elems:
                abstract_text = self.extract_element_text(abstract)
                if abstract_text:
                    abstract_texts.append(abstract_text)
            abstract = '\n\n'.join(abstract_texts)
            
            # Отримання основного тексту
            body_elem = root.find('.//body')
            full_text = ""
            if body_elem is not None:
                full_text = self.extract_element_text(body_elem)
            
            # Отримання ключових слів
            kwd_elems = root.findall('.//kwd')
            keywords = [self.extract_element_text(kwd) for kwd in kwd_elems if self.extract_element_text(kwd)]
            
            return {
                'title': self.clean_text(title),
                'abstract': self.clean_text(abstract),
                'full_text': self.clean_text(full_text),
                'keywords': '; '.join(keywords)
            }
            
        except ET.ParseError:
            # Резервний метод для складних XML
            return self.parse_xml_fallback(xml_content)
        except Exception as e:
            print(f"   ❌ Помилка парсингу: {e}")
            return None
    
    def parse_xml_fallback(self, xml_content):
        """Резервний метод парсингу XML"""
        try:
            # Видалення тегів
            text = re.sub(r'<[^>]+>', ' ', xml_content)
            text = re.sub(r'\s+', ' ', text)
            text = html.unescape(text)
            text = self.clean_text(text)
            
            return {
                'title': '',
                'abstract': '',
                'full_text': text,
                'keywords': ''
            }
        except Exception:
            return None
    
    def clean_text(self, text):
        """Очищення тексту"""
        if not text:
            return ""
        
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\s+([.,;:!?])', r'\1', text)
        text = re.sub(r'([.,;:!?])\s+', r'\1 ', text)
        text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', text)
        
        return text.strip()
    
    def is_valid_article(self, article_data, min_text_length=20000):
        """Перевірка чи стаття є валидною (не обмежена)"""
        if not article_data:
            return False
        
        full_text = article_data.get('full_text', '')
        
        # Якщо текст занадто короткий, ймовірно стаття обмежена
        if len(full_text) < min_text_length:
            return False
        
        # Перевірка наявності основних розділів
        if not any(word in full_text.lower() for word in ['method', 'result', 'discussion', 'conclusion', 'introduction']):
            return False
        
        return True
    
    def save_to_csv(self, results, output_file):
        """Збереження результатів у CSV файл"""
        try:
            # Створюємо DataFrame
            df_data = []
            for result in results:
                if result['status'] == 'SUCCESS':
                    row = {
                        'Original_Title': result['original_name'],
                        'URL': result['original_url'],
                        'PMC_ID': result['pmc_id'],
                        'Article_Title': result['title'],
                        'Authors': result['authors'],
                        'Journal': result['journal'],
                        'Publication_Date': result['pub_date'],
                        'DOI': result['doi'],
                        'Keywords': result.get('keywords', ''),
                        'Abstract': result['abstract'],
                        'Full_Text': result['full_text'],
                        'Text_Length': len(result['full_text']),
                        'Download_Date': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
                    df_data.append(row)
            
            if not df_data:
                print("❌ Немає даних для збереження")
                return False
            
            df = pd.DataFrame(df_data)
            
            # Зберігаємо у CSV з кодуванням UTF-8
            df.to_csv(output_file, index=False, encoding='utf-8')
            
            return True
            
        except Exception as e:
            print(f"❌ Помилка збереження CSV: {e}")
            return False
    
    def save_full_text_files(self, results, output_dir):
        """Збереження повних текстів у окремі файли"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for result in results:
                if result['status'] == 'SUCCESS':
                    pmc_id = result['pmc_id']
                    filename = os.path.join(output_dir, f"{pmc_id}_full_text.txt")
                    
                    with open(filename, 'w', encoding='utf-8') as f:
                        f.write(f"PMC ID: {pmc_id}\n")
                        f.write(f"Title: {result['title']}\n")
                        f.write(f"Authors: {result['authors']}\n")
                        f.write(f"Journal: {result['journal']}\n")
                        f.write(f"Publication Date: {result['pub_date']}\n")
                        f.write(f"DOI: {result['doi']}\n")
                        f.write("="*80 + "\n\n")
                        f.write("ABSTRACT:\n")
                        f.write(result['abstract'])
                        f.write("\n\n" + "="*80 + "\n\n")
                        f.write("FULL TEXT:\n")
                        f.write(result['full_text'])
            
            return True
        except Exception as e:
            print(f"❌ Помилка збереження текстових файлів: {e}")
            return False
    
    def process_csv_file(self, input_csv, output_csv, text_files_dir=None):
        """Основний метод обробки CSV файлу"""
        try:
            print(f"📖 Читання CSV файлу: {input_csv}")
            df = pd.read_csv(input_csv)
            
            if len(df.columns) < 2:
                print("❌ Файл має менше 2 колонок")
                return False
            
            name_column = df.columns[0]
            url_column = df.columns[1]
            
            print(f"🔍 Використовуються колонки: '{name_column}' (назви), '{url_column}' (посилання)")
            
            results = []
            
            for index, row in df.iterrows():
                self.stats['total_processed'] += 1
                
                article_name = str(row[name_column]) if pd.notna(row[name_column]) else "Без назви"
                article_url = str(row[url_column]) if pd.notna(row[url_column]) else ""
                
                print(f"\n📄 Обробка {index + 1}/{len(df)}: {article_name[:60]}...")
                print(f"   🔗 URL: {article_url}")
                
                # Витягнення PMC ID
                pmc_id = self.extract_pmc_id(article_url)
                
                if not pmc_id:
                    print("   ❌ Не вдалося витягти PMC ID")
                    self.stats['failed'] += 1
                    continue
                
                print(f"   ✅ PMC ID: {pmc_id}")
                
                # Отримання метаданих
                print("   📊 Отримання метаданих...")
                metadata = self.get_article_metadata(pmc_id)
                
                # Отримання повного тексту через API
                print("   📡 Завантаження повного тексту...")
                xml_content = self.get_article_via_api(pmc_id)
                
                if not xml_content:
                    print("   ❌ Не вдалося завантажити статтю")
                    self.stats['failed'] += 1
                    continue
                
                # Перевірка обмежень доступу
                if self.check_article_restrictions(xml_content):
                    print("   ⏭️  Стаття обмежена видавцем - пропускаємо")
                    self.stats['restricted_skipped'] += 1
                    continue
                
                # Парсинг вмісту статті
                print("   🔍 Парсинг вмісту...")
                content_data = self.parse_article_content(xml_content, pmc_id)
                
                if not content_data:
                    print("   ❌ Не вдалося розпарсити вміст")
                    self.stats['failed'] += 1
                    continue
                
                # Перевірка чи стаття є валидною (не обмежена)
                if not self.is_valid_article(content_data):
                    print("   ⏭️  Стаття має обмежений доступ або недостатньо тексту - пропускаємо")
                    self.stats['restricted_skipped'] += 1
                    continue
                
                # Об'єднуємо метадані та контент
                article_result = {
                    **metadata,
                    **content_data,
                    'original_name': article_name,
                    'original_url': article_url,
                    'pmc_id': pmc_id,
                    'status': 'SUCCESS'
                }
                
                results.append(article_result)
                self.stats['successful'] += 1
                
                print(f"   ✅ Успішно! Текст: {len(content_data['full_text']):,} символів")
                
                # Дотримання лімітів NCBI
                time.sleep(self.delay)
            
            # Збереження результатів у CSV
            print(f"\n💾 Збереження результатів у CSV...")
            if self.save_to_csv(results, output_csv):
                print(f"✅ Результати збережено у: {output_csv}")
            else:
                print("❌ Не вдалося зберегти результати")
            
            # Збереження текстових файлів (опціонально)
            if text_files_dir:
                print(f"💾 Збереження текстових файлів...")
                if self.save_full_text_files(results, text_files_dir):
                    print(f"✅ Текстові файли збережено у: {text_files_dir}")
            
            self.print_statistics()
            
            return True
            
        except Exception as e:
            print(f"❌ Критична помилка: {e}")
            return False
    
    def print_statistics(self):
        """Вивід статистики обробки"""
        print("\n" + "="*60)
        print("📊 СТАТИСТИКА ОБРОБКИ:")
        print(f"   📄 Всього оброблено: {self.stats['total_processed']}")
        print(f"   ✅ Успішно збережено: {self.stats['successful']}")
        print(f"   ⏭️  Пропущено (обмежений доступ): {self.stats['restricted_skipped']}")
        print(f"   ❌ Помилок: {self.stats['failed']}")
        
        if self.stats['successful'] > 0:
            success_rate = (self.stats['successful'] / self.stats['total_processed']) * 100
            print(f"   📈 Успішність: {success_rate:.1f}%")


def main():
    """Головна функція"""
    extractor = NCBIArticleExtractor(
        email="oleksiyvolkov2435@gmail.com",  # Замініть на ваш email
        max_retries=3,
        delay=0.34
    )
    
    success = extractor.process_csv_file(
        input_csv='SB_publication_PMC.csv',           # Ваш CSV файл
        output_csv='articles_results.csv',  # Вихідний CSV файл
        text_files_dir='full_texts'     # Папка для текстових файлів (опціонально)
    )
    
    if success:
        print("\n🎉 Обробка завершена успішно!")
        print("📝 Тільки статті з повним доступом збережено у CSV файл")
        print("💾 Повні тексти збережено без обрізання")
    else:
        print("\n❌ Обробка завершена з помилками")


if __name__ == "__main__":
    main()