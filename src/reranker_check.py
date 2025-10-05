from sentence_transformers import CrossEncoder
from typing import List, Tuple
import time

# --- Визначення Класу Reranker (Переранжувальника) ---

CROSS_MODEL = "BAAI/bge-reranker-large"

class Reranker:
    def __init__(self, model_name: str = CROSS_MODEL):
        print(f"Завантаження моделі CrossEncoder: {model_name}...")
        self.model = CrossEncoder(model_name)
        print("Модель завантажено.")

    def rerank(self, query: str, candidates: List[str], top_k: int = 3) -> List[Tuple[int, float]]:
        """
        candidates: список текстів для ранжування.
        
        Повертає список (index_in_candidates, score) від найкращого до найгіршого, 
        обмежений top_k.
        """
        # Створення пар (запит, кандидат)
        pairs = [[query, c] for c in candidates]
        
        # Отримання оцінок релевантності
        scores = self.model.predict(pairs)
        
        # Сортування індексів за спаданням оцінок і вибір top_k
        idx_sorted = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        # Формування результату: (індекс, оцінка)
        return [(i, float(scores[i])) for i in idx_sorted]

# --- Синтетичні Дані (Космічна Біологія та Медицина) ---

query = "Як мікрогравітація впливає на щільність кісткової тканини у космонавтів?"

candidates = [
    # Кандидат 0: Дуже релевантний
    "Дослідження на МКС підтверджують, що тривалий вплив мікрогравітації призводить до **значної втрати кісткової маси** (остеопенії) в астронавтів, що вимагає **контрзаходів** у вигляді інтенсивних фізичних вправ та фармакологічної підтримки.",
    
    # Кандидат 1: Трохи релевантний (загальна біологія)
    "Аналіз зразків ґрунту з Марса показав відсутність активних форм життя, що звужує пошуки **позаземної біології** до підповерхневих шарів планети.",
    
    # Кандидат 2: Релевантний (медицина)
    "Одним із основних ризиків **космічної медицини** є **радіаційне опромінення**, яке підвищує ймовірність розвитку катаракти та злоякісних пухлин.",
    
    # Кандидат 3: Найбільш релевантний (пряме потрапляння)
    "Механізми **втрати кісткової тканини в космосі** напряму пов'язані з відсутністю гравітаційного навантаження.",
    
    # Кандидат 4: Нерелевантний
    "Історія скафандрів: від ранніх моделей 'Орлан' до сучасних легких систем із покращеною рухливістю для **виходів у відкритий космос**."
]

print(f"Запит: {query}")
print(f"Кількість кандидатів: {len(candidates)}")

# --- Ініціалізація та Тестування Reranker ---

try:
    reranker = Reranker()
except Exception as e:
    print(f"Сталася помилка при ініціалізації Reranker: {e}")
    raise

print("\nПочаток переранжування...")
start_time = time.time()
# Виконання переранжування (top_k=3)
top_results = reranker.rerank(query, candidates, top_k=3)
end_time = time.time()

print(f"Переранжування завершено за {end_time - start_time:.4f} секунд.")

# --- Результати Переранжування ---

print("\n--- Топ-3 Результати Переранжування ---")
print(f"Запит: {query}\n")

for rank, (index, score) in enumerate(top_results):
    print(f"Ранг {rank + 1}: (Індекс {index}, Оцінка: {score:.4f})")
    print(f"Кандидат: {candidates[index]}\n")
    
print("---------------------------------")