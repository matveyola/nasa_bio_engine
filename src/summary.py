import os
import math
import pandas as pd
from groq import Groq

# --- налаштування ---
MODEL = "llama-3.1-8b-instant"  
CHUNK_CHARS = 3000    
PARTIAL_MAX_TOKENS = 700             
FINAL_MAX_TOKENS = 700
TEMP = 0.3

# Ініціалізація клієнта (потрібен GROQ_API_KEY в env)
client = Groq()

# Дані
df = pd.read_csv("../data/exmpl_for_clustering.csv")
col = "abstract" if "abstract" in df.columns else df.columns[0]  # підстрахуємось
article_texts = df[col].fillna("").astype(str).tolist()

def chat_once(user_content: str, system_prompt: str = "Ти — помічник з біомедичних наук. Відповідай українською.",
              max_tokens: int = 400, temperature: float = 0.3) -> str:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        stream=False
    )
    return resp.choices[0].message.content.strip()

summaries = []

for idx, article_text in enumerate(article_texts, start=1):
    text = article_text.strip()
    if not text:
        summaries.append("")
        continue

    # 1) Розбиваємо на частини
    chunks = [text[i:i+CHUNK_CHARS] for i in range(0, len(text), CHUNK_CHARS)]

    # 2) Локальні самарі по кожному чанку
    partials = []
    for j, chunk in enumerate(chunks, start=1):
        prompt = (
            "Стисни цей фрагмент на 3–5 речень, згадай ціль, методи, ключові результати і висновки.\n\n"
            f"Фрагмент {j}/{len(chunks)}:\n{chunk}"
        )
        partial = chat_once(prompt, max_tokens=PARTIAL_MAX_TOKENS, temperature=TEMP)
        partials.append(partial)

    # 3) Фінальний самарі на основі часткових
    final_prompt = (
        "На основі наведених часткових підсумків склади цілісний підсумок статті на 5–10 речень, "
        "чітко і без преамбул/маркерів, українською:\n\n" + "\n\n".join(partials)
    )
    final_summary = chat_once(final_prompt, max_tokens=FINAL_MAX_TOKENS, temperature=TEMP)
    summaries.append(final_summary)

# Запис у датафрейм/файл
df["summary"] = summaries
df.to_csv("../data/exmpl_for_clustering_with_summaries.csv", index=False)
print("✅ Готово: колонка 'summary' додана.")
