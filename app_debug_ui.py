import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import time

# KONFIGURACJA
st.set_page_config(page_title="DEBUG UI – analiza kontekstowa", layout="centered")

@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

st.title("🧪 Debug: analiza semantyczna (z logiem w UI)")

st.header("🎯 Wprowadź dane wejściowe:")
product = st.text_area("🛍️ Opis produktu", "Ekskluzywne zegarki dla mężczyzn")
audience = st.text_area("👥 Grupa docelowa", "Zamożni mężczyźni 35+, zainteresowani modą i prestiżem")

def load_sites_from_file(file_path="sites.txt"):
    try:
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        st.error("❌ Nie znaleziono pliku sites.txt.")
        return []

def is_allowed_path_depth(url):
    path = urlparse(url).path
    segments = [seg for seg in path.split("/") if seg]
    return len(segments) <= 2

def extract_links(base_url, limit=30):
    try:
        page = requests.get(base_url, timeout=5)
        soup = BeautifulSoup(page.text, "html.parser")
        links = set()
        for tag in soup.find_all("a", href=True):
            href = tag["href"]
            full_url = urljoin(base_url, href)
            if urlparse(full_url).netloc == urlparse(base_url).netloc:
                if is_allowed_path_depth(full_url):
                    links.add(full_url)
            if len(links) >= limit:
                break
        return list(links)
    except Exception as e:
        return []

def extract_text(url):
    try:
        page = requests.get(url, timeout=5)
        soup = BeautifulSoup(page.text, "html.parser")
        return ' '.join(soup.stripped_strings)[:3000]
    except:
        return ""

# ANALIZA
if st.button("🚀 Uruchom analizę z logiem UI"):
    if not product or not audience:
        st.warning("❗ Uzupełnij dane.")
        st.stop()

    st.info("📂 Wczytuję domeny z `sites.txt`...")
    sites = load_sites_from_file()
    if not sites:
        st.warning("Brak domen.")
        st.stop()

    query_embedding = model.encode(product + " " + audience, convert_to_tensor=True)
    results = []
    progress = st.progress(0)
    eta_placeholder = st.empty()
    log_area = st.empty()

    total = len(sites)
    start_time = time.time()
    log_text = ""

    for idx, site in enumerate(sites):
        log_text += f"\n🌐 Domena: {site}\n"
        links = extract_links(site, limit=30)
        best_score = 0.0
        best_url = ""
        found_texts = 0

        for link in links:
            log_text += f"→ Link: {link}\n"
            text = extract_text(link)
            log_text += f"   📄 Długość tekstu: {len(text)}\n"

            if text.strip():
                found_texts += 1
                page_embedding = model.encode(text, convert_to_tensor=True)
                score = util.pytorch_cos_sim(query_embedding, page_embedding).item()
                log_text += f"   ✅ Score: {score:.4f}\n"
                if score > best_score:
                    best_score = score
                    best_url = link
            else:
                log_text += f"   ⚠️ Brak treści lub błąd pobrania\n"

            time.sleep(0.5)

        results.append({
            "Domena": site,
            "Najlepiej dopasowana podstrona": best_url,
            "Dopasowanie (%)": round(best_score * 100, 2),
            "Przeanalizowane linki": found_texts
        })

        # Postęp i czas
        progress.progress((idx + 1) / total)
        elapsed = time.time() - start_time
        avg_time = elapsed / (idx + 1)
        remaining = avg_time * (total - idx - 1)
        mins, secs = divmod(int(remaining), 60)
        eta_placeholder.info(f"⏱️ Szacowany czas do końca: {mins} min {secs} sek")
        log_area.code(log_text)

    df = pd.DataFrame(sorted(results, key=lambda x: -x["Dopasowanie (%)"]))
    st.success(f"✅ Gotowe. Zanalizowano {len(df)} domen.")
    st.dataframe(df)
    st.download_button("📥 Pobierz CSV", df.to_csv(index=False), file_name="debug_ui_ranking.csv")
