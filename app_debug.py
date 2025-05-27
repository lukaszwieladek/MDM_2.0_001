import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import time

# ⬅️ MUSI BYĆ PIERWSZA
st.set_page_config(page_title="DEBUG: analiza kontekstowa", layout="centered")

# MODEL
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

# INTERFEJS
st.title("🐞 Debug: analiza semantyczna witryn")

st.header("🎯 Wprowadź dane wejściowe:")
product = st.text_area("🛍️ Opis produktu", "Ekskluzywne zegarki dla mężczyzn")
audience = st.text_area("👥 Grupa docelowa", "Zamożni mężczyźni 35+, zainteresowani modą i prestiżem")

# Wczytywanie listy domen z pliku
def load_sites_from_file(file_path="sites.txt"):
    try:
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        st.error("❌ Nie znaleziono pliku sites.txt w repozytorium.")
        return []

# Ścieżka z maks. 2 segmentami
def is_allowed_path_depth(url):
    path = urlparse(url).path
    segments = [seg for seg in path.split("/") if seg]
    return len(segments) <= 2

# Pobieramy linki wewnętrzne z domeny
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
        print(f"Błąd przy pobieraniu linków z {base_url}: {e}")
        return []

# Pobieranie tekstu
def extract_text(url):
    try:
        page = requests.get(url, timeout=5)
        soup = BeautifulSoup(page.text, "html.parser")
        return ' '.join(soup.stripped_strings)[:3000]
    except Exception as e:
        print(f"Błąd przy pobieraniu treści z {url}: {e}")
        return ""

# ANALIZA
if st.button("🚀 Uruchom analizę debugującą"):
    if not product or not audience:
        st.warning("❗ Wprowadź opis produktu i grupy docelowej.")
        st.stop()

    st.info("📂 Wczytywanie domen z `sites.txt`...")
    sites = load_sites_from_file()
    if not sites:
        st.warning("Brak domen.")
        st.stop()

    query_embedding = model.encode(product + " " + audience, convert_to_tensor=True)
    results = []

    st.info("🔬 Analiza trwa... obserwuj logi poniżej.")
    progress = st.progress(0)
    eta_placeholder = st.empty()

    total = len(sites)
    start_time = time.time()
    real_scans = 0

    for idx, site in enumerate(sites):
        print(f"\n🌐 Domena: {site}")
        links = extract_links(site, limit=30)
        best_score = 0.0
        best_url = ""
        found_texts = 0

        for link in links:
            print(f"→ Link: {link}")
            text = extract_text(link)
            print(f"   📄 Długość tekstu: {len(text)}")

            if text.strip():
                found_texts += 1
                page_embedding = model.encode(text, convert_to_tensor=True)
                score = util.pytorch_cos_sim(query_embedding, page_embedding).item()
                print(f"   ✅ Score: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_url = link
            else:
                print(f"   ⚠️ Pusta treść lub niedostępna")

            time.sleep(0.5)

        if found_texts > 0:
            real_scans += 1

        results.append({
            "Domena": site,
            "Najlepiej dopasowana podstrona": best_url,
            "Dopasowanie (%)": round(best_score * 100, 2),
            "Ilość przeanalizowanych linków": found_texts
        })

        # Postęp i ETA
        progress.progress((idx + 1) / total)
        elapsed = time.time() - start_time
        avg_time = elapsed / (idx + 1)
        remaining = avg_time * (total - idx - 1)
        mins, secs = divmod(int(remaining), 60)
        eta_placeholder.info(f"⏱️ Szacowany czas do końca: {mins} min {secs} sek")

    df = pd.DataFrame(sorted(results, key=lambda x: -x["Dopasowanie (%)"]))
    st.success(f"✅ Gotowe. Przeanalizowano {real_scans}/{len(sites)} domen z treścią.")
    st.dataframe(df)
    st.download_button("📥 Pobierz CSV", df.to_csv(index=False), file_name="debug_ranking.csv")
