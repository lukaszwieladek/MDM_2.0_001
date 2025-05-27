import streamlit as st
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from sentence_transformers import SentenceTransformer, util
import pandas as pd
import time

# ⬅️ KONFIGURACJA APLIKACJI – MUSI BYĆ NA SAMYM POCZĄTKU
st.set_page_config(page_title="Analiza kontekstowa domen", layout="centered")

# MODEL
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

# INTERFEJS
st.title("🔍 Analiza semantyczna witryn z pliku")

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

# Filtrujemy tylko linki z maks. 2 poziomami ścieżki
def is_allowed_path_depth(url):
    path = urlparse(url).path
    segments = [seg for seg in path.split("/") if seg]
    return len(segments) <= 2

# Pobieramy podlinkowane URL-e z domeny
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
    except:
        return []

# Pobieranie tekstu ze strony
def extract_text(url):
    try:
        page = requests.get(url, timeout=5)
        soup = BeautifulSoup(page.text, "html.parser")
        return ' '.join(soup.stripped_strings)[:3000]
    except:
        return ""

# PRZYCISK START
if st.button("🚀 Rozpocznij analizę"):
    if not product or not audience:
        st.warning("❗ Wprowadź opis produktu i grupy docelowej.")
        st.stop()

    st.info("📂 Wczytuję listę domen z pliku `sites.txt`...")
    sites = load_sites_from_file()

    if not sites:
        st.warning("Brak domen do analizy.")
        st.stop()

    query_embedding = model.encode(product + " " + audience, convert_to_tensor=True)
    results = []

    st.info("🔬 Trwa analiza... może to potrwać kilka minut.")
    progress = st.progress(0)
    total = len(sites)

    for idx, site in enumerate(sites):
        links = extract_links(site, limit=30)
        best_score = 0.0
        best_url = ""

        for link in links:
            text = extract_text(link)
            if text:
                page_embedding = model.encode(text, convert_to_tensor=True)
                score = util.pytorch_cos_sim(query_embedding, page_embedding).item()
                if score > best_score:
                    best_score = score
                    best_url = link
            time.sleep(0.5)

        results.append({
            "Domena": site,
            "Najlepiej dopasowana podstrona": best_url,
            "Dopasowanie (%)": round(best_score * 100, 2)
        })
        progress.progress((idx + 1) / total)

    df = pd.DataFrame(sorted(results, key=lambda x: -x["Dopasowanie (%)"]))
    st.success(f"✅ Przeanalizowano {len(df)} witryn.")
    st.dataframe(df)
    st.download_button("📥 Pobierz wyniki jako CSV", df.to_csv(index=False), file_name="ranking_stron.csv")
