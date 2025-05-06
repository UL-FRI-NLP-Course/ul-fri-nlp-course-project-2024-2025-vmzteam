import pandas as pd
from datetime import datetime, timedelta
import re

# new imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load CSV
df = pd.read_csv('../podatki.csv', sep=';')
df['ts'] = pd.to_datetime(df['Datum'], format='mixed', dayfirst=True)

# 2. Your RTF metadata/body as before
rtf_ts   = datetime(2024,3,2,6,0)
rtf_body = """Na obalni hitri cesti je zaradi del med Koprom in Izolo v predoru Markovec zaprt prehitevalni pas proti Ljubljani.

Regionalna cesta Krka-Žužemberk bo prav tako zaradi del v Žužemberku zaprta do ponedeljka.

Cesta čez prelaz Vršič je zaradi zimskih razmer zaprta.

Na primorski avtocesti bo zaradi del danes od 11-ih do jutri  do 17-ih zaprt vozni pas med Postojno in Razdrtim proti Kopru.

"""

# 3. Time-window filter
candidates = df[
    (df.ts >= rtf_ts - timedelta(minutes=180)) &
    (df.ts <= rtf_ts + timedelta(minutes=5))
].copy()

# helper to strip HTML
def strip_html(text):
    if not isinstance(text, str):
        return ""
    return re.sub(r'<.*?>', '', text).strip()

# 4. Build a “document” for each candidate
docs = []
for idx, row in candidates.iterrows():
    parts = []
    for col in [
        "TitlePomembnoSLO","ContentPomembnoSLO",
        "TitleNesreceSLO","ContentNesreceSLO",
        "TitleZastojiSLO","ContentZastojiSLO",
        "TitleVremeSLO","ContentVremeSLO",
        "TitleOvireSLO","ContentOvireSLO",
        "TitleDeloNaCestiSLO","ContentDeloNaCestiSLO",
        "TitleOpozorilaSLO","ContentOpozorilaSLO",
        "ContentMednarodneInformacijeSLO",
        "TitleSplosnoSLO","ContentSplosnoSLO"
    ]:
        parts.append(strip_html(row.get(col, "")))
    docs.append("\n".join(parts))

# 5. Vectorize
#    - we fit on all candidate docs + the single RTF body,
#      so term-space is shared
vectorizer = TfidfVectorizer(
    lowercase=True,
    strip_accents='unicode'
)

# create a corpus: all candidate docs, plus rtf_body at the end
corpus = docs + [rtf_body]
tfidf = vectorizer.fit_transform(corpus)

# 6. Compute cosine similarities between each candidate and the rtf_body
#    the rtf_body vector is the last row of tfidf
rtf_vec     = tfidf[-1]
cand_vecs   = tfidf[:-1]
sims        = cosine_similarity(cand_vecs, rtf_vec.reshape(1, -1)).flatten()

# 7. Pick best match
candidates = candidates.reset_index(drop=True)
candidates['tfidf_sim'] = sims
best_idx   = candidates['tfidf_sim'].idxmax()
best_row   = candidates.loc[best_idx]

print("Best match (similarity = {:.3f}):".format(best_row['tfidf_sim']))
print(best_row)
