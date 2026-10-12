
from io import BytesIO
from urllib.parse import urlparse

import pdfplumber
import trafilatura
from bs4 import BeautifulSoup


def parse_page(url, html="", pdf_bytes=None):
    title = ""
    text = ""

    if pdf_bytes:
        try:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                pages = []
                for page in pdf.pages:
                    pages.append(page.extract_text() or "")

                    for table in page.extract_tables():
                        for row in table:
                            cells = [str(cell or "").strip() for cell in row]
                            pages.append(" | ".join(cells))

                text = "\n".join(pages).strip()
        except Exception:
            text = ""

    elif html:
        soup = BeautifulSoup(html, "lxml")
        title_tag = soup.find("title")
        title = title_tag.get_text(" ", strip=True) if title_tag else ""

        extracted = trafilatura.extract(
            html,
            include_tables=True,
            include_links=False,
            favor_precision=True,
        )

        if extracted:
            text = extracted.strip()
        else:
            for element in soup(["script", "style", "noscript", "nav", "footer"]):
                element.decompose()

            text = soup.get_text(" ", strip=True)

    if not title:
        path = urlparse(url).path.rstrip("/")
        title = path.split("/")[-1] or urlparse(url).netloc

    return title, text
