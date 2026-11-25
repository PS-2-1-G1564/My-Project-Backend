
"""
literature_service.py
Retrieves journal-paper metadata from Crossref.
Crossref search results are not automatically verified scientific evidence.
"""

import re
import requests
from html import unescape

CROSSREF_API = "https://api.crossref.org/works"


def clean_abstract(raw_abstract):
    if not raw_abstract:
        return None

    text = re.sub(r"<[^>]+>", " ", raw_abstract)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()

    return text or None


def normalize_text(value):
    if not value:
        return ""
    value = str(value).translate(
        str.maketrans("₀₁₂₃₄₅₆₇₈₉", "0123456789")
    )
    return re.sub(r"\s+", "", value).lower()


def formula_mentioned(formula, text):
    """Check for the complete formula, not a shared chemical fragment."""
    normalized_formula = normalize_text(formula)
    normalized_text = normalize_text(text)

    return bool(
        normalized_formula and normalized_formula in normalized_text
    )


def search_research_papers(formula, rows=10):

    if not isinstance(formula, str) or not formula.strip():
        raise ValueError("A valid chemical formula is required.")

    formula = formula.strip()
    rows = max(1, min(int(rows), 20))

    params = {
        "query": formula,
        "rows": rows,
        "select": (
            "DOI,title,author,published,published-print,"
            "published-online,container-title,publisher,"
            "type,URL,abstract"
        )
    }

    headers = {
        "User-Agent": "CrystalVolts/1.0 (research material information)"
    }

    try:
        response = requests.get(
            CROSSREF_API,
            params=params,
            headers=headers,
            timeout=20
        )
        response.raise_for_status()
        data = response.json()

    except requests.RequestException as error:
        return {
            "status": "error",
            "message": f"Crossref request failed: {error}",
            "papers": [],
            "related_background": [],
            "excluded_records": []
        }

    except ValueError as error:
        return {
            "status": "error",
            "message": f"Invalid Crossref response: {error}",
            "papers": [],
            "related_background": [],
            "excluded_records": []
        }

    items = data.get("message", {}).get("items", [])

    papers = []
    related_background = []
    excluded_records = []

    for item in items:

        titles = item.get("title") or []
        title = titles[0] if titles else "Title unavailable"

        publication_type = item.get("type", "")

        if publication_type != "journal-article":
            excluded_records.append({
                "title": title,
                "type": publication_type,
                "reason": "Not classified as a journal article"
            })
            continue

        abstract = clean_abstract(item.get("abstract"))

        title_match = formula_mentioned(formula, title)
        abstract_match = formula_mentioned(formula, abstract or "")

        authors = []
        for author in item.get("author", []):
            name = (
                f"{author.get('given', '')} "
                f"{author.get('family', '')}"
            ).strip()

            if name:
                authors.append(name)

        year = None

        for date_field in (
            "published",
            "published-print",
            "published-online"
        ):
            date_parts = item.get(date_field, {}).get(
                "date-parts", []
            )

            if date_parts and date_parts[0]:
                year = date_parts[0][0]
                break

        doi = item.get("DOI")
        journals = item.get("container-title") or []

        paper = {
            "title": title,
            "authors": authors,
            "year": year,
            "journal": journals[0] if journals else None,
            "publisher": item.get("publisher"),
            "type": publication_type,
            "doi": doi,
            "url": (
                f"https://doi.org/{doi}"
                if doi else item.get("URL")
            ),
            "abstract": abstract,
            "abstract_available": bool(abstract),
            "source": "Crossref",
            "material_formula": formula,
            "formula_found_in_title": title_match,
            "formula_found_in_abstract": abstract_match,
            "scientific_claims_verified": False
        }

        if title_match or abstract_match:
            paper["relevance"] = "Exact formula mentioned"
            papers.append(paper)
        else:
            paper["relevance"] = (
                "Exact material relevance not confirmed"
            )
            related_background.append(paper)

    return {
        "status": "success",
        "query": formula,
        "total_returned": len(items),
        "total_journal_articles": len(papers),
        "total_related_background": len(related_background),
        "total_excluded": len(excluded_records),
        "papers": papers,
        "related_background": related_background,
        "excluded_records": excluded_records,
        "message": (
            "Exact-formula matches are separated from other search "
            "results. Original papers must still be checked."
        )
    }


def build_literature_description(formula, rows=10):

    result = search_research_papers(
        formula=formula,
        rows=rows
    )

    return {
        "working_mechanism": [],
        "advantages": [],
        "limitations": [],

        "references": result.get("papers", []),
        "related_background": result.get(
            "related_background", []
        ),

        "search_status": result.get("status"),
        "search_message": result.get("message"),
        "search_query": formula,

        "total_journal_articles": result.get(
            "total_journal_articles", 0
        ),

        "total_related_background": result.get(
            "total_related_background", 0
        ),

        "total_excluded": result.get(
            "total_excluded", 0
        )
    }
