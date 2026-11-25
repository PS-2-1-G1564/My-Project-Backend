
import os
import requests

OPENALEX_API = "https://api.openalex.org/works"


def reconstruct_abstract(inverted_index):
    """Convert OpenAlex inverted-index abstract into readable text."""

    if not inverted_index:
        return None

    words = {}

    for word, positions in inverted_index.items():
        for position in positions:
            words[position] = word

    if not words:
        return None

    abstract = " ".join(
        words[position]
        for position in sorted(words)
    )

    return abstract.strip() or None


def get_paper_abstract(doi):
    """Retrieve one research-paper abstract from OpenAlex using DOI."""

    if not doi:
        return None

    doi = doi.strip()

    if doi.lower().startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]

    if doi.lower().startswith("http://doi.org/"):
        doi = doi[len("http://doi.org/"):]

    email = os.getenv("OPENALEX_EMAIL")

    params = {
        "filter": f"doi:https://doi.org/{doi}",
        "per-page": 1
    }

    if email:
        params["mailto"] = email

    try:
        response = requests.get(
            OPENALEX_API,
            params=params,
            timeout=20
        )

        response.raise_for_status()

        results = response.json().get("results", [])

        if not results:
            return None

        inverted_index = results[0].get(
            "abstract_inverted_index"
        )

        return reconstruct_abstract(inverted_index)

    except requests.RequestException as error:
        print(f"OpenAlex request failed for DOI {doi}: {error}")
        return None

    except (ValueError, TypeError, KeyError) as error:
        print(f"Abstract processing failed for DOI {doi}: {error}")
        return None


def get_papers_with_abstracts(papers):
    """
    Enrich Crossref paper records with abstracts from OpenAlex.
    Existing paper metadata is preserved.
    """

    enriched_papers = []

    for paper in papers:

        updated_paper = paper.copy()

        doi = updated_paper.get("doi")

        abstract = None

        if doi:
            abstract = get_paper_abstract(doi)

        # Prefer the OpenAlex abstract when available.
        # Otherwise retain an existing Crossref abstract.
        if abstract:
            updated_paper["abstract"] = abstract
            updated_paper["abstract_source"] = "OpenAlex"

        elif updated_paper.get("abstract"):
            updated_paper["abstract_source"] = "Crossref"

        else:
            updated_paper["abstract"] = None
            updated_paper["abstract_source"] = None

        updated_paper["abstract_available"] = bool(
            updated_paper.get("abstract")
        )

        enriched_papers.append(updated_paper)

    return enriched_papers
