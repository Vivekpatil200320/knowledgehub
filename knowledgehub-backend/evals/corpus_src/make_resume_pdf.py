"""Generates the fictional resume fixture used by the eval suite.

The person is invented. The structure is what matters: a contact header, skills
spread across sections, and an EDUCATION block with two degrees at two distinct
universities. That block is the regression target — with relevance-ordered context
and small chunks, a model mis-pairs the degrees and invents a third. Committed as a
binary fixture (evals/corpus/candidate-profile.pdf); regenerate with:

    python evals/corpus_src/make_resume_pdf.py

fpdf2 is only needed to regenerate, not to run the app or the evals.
"""

from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).resolve().parents[1] / "corpus" / "candidate-profile.pdf"

LINES = [
    ("H1", "Priya Nair"),
    ("META", "priya.nair@example.com | +1-555-0142 | Austin, TX | github.com/example"),
    ("", ""),
    ("H2", "SUMMARY"),
    ("P", "Backend engineer focused on data platforms and search. Owns services "
          "end to end, from schema design through deployment and on-call."),
    ("", ""),
    ("H2", "SKILLS"),
    ("P", "Languages: Python, Go, SQL. Data: PostgreSQL, Kafka, Qdrant, dbt. "
          "Infra: Docker, Kubernetes, Terraform, AWS."),
    ("", ""),
    ("H2", "EXPERIENCE"),
    ("H3", "Senior Backend Engineer, Meridian Freight (2023 - present)"),
    ("P", "Rebuilt the shipment-tracking pipeline on Kafka, cutting event lag from "
          "40 seconds to under 2. Led the migration of the pricing service to Go, and "
          "introduced contract tests that dropped production incidents by roughly a third."),
    ("H3", "Backend Engineer, Cobalt Health (2020 - 2023)"),
    ("P", "Built the patient-records search API on PostgreSQL full-text search, later "
          "adding a vector layer for semantic lookup. Owned the service's SLO and on-call "
          "rotation, and mentored two junior engineers through their first production launches."),
    ("H3", "Software Engineer, Lantern Analytics (2018 - 2020)"),
    ("P", "Wrote ETL jobs in Python feeding a reporting warehouse, and built an internal "
          "dashboard used daily by the operations team to track pipeline freshness."),
    ("", ""),
    ("H2", "EDUCATION"),
    ("H3", "Master of Science in Computer Science"),
    ("P", "Ashford Institute of Technology, Boston -- 2018 to 2020 -- GPA 3.8/4.0"),
    ("H3", "Bachelor of Engineering in Information Technology"),
    ("P", "Westbrook University, Portland -- 2014 to 2018 -- GPA 3.6/4.0"),
    ("", ""),
    ("H2", "CERTIFICATIONS"),
    ("P", "AWS Certified Solutions Architect - Associate (2022). "
          "Certified Kubernetes Administrator (2023)."),
]


def build() -> None:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_margins(20, 20, 20)
    for kind, text in LINES:
        if kind == "H1":
            pdf.set_font("Helvetica", "B", 20)
            pdf.multi_cell(pdf.epw, 9, text)
        elif kind == "META":
            pdf.set_font("Helvetica", "", 10)
            pdf.multi_cell(pdf.epw, 6, text)
        elif kind == "H2":
            pdf.set_font("Helvetica", "B", 13)
            pdf.multi_cell(pdf.epw, 8, text)
        elif kind == "H3":
            pdf.set_font("Helvetica", "B", 11)
            pdf.multi_cell(pdf.epw, 7, text)
        elif kind == "P":
            pdf.set_font("Helvetica", "", 11)
            pdf.multi_cell(pdf.epw, 6, text)
        else:
            pdf.ln(3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    build()
