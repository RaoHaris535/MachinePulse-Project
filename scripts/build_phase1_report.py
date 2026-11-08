"""Build the four-page MachinePulse Phase 1 submission report."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs" / "submission" / "evidence"
NAVY = colors.HexColor("#0F172A")
SLATE = colors.HexColor("#334155")
MUTED = colors.HexColor("#64748B")
LIGHT = colors.HexColor("#E2E8F0")
PALE = colors.HexColor("#F1F5F9")
CYAN = colors.HexColor("#0891B2")
GREEN = colors.HexColor("#15803D")
AMBER = colors.HexColor("#B45309")


def _styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "Title",
            parent=sample["Title"],
            fontName="Helvetica-Bold",
            fontSize=27,
            leading=31,
            textColor=NAVY,
            alignment=TA_LEFT,
            spaceAfter=8,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle",
            parent=sample["Normal"],
            fontName="Helvetica",
            fontSize=12,
            leading=17,
            textColor=SLATE,
            spaceAfter=13,
        ),
        "h1": ParagraphStyle(
            "H1",
            parent=sample["Heading1"],
            fontName="Helvetica-Bold",
            fontSize=19,
            leading=23,
            textColor=NAVY,
            spaceBefore=2,
            spaceAfter=9,
        ),
        "h2": ParagraphStyle(
            "H2",
            parent=sample["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=12,
            leading=15,
            textColor=CYAN,
            spaceBefore=7,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13.1,
            textColor=SLATE,
            spaceAfter=6,
        ),
        "small": ParagraphStyle(
            "Small",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8,
            leading=10.5,
            textColor=MUTED,
            spaceAfter=3,
        ),
        "caption": ParagraphStyle(
            "Caption",
            parent=sample["BodyText"],
            fontName="Helvetica-Oblique",
            fontSize=7.5,
            leading=10,
            textColor=MUTED,
            alignment=TA_CENTER,
            spaceBefore=3,
            spaceAfter=6,
        ),
        "card": ParagraphStyle(
            "Card",
            parent=sample["BodyText"],
            fontName="Helvetica-Bold",
            fontSize=10,
            leading=13,
            textColor=NAVY,
            alignment=TA_CENTER,
        ),
        "link": ParagraphStyle(
            "Link",
            parent=sample["BodyText"],
            fontName="Helvetica",
            fontSize=8.2,
            leading=11,
            textColor=CYAN,
            spaceAfter=4,
        ),
        "code": ParagraphStyle(
            "Code",
            parent=sample["Code"],
            fontName="Courier",
            fontSize=7.7,
            leading=10.5,
            textColor=colors.white,
            backColor=NAVY,
            borderPadding=8,
            spaceBefore=4,
            spaceAfter=7,
        ),
    }


def _scaled_image(path: Path, width: float, crop: tuple[int, int, int, int] | None = None) -> Image:
    with PILImage.open(path) as source:
        image = source.convert("RGB")
        if crop is not None:
            image = image.crop(crop)
        pixel_width, pixel_height = image.size
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    height = width * pixel_height / pixel_width
    return Image(buffer, width=width, height=height)


def _result_table(styles: dict[str, ParagraphStyle]) -> Table:
    data = [
        ["Model", "Test F1", "ROC-AUC", "Avg. precision"],
        ["Logistic Regression", "0.0000", "0.0249", "0.0023"],
        ["LSTM Autoencoder", "0.0110", "0.9825", "0.2787"],
    ]
    table = Table(data, colWidths=[2.55 * inch, 1.0 * inch, 1.05 * inch, 1.25 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
                ("GRID", (0, 0), (-1, -1), 0.5, LIGHT),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    return table


def _card_row(styles: dict[str, ParagraphStyle]) -> Table:
    cards = [
        Paragraph(
            "<font color='#15803D'>PASSED</font><br/><font size='8'>data validation</font>",
            styles["card"],
        ),
        Paragraph("1,516,948<br/><font size='8'>raw sensor rows</font>", styles["card"]),
        Paragraph("0<br/><font size='8'>nulls or duplicate times</font>", styles["card"]),
    ]
    table = Table([cards], colWidths=[2.05 * inch] * 3, hAlign="CENTER")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), PALE),
                ("BOX", (0, 0), (-1, -1), 0.6, LIGHT),
                ("INNERGRID", (0, 0), (-1, -1), 0.6, colors.white),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
            ]
        )
    )
    return table


def _header_footer(canvas, document) -> None:  # noqa: ANN001
    canvas.saveState()
    page_width, page_height = letter
    canvas.setFillColor(NAVY)
    canvas.rect(0, page_height - 0.28 * inch, page_width, 0.28 * inch, fill=1, stroke=0)
    canvas.setFont("Helvetica-Bold", 7.5)
    canvas.setFillColor(colors.white)
    canvas.drawString(0.58 * inch, page_height - 0.19 * inch, "MACHINEPULSE / PHASE 1")
    canvas.setStrokeColor(LIGHT)
    canvas.line(0.58 * inch, 0.47 * inch, page_width - 0.58 * inch, 0.47 * inch)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7.5)
    canvas.drawString(
        0.58 * inch,
        0.28 * inch,
        "Reproducible predictive maintenance with DVC and MLflow",
    )
    canvas.drawRightString(page_width - 0.58 * inch, 0.28 * inch, f"Page {document.page}")
    canvas.restoreState()


def build_report(output_path: Path) -> None:
    """Create the final four-page Phase 1 report."""
    styles = _styles()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = SimpleDocTemplate(
        output_path.as_posix(),
        pagesize=letter,
        rightMargin=0.58 * inch,
        leftMargin=0.58 * inch,
        topMargin=0.52 * inch,
        bottomMargin=0.62 * inch,
        title="MachinePulse Phase 1 Implementation Report",
        author="Rao Haris",
        subject="MLOps Phase 1 submission",
    )

    story = []

    # Page 1: project summary and outcomes.
    story.extend(
        [
            Spacer(1, 0.12 * inch),
            Paragraph("MachinePulse", styles["title"]),
            Paragraph(
                "Phase 1 implementation report - reproducible industrial "
                "time-series anomaly detection",
                styles["subtitle"],
            ),
            HRFlowable(width="100%", thickness=2, color=CYAN, spaceAfter=12),
            Paragraph("Implementation overview", styles["h1"]),
            Paragraph(
                "MachinePulse uses the UCI MetroPT-3 air-compressor sensor dataset to compare a "
                "class-balanced Logistic Regression baseline with a PyTorch LSTM Autoencoder. "
                "Phase 1 delivers a fully local and reproducible ML pipeline; deployment, APIs, "
                "containers, monitoring, and cloud services are intentionally deferred.",
                styles["body"],
            ),
            Paragraph("Repository and tracking links", styles["h2"]),
            Paragraph(
                "<b>GitHub:</b> <link href='https://github.com/RaoHaris535/MachinePulse-Project'>"
                "github.com/RaoHaris535/MachinePulse-Project</link>",
                styles["link"],
            ),
            Paragraph(
                "<b>DagsHub:</b> <link href='https://dagshub.com/RaoHaris535/MachinePulse-Project'>"
                "dagshub.com/RaoHaris535/MachinePulse-Project</link>",
                styles["link"],
            ),
            Paragraph(
                "<b>MLflow:</b> <link href='https://dagshub.com/RaoHaris535/"
                "MachinePulse-Project.mlflow/#/experiments/0'>remote "
                "machinepulse-phase1 experiment</link>",
                styles["link"],
            ),
            Paragraph("Phase 1 completion", styles["h2"]),
        ]
    )
    checklist = [
        ["Data", "Official-source ingestion, schema/null/range/time validation, DVC remote"],
        ["Systems", "Logistic Regression baseline and LSTM Autoencoder main model"],
        ["Tracking", "Remote MLflow parameters, metrics, artifacts, and registered Staging model"],
        ["Pipeline", "Six named DVC stages reproducible with one dvc repro command"],
        ["Quality", "17 automated tests, Ruff linting, deterministic seeds, no raw data in Git"],
    ]
    checklist_table = Table(checklist, colWidths=[1.05 * inch, 5.35 * inch])
    checklist_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), PALE),
                ("TEXTCOLOR", (0, 0), (0, -1), CYAN),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.4),
                ("LEADING", (0, 0), (-1, -1), 11),
                ("GRID", (0, 0), (-1, -1), 0.5, LIGHT),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend(
        [
            checklist_table,
            Spacer(1, 8),
            Paragraph("Held-out test results", styles["h2"]),
            _result_table(styles),
            Spacer(1, 7),
            Paragraph(
                "The main model sharply improves threshold-independent anomaly ranking. The low "
                "fixed-threshold F1 is retained as an honest limitation caused by temporal "
                "distribution shift between failure periods; the held-out test set was not used "
                "for threshold tuning.",
                styles["small"],
            ),
            PageBreak(),
        ]
    )

    # Page 2: data pipeline and DVC.
    story.extend(
        [
            Paragraph("Data pipeline and versioning", styles["h1"]),
            _card_row(styles),
            Paragraph("Ingestion and validation", styles["h2"]),
            Paragraph(
                "The ingest stage streams the official UCI ZIP, extracts only the configured "
                "MetroPT-3 CSV, and records SHA-256 hashes in a source manifest. Validation scans "
                "the 1.5 million rows in chunks and checks the required schema, row-count bounds, "
                "nulls, numeric conversion, finite values, physical ranges, binary domains, "
                "timestamp parsing, ordering, and duplicates. Any failed check exits non-zero and "
                "blocks preprocessing.",
                styles["body"],
            ),
            Paragraph("Preprocessing and sequence generation", styles["h2"]),
            Paragraph(
                "Raw 10-second readings are aggregated to 5-minute intervals. Published failure "
                "windows create active-failure labels and a one-hour pre-failure warning target. "
                "Chronological train/validation/test splits isolate later incidents. Scaling is "
                "fit only on normal training rows, and the LSTM receives 12-step (one-hour) "
                "sequences that never cross missing-time gaps.",
                styles["body"],
            ),
        ]
    )
    dag_image = _scaled_image(EVIDENCE / "dvc-dag.png", 3.15 * inch, crop=(0, 85, 850, 1300))
    dvc_notes = [
        Paragraph("<b>Meaningful DVC outputs</b>", styles["h2"]),
        Paragraph(
            "Raw and processed data, model bundles, figures, and JSON metrics are cached by DVC "
            "and synchronized to the DagsHub S3-compatible remote. Git stores only code, "
            "configuration, dvc.yaml, and dvc.lock.",
            styles["body"],
        ),
        Paragraph("<b>Reproducibility controls</b>", styles["h2"]),
        Paragraph(
            "All stage dependencies and outputs are explicit. Seeds, split boundaries, failure "
            "windows, model hyperparameters, and early-stopping settings are versioned in YAML. "
            "The validation report is a dependency of preprocessing, making validation a true "
            "training gate.",
            styles["body"],
        ),
        Paragraph("<b>Remote status</b>", styles["h2"]),
        Paragraph(
            "After the final run, dvc push reported everything up to date and dvc status -c "
            "confirmed that the local cache and remote origin were synchronized.",
            styles["body"],
        ),
    ]
    dvc_layout = Table([[dag_image, dvc_notes]], colWidths=[3.4 * inch, 3.0 * inch])
    dvc_layout.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (0, -1), 10),
                ("RIGHTPADDING", (1, 0), (1, -1), 0),
            ]
        )
    )
    story.extend(
        [
            Spacer(1, 3),
            dvc_layout,
            Paragraph(
                "Figure 1. Live DVC pipeline output after validation was made a training gate.",
                styles["caption"],
            ),
            PageBreak(),
        ]
    )

    # Page 3: models and remote tracking.
    story.extend(
        [
            Paragraph("Models and remote experiment tracking", styles["h1"]),
            Paragraph(
                "The Logistic Regression baseline uses class balancing and validation-F1 threshold "
                "selection on tabular intervals. The LSTM Autoencoder is trained only on normal "
                "training sequences, uses reconstruction MSE as its anomaly score, and restores "
                "the best validation-normal checkpoint through early stopping.",
                styles["body"],
            ),
            _result_table(styles),
            Spacer(1, 8),
            _scaled_image(EVIDENCE / "mlflow-experiment-runs.png", 6.35 * inch),
            Paragraph(
                "Figure 2. Remote machinepulse-phase1 experiment with multiple completed baseline "
                "and main-model runs; the red cross records the earlier failed registration "
                "attempt.",
                styles["caption"],
            ),
            _scaled_image(
                EVIDENCE / "mlflow-lstm-run.png",
                6.35 * inch,
                crop=(0, 0, 1600, 820),
            ),
            Paragraph(
                "Figure 3. Successful LSTM run showing logged losses, validation metrics, "
                "hyperparameters, tags, Finished status, and the registered-model association.",
                styles["caption"],
            ),
            PageBreak(),
        ]
    )

    # Page 4: registry, reproducibility, and limitations.
    story.extend(
        [
            Paragraph("Model registry and reproducibility", styles["h1"]),
            _scaled_image(EVIDENCE / "mlflow-model-staging.png", 6.35 * inch),
            Paragraph(
                "Figure 4. MachinePulse-LSTM-Autoencoder version 1 registered on the remote MLflow "
                "server with lifecycle stage Staging.",
                styles["caption"],
            ),
            Paragraph("Repeatable execution", styles["h2"]),
            Paragraph(
                "A fresh authorized clone can restore DVC outputs and reproduce the entire "
                "pipeline with the following commands. Remote credentials are stored only in the "
                "ignored .dvc/config.local file; raw data, model weights, .env files, and tokens "
                "are excluded from Git.",
                styles["body"],
            ),
            Paragraph(
                "git clone https://github.com/RaoHaris535/MachinePulse-Project.git<br/>"
                "cd MachinePulse-Project &amp;&amp; uv sync --group dev<br/>"
                "source .venv/bin/activate &amp;&amp; dvc pull<br/>"
                "dvc repro &amp;&amp; pytest -q &amp;&amp; ruff check .<br/>"
                "dvc push &amp;&amp; dvc status -c",
                styles["code"],
            ),
            Paragraph("Evaluation interpretation and limitation", styles["h2"]),
            Paragraph(
                "The chronological split exposes severe incident-to-incident distribution shift. "
                "The baseline transfers poorly to the final incident. The LSTM ranks test "
                "anomalies well (ROC-AUC 0.9825; average precision 0.2787), but the "
                "validation-selected threshold predicts every test sequence as anomalous, "
                "producing low precision and F1. This is documented rather than hidden or "
                "corrected using test labels. Future work should add calibration across multiple "
                "historical failure periods and drift monitoring, which are outside Phase 1.",
                styles["body"],
            ),
            Paragraph("Conclusion", styles["h2"]),
            Paragraph(
                "Phase 1 satisfies the required local MLOps foundation: source-controlled code and "
                "configuration, DVC-versioned artifacts on a remote, validation-gated processing, "
                "two evaluated systems, remote MLflow tracking, a Staging registered model, and an "
                "end-to-end named pipeline. The repository is ready for later deployment and "
                "monitoring phases without mixing those concerns into this submission.",
                styles["body"],
            ),
            Spacer(1, 6),
            HRFlowable(width="100%", thickness=1, color=LIGHT),
            Paragraph(
                "Prepared 8 November 2026 | Dataset: MetroPT-3, UCI Machine Learning Repository "
                "(DOI 10.24432/C5VW3R)",
                styles["small"],
            ),
        ]
    )

    document.build(story, onFirstPage=_header_footer, onLaterPages=_header_footer)


def main() -> None:
    """Parse the output path and build the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "submission" / "MachinePulse-Phase1-Report.pdf",
    )
    args = parser.parse_args()
    build_report(args.output)


if __name__ == "__main__":
    main()
