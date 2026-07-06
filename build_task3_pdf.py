from __future__ import annotations

import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.pdfmetrics import registerFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer


BASE_DIR = Path(__file__).resolve().parent
REPORT_MD = BASE_DIR / "report" / "task3_report_final.md"
OUTPUT_PDF = BASE_DIR / "report" / "姓名+TASK3.pdf"


def build_styles():
    registerFont(UnicodeCIDFont("STSong-Light"))
    stylesheet = getSampleStyleSheet()

    title = ParagraphStyle(
        "TitleCN",
        parent=stylesheet["Title"],
        fontName="STSong-Light",
        fontSize=18,
        leading=24,
        alignment=TA_CENTER,
        textColor=colors.black,
        spaceAfter=10,
    )
    heading1 = ParagraphStyle(
        "Heading1CN",
        parent=stylesheet["Heading1"],
        fontName="STSong-Light",
        fontSize=14,
        leading=21,
        alignment=TA_LEFT,
        textColor=colors.black,
        spaceBefore=8,
        spaceAfter=5,
    )
    heading2 = ParagraphStyle(
        "Heading2CN",
        parent=stylesheet["Heading2"],
        fontName="STSong-Light",
        fontSize=12,
        leading=18,
        alignment=TA_LEFT,
        textColor=colors.black,
        spaceBefore=6,
        spaceAfter=4,
    )
    body = ParagraphStyle(
        "BodyCN",
        parent=stylesheet["BodyText"],
        fontName="STSong-Light",
        fontSize=10.5,
        leading=15.75,
        alignment=TA_JUSTIFY,
        textColor=colors.black,
        wordWrap="CJK",
        spaceBefore=0,
        spaceAfter=0,
        firstLineIndent=0,
    )
    bullet = ParagraphStyle(
        "BulletCN",
        parent=body,
        leftIndent=14,
        firstLineIndent=-10,
    )
    caption = ParagraphStyle(
        "CaptionCN",
        parent=body,
        alignment=TA_CENTER,
        fontSize=10,
        leading=15,
    )
    return title, heading1, heading2, body, bullet, caption


def escape_html(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def build_story():
    title_style, h1_style, h2_style, body_style, bullet_style, caption_style = build_styles()
    story = []
    lines = REPORT_MD.read_text(encoding="utf-8").splitlines()

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 0.15 * cm))
            continue

        if line.startswith("# "):
            story.append(Paragraph(escape_html(line[2:]), title_style))
            story.append(Spacer(1, 0.2 * cm))
            continue

        if line.startswith("## "):
            story.append(Paragraph(escape_html(line[3:]), h1_style))
            continue

        if line.startswith("### "):
            story.append(Paragraph(escape_html(line[4:]), h2_style))
            continue

        if line.startswith("- "):
            content = escape_html("• " + line[2:])
            story.append(Paragraph(content, bullet_style))
            continue

        if re.match(r"^\d+\.\s", line):
            content = escape_html(line)
            story.append(Paragraph(content, body_style))
            continue

        if line.startswith("文件位置："):
            match = re.search(r"`([^`]+)`", line)
            if match:
                image_path = BASE_DIR / match.group(1)
                if image_path.exists():
                    story.append(Spacer(1, 0.12 * cm))
                    img = Image(str(image_path))
                    max_width = 16.2 * cm
                    max_height = 9.4 * cm
                    scale = min(max_width / img.imageWidth, max_height / img.imageHeight)
                    img.drawWidth = img.imageWidth * scale
                    img.drawHeight = img.imageHeight * scale
                    story.append(img)
                    story.append(Spacer(1, 0.1 * cm))
            continue

        story.append(Paragraph(escape_html(line), body_style))

    return story


def main():
    doc = SimpleDocTemplate(
        str(OUTPUT_PDF),
        pagesize=A4,
        leftMargin=2.2 * cm,
        rightMargin=2.2 * cm,
        topMargin=2.2 * cm,
        bottomMargin=2.2 * cm,
        title="Task3 双均线策略报告",
        author="Codex",
    )
    doc.build(build_story())
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
