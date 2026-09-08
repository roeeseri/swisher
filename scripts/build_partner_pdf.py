"""Build the partner brief PDF from its versioned Hebrew Markdown source."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from bidi.algorithm import get_display
from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('docs/SWISHER_PARTNER_BRIEF_HE.md'))
    parser.add_argument('--output', type=Path, default=Path('output/pdf/Swisher_Partner_Brief_HE.pdf'))
    parser.add_argument('--font-dir', type=Path, default=Path('/System/Library/Fonts/Supplemental'))
    args = parser.parse_args()
    pdfmetrics.registerFont(TTFont('Body', str(args.font_dir / 'Arial.ttf')))
    pdfmetrics.registerFont(TTFont('Title', str(args.font_dir / 'Arial Bold.ttf')))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    document = canvas.Canvas(str(args.output), pagesize=A4)
    document.setTitle('Swisher - Team MVP and Year One Plan')
    document.setAuthor('Swisher founding team')
    width, height = A4
    pages = args.source.read_text().split('\n---\n')
    margin = 48

    def visual(text):
        return get_display(text, base_dir='R')

    def lines(text, font, size):
        words, result, row = text.split(), [], ''
        for word in words:
            trial = (row + ' ' + word).strip()
            if pdfmetrics.stringWidth(visual(trial), font, size) > width - 2*margin and row:
                result.append(row)
                row = word
            else:
                row = trial
        if row:
            result.append(row)
        return result

    for index, page in enumerate(pages):
        document.setFillColor(HexColor('#101f34'))
        document.rect(0, height-19, width, 19, fill=1, stroke=0)
        document.setFillColor(HexColor('#ea864c'))
        document.setFont('Title', 10)
        document.drawString(margin, height-47, 'SWISHER / PARTNER BRIEF')
        document.setFillColor(HexColor('#617085'))
        document.setFont('Body', 9)
        document.drawRightString(width-margin, height-47, '08.09.2026  |  v1.0')
        y = height-88
        for block in page.strip().split('\n\n'):
            block = block.strip()
            if not block:
                continue
            if block.startswith('# '):
                font, size, leading, gap = 'Title', 23, 29, 17
                text = block[2:]
                color = '#14243b'
            elif block.startswith('## '):
                font, size, leading, gap = 'Title', 12.5, 17, 5
                text = block[3:]
                color = '#b95222'
            else:
                font, size, leading, gap = 'Body', 11.2, 16.3, 11
                text = block
                color = '#25364c'
            document.setFont(font, size)
            document.setFillColor(HexColor(color))
            # Preserve list entries and source URLs as individually wrapped paragraphs.
            pieces = text.split('\n') if ('\n-' in text or 'https://' in text) else [re.sub(r'\s+', ' ', text)]
            for piece in pieces:
                if piece.startswith('https://'):
                    document.setFont('Body', 7.4)
                    document.drawString(margin, y, piece)
                    document.linkURL(piece, (margin,y-2,width-margin,y+9), relative=0)
                    document.setFont(font, size)
                    y -= 13
                    continue
                for line in lines(piece, font, size):
                    if y < 65:
                        raise ValueError(f'Page {index+1} overflows. Shorten text or adjust layout.')
                    document.drawRightString(width-margin, y, visual(line))
                    y -= leading
            y -= gap
        document.setStrokeColor(HexColor('#dfe6ef'))
        document.line(margin, 43, width-margin, 43)
        document.setFont('Body', 9)
        document.setFillColor(HexColor('#617085'))
        document.drawString(margin, 27, f'{index+1:02d} / {len(pages):02d}')
        document.drawRightString(width-margin, 27, visual('מסמך פנימי | יעדי התוכנית אינם הבטחת ביצועים'))
        document.showPage()
    document.save()
    print(f'Created {args.output}: {len(pages)} pages')


if __name__ == '__main__':
    main()
