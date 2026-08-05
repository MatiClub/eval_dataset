#!/usr/bin/env python3
"""Generate synthetic documents to diversify and balance the benchmark corpus.

Deterministic (fixed seed): re-running produces byte-identical content layouts.
Generated files carry a SYN- prefix so they are distinguishable from scraped data.

New categories:
    contract-photo, bank_statement-photo, form-photo, business_card-photo,
    presentation-photo, spreadsheet-photo, screenshot-photo, note-photo, letter-txt

Top-ups for underrepresented categories:
    medical-photo, warranty-photo, diploma-photo, syllabus-pdf

Requires: pillow, fpdf2 (see the `prep` optional dependency group).

Usage:
    python3 prep_scripts/generate_synthetic_docs.py [--data-dir data]
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

SEED = 20260804

FONT_DIR = Path("/usr/share/fonts/truetype")
FONTS = {
    "sans": FONT_DIR / "dejavu/DejaVuSans.ttf",
    "sans_bold": FONT_DIR / "dejavu/DejaVuSans-Bold.ttf",
    "serif": FONT_DIR / "dejavu/DejaVuSerif.ttf",
    "serif_bold": FONT_DIR / "dejavu/DejaVuSerif-Bold.ttf",
    "mono": FONT_DIR / "dejavu/DejaVuSansMono.ttf",
    "oblique": FONT_DIR / "freefont/FreeSansOblique.ttf",
}

FIRST_NAMES = ["Anna", "Piotr", "Maria", "Jan", "Katarzyna", "Tomasz", "Ewa", "Marek",
               "John", "Sarah", "Michael", "Emma", "David", "Laura", "James", "Olivia"]
LAST_NAMES = ["Kowalski", "Nowak", "Wiśniewska", "Zieliński", "Mazur", "Kaczmarek",
              "Smith", "Johnson", "Brown", "Davis", "Miller", "Wilson", "Moore", "Taylor"]
COMPANIES = ["Nordic Trade Sp. z o.o.", "Apex Solutions Ltd.", "BlueRiver Software",
             "Grand Hotel Warszawa", "TechNova GmbH", "Delta Logistics S.A.",
             "GreenLeaf Foods", "Orion Consulting", "Vertex Media", "Polar Energy"]
CITIES = ["Warszawa", "Kraków", "Berlin", "London", "Gdańsk", "Vienna", "Poznań", "Prague"]


def _font(name: str, size: int):
    from PIL import ImageFont
    path = FONTS[name]
    if not path.exists():
        path = FONTS["sans"]
    return ImageFont.truetype(str(path), size)


def _person(rng: random.Random) -> str:
    return f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"


def _date(rng: random.Random) -> str:
    return f"{rng.randint(1, 28):02d}.{rng.randint(1, 12):02d}.{rng.randint(2019, 2025)}"


def _paper(rng: random.Random, w: int, h: int, tint=(255, 255, 253)) -> Image.Image:
    img = Image.new("RGB", (w, h), tint)
    # subtle vignette / scan shading so pages don't look sterile
    shade = Image.new("L", (w, h), 0)
    sd = ImageDraw.Draw(shade)
    for i in range(6):
        x = rng.randint(-w // 2, w)
        y = rng.randint(-h // 2, h)
        r = rng.randint(w // 2, w)
        sd.ellipse([x, y, x + r, y + r], fill=rng.randint(4, 10))
    shade = shade.filter(ImageFilter.GaussianBlur(80))
    dark = Image.new("RGB", (w, h), (235, 233, 226))
    return Image.composite(dark, img, shade)


def _save(img: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".png":
        img.save(path)
    else:
        img.save(path, quality=88)
    print(f"wrote {path}")


# ---------------------------------------------------------------- contracts

CONTRACT_KINDS = [
    ("SERVICE AGREEMENT", "en"), ("EMPLOYMENT CONTRACT", "en"), ("LEASE AGREEMENT", "en"),
    ("NON-DISCLOSURE AGREEMENT", "en"), ("SALES CONTRACT", "en"), ("CONSULTING AGREEMENT", "en"),
    ("UMOWA O DZIEŁO", "pl"), ("UMOWA NAJMU LOKALU", "pl"), ("UMOWA ZLECENIE", "pl"),
    ("UMOWA O PRACĘ", "pl"), ("UMOWA SPRZEDAŻY", "pl"), ("LOAN AGREEMENT", "en"),
]

CLAUSES_EN = [
    "The Contractor shall perform the services described in Appendix A with due diligence.",
    "Payment shall be made within 14 days of receipt of a correctly issued invoice.",
    "Either party may terminate this agreement with 30 days written notice.",
    "All intellectual property created under this agreement belongs to the Client.",
    "This agreement is governed by the laws of the jurisdiction stated above.",
    "Confidential information shall not be disclosed to any third party.",
    "The monthly remuneration is fixed for the duration of the agreement.",
]
CLAUSES_PL = [
    "Wykonawca zobowiązuje się do wykonania dzieła z należytą starannością.",
    "Zapłata nastąpi w terminie 14 dni od dnia doręczenia prawidłowej faktury.",
    "Każda ze stron może rozwiązać umowę z zachowaniem 30-dniowego okresu wypowiedzenia.",
    "Wszelkie zmiany niniejszej umowy wymagają formy pisemnej pod rygorem nieważności.",
    "W sprawach nieuregulowanych stosuje się przepisy Kodeksu cywilnego.",
    "Wynajmujący oddaje Najemcy lokal do używania na cele mieszkaniowe.",
]


def gen_contracts(rng: random.Random, out: Path, count: int) -> None:
    for i in range(count):
        title, lang = CONTRACT_KINDS[i % len(CONTRACT_KINDS)]
        w, h = 820, 1120
        img = _paper(rng, w, h)
        d = ImageDraw.Draw(img)
        d.text((w // 2, 70), title, font=_font("serif_bold", 30), fill=(20, 20, 25), anchor="mm")
        sub = ("zawarta w dniu" if lang == "pl" else "concluded on") + f" {_date(rng)}" \
              + (" w " if lang == "pl" else " in ") + rng.choice(CITIES)
        d.text((w // 2, 112), sub, font=_font("serif", 16), fill=(60, 60, 65), anchor="mm")
        party_lbl = ("pomiędzy:", "a:") if lang == "pl" else ("between:", "and:")
        y = 170
        for lbl, party in zip(party_lbl, [rng.choice(COMPANIES), _person(rng)]):
            d.text((70, y), lbl, font=_font("serif", 15), fill=(50, 50, 55))
            d.text((100, y + 24), party, font=_font("serif_bold", 17), fill=(20, 20, 25))
            y += 62
        clauses = CLAUSES_PL if lang == "pl" else CLAUSES_EN
        picked = rng.sample(clauses, k=min(5, len(clauses)))
        for n, clause in enumerate(picked, 1):
            d.text((70, y), f"§ {n}", font=_font("serif_bold", 17), fill=(20, 20, 25))
            words, line, yy = clause.split(), "", y + 26
            for word in words:
                if d.textlength(line + word + " ", font=_font("serif", 15)) > w - 160:
                    d.text((90, yy), line, font=_font("serif", 15), fill=(35, 35, 40))
                    yy += 22
                    line = ""
                line += word + " "
            d.text((90, yy), line, font=_font("serif", 15), fill=(35, 35, 40))
            y = yy + 48
        sig_y = h - 120
        for x, who in [(120, "Zleceniodawca" if lang == "pl" else "Client"),
                       (w - 320, "Wykonawca" if lang == "pl" else "Contractor")]:
            d.line([x, sig_y, x + 200, sig_y], fill=(60, 60, 65), width=1)
            d.text((x + 100, sig_y + 16), who, font=_font("serif", 14), fill=(70, 70, 75), anchor="mm")
        _save(img, out / "contract-photo" / f"SYN-contract-{i + 1:04d}.jpg")


# ----------------------------------------------------------- bank statements

BANKS = ["Vistula Bank S.A.", "Northgate Bank", "Credit Polonia", "Meridian Savings"]
TXN_DESC = ["CARD PAYMENT GROCERY STORE", "TRANSFER RENT", "ATM WITHDRAWAL", "SALARY",
            "ONLINE PURCHASE ELECTRONICS", "UTILITY BILL ENERGY", "SUBSCRIPTION STREAMING",
            "RESTAURANT PAYMENT", "PHARMACY PURCHASE", "FUEL STATION", "INSURANCE PREMIUM"]


def gen_bank_statements(rng: random.Random, out: Path, count: int) -> None:
    for i in range(count):
        w, h = 850, 1100
        img = _paper(rng, w, h)
        d = ImageDraw.Draw(img)
        bank = rng.choice(BANKS)
        accent = rng.choice([(0, 75, 135), (140, 20, 30), (0, 100, 60), (70, 40, 120)])
        d.rectangle([0, 0, w, 90], fill=accent)
        d.text((40, 45), bank, font=_font("sans_bold", 28), fill=(255, 255, 255), anchor="lm")
        d.text((w - 40, 45), "ACCOUNT STATEMENT", font=_font("sans", 16), fill=(255, 255, 255), anchor="rm")
        holder = _person(rng)
        iban = f"PL{rng.randint(10, 99)} {rng.randint(1000, 9999)} " + " ".join(
            f"{rng.randint(0, 9999):04d}" for _ in range(5))
        month = f"{rng.randint(1, 12):02d}/{rng.randint(2022, 2025)}"
        y = 120
        for lbl, val in [("Account holder", holder), ("IBAN", iban), ("Statement period", month)]:
            d.text((40, y), f"{lbl}:", font=_font("sans", 15), fill=(90, 90, 95))
            d.text((220, y), val, font=_font("mono", 15), fill=(25, 25, 30))
            y += 28
        y += 20
        cols = [(40, "Date"), (160, "Description"), (560, "Amount"), (700, "Balance")]
        d.rectangle([30, y - 6, w - 30, y + 22], fill=(232, 234, 238))
        for x, name in cols:
            d.text((x, y), name, font=_font("sans_bold", 14), fill=(40, 40, 45))
        y += 34
        balance = rng.uniform(1500, 9000)
        for _ in range(rng.randint(12, 16)):
            amt = round(rng.uniform(-800, 900), 2) or 100.0
            balance += amt
            row = [(_date(rng)), rng.choice(TXN_DESC), f"{amt:+,.2f}", f"{balance:,.2f}"]
            for (x, _), val in zip(cols, row):
                color = (150, 25, 25) if val.startswith("-") else (25, 25, 30)
                d.text((x, y), val, font=_font("mono", 13), fill=color)
            d.line([30, y + 22, w - 30, y + 22], fill=(215, 215, 220), width=1)
            y += 30
        d.text((40, y + 24), f"Closing balance: {balance:,.2f} PLN",
               font=_font("sans_bold", 16), fill=(25, 25, 30))
        _save(img, out / "bank_statement-photo" / f"SYN-statement-{i + 1:04d}.jpg")


# ------------------------------------------------------------------- forms

FORM_KINDS = [
    ("PATIENT REGISTRATION FORM", ["Full name", "Date of birth", "PESEL / ID number",
                                   "Address", "Phone", "Insurance provider", "Allergies"]),
    ("JOB APPLICATION FORM", ["Full name", "Position applied for", "Email", "Phone",
                              "Highest education", "Years of experience", "Earliest start date"]),
    ("VISA APPLICATION FORM", ["Surname", "Given names", "Passport number", "Nationality",
                               "Purpose of travel", "Intended arrival date", "Duration of stay"]),
    ("VEHICLE REGISTRATION FORM", ["Owner name", "Vehicle make", "Model", "VIN",
                                   "Year of manufacture", "License plate", "Insurance policy no."]),
    ("GYM MEMBERSHIP FORM", ["Member name", "Date of birth", "Email", "Membership type",
                             "Start date", "Emergency contact", "Health conditions"]),
    ("LIBRARY CARD APPLICATION", ["Full name", "Address", "Email", "Phone",
                                  "Date of birth", "Preferred branch", "Card type"]),
]


def gen_forms(rng: random.Random, out: Path, count: int) -> None:
    for i in range(count):
        title, fields = FORM_KINDS[i % len(FORM_KINDS)]
        w, h = 820, 1080
        img = _paper(rng, w, h)
        d = ImageDraw.Draw(img)
        d.rectangle([40, 40, w - 40, 110], outline=(30, 30, 35), width=2)
        d.text((w // 2, 75), title, font=_font("sans_bold", 26), fill=(25, 25, 30), anchor="mm")
        d.text((w - 50, 130), f"Form no. {rng.randint(100, 999)}/{rng.randint(2022, 2025)}",
               font=_font("mono", 13), fill=(90, 90, 95), anchor="rm")
        y = 180
        values = {
            "name": _person(rng), "date": _date(rng), "num": str(rng.randint(10 ** 8, 10 ** 9 - 1)),
        }
        for field in fields:
            d.text((60, y), field.upper(), font=_font("sans", 13), fill=(90, 90, 95))
            d.rectangle([60, y + 22, w - 60, y + 58], outline=(120, 120, 125), width=1)
            fl = field.lower()
            if "name" in fl:
                val = values["name"]
            elif "date" in fl:
                val = _date(rng)
            elif any(k in fl for k in ("number", "pesel", "vin", "phone", "plate", "policy")):
                val = values["num"]
            elif "email" in fl:
                val = values["name"].lower().replace(" ", ".") + "@example.com"
            else:
                val = rng.choice(["Standard", "N/A", rng.choice(CITIES), "Full-time", "Tourism"])
            d.text((75, y + 40), val, font=_font("oblique", 17), fill=(25, 30, 90), anchor="lm")
            y += 86
        d.text((60, y + 10), "Signature:", font=_font("sans", 13), fill=(90, 90, 95))
        d.line([160, y + 40, 400, y + 40], fill=(120, 120, 125), width=1)
        _save(img, out / "form-photo" / f"SYN-form-{i + 1:04d}.jpg")


# ----------------------------------------------------------- business cards

TITLES = ["CEO", "Software Engineer", "Sales Director", "Marketing Manager",
          "Legal Counsel", "Product Designer", "Data Analyst", "Head of Operations"]


def gen_business_cards(rng: random.Random, out: Path, count: int) -> None:
    for i in range(count):
        w, h = 1000, 570
        accent = rng.choice([(20, 60, 110), (150, 35, 45), (25, 105, 75),
                             (85, 45, 130), (200, 120, 20), (40, 40, 45)])
        img = Image.new("RGB", (w, h), (250, 250, 248) if i % 3 else accent)
        dark_bg = i % 3 == 0
        fg = (255, 255, 255) if dark_bg else (30, 30, 35)
        sub = (220, 220, 225) if dark_bg else (95, 95, 100)
        d = ImageDraw.Draw(img)
        if not dark_bg:
            if i % 2:
                d.rectangle([0, 0, w, 90], fill=accent)
            else:
                d.rectangle([0, h - 70, w, h], fill=accent)
        person, company = _person(rng), rng.choice(COMPANIES)
        d.text((70, 200), person, font=_font("sans_bold", 52), fill=fg)
        d.text((70, 270), rng.choice(TITLES), font=_font("sans", 30), fill=sub)
        d.text((70, 360), company, font=_font("serif_bold", 26), fill=fg)
        email = person.lower().replace(" ", ".").replace("ś", "s").replace("ń", "n") + "@" + \
            company.split()[0].lower().replace(".", "") + ".com"
        d.text((70, 430), f"+48 {rng.randint(500, 899)} {rng.randint(100, 999)} {rng.randint(100, 999)}",
               font=_font("mono", 22), fill=sub)
        d.text((70, 465), email, font=_font("mono", 22), fill=sub)
        _save(img, out / "business_card-photo" / f"SYN-card-{i + 1:04d}.jpg")


# ------------------------------------------------------------ presentations

SLIDES = [
    ("Q3 Financial Results", ["Revenue up 12% year over year", "Operating margin at 18.4%",
                              "Cloud segment fastest growing", "Guidance raised for Q4"]),
    ("Product Roadmap 2026", ["Mobile app redesign ships in March", "API v3 public beta",
                              "Self-serve onboarding funnel", "Enterprise SSO and audit logs"]),
    ("Machine Learning Architecture", ["Feature store consolidates pipelines",
                                       "Batch and streaming inference paths",
                                       "Model registry with staged rollouts", "Monitoring via drift detection"]),
    ("Marketing Strategy", ["Target mid-market segment", "Double content production",
                            "Partner co-marketing program", "Attribution model refresh"]),
    ("Team Onboarding", ["Week 1: environment setup", "Week 2: first supervised ticket",
                         "Mentor assigned to every hire", "30/60/90 day check-ins"]),
    ("Security Review", ["Zero critical findings", "Patch SLA reduced to 48h",
                         "MFA enforced org-wide", "Next pentest scheduled Q1"]),
]


def gen_presentations(rng: random.Random, out: Path, count: int) -> None:
    for i in range(count):
        title, bullets = SLIDES[i % len(SLIDES)]
        w, h = 1280, 720
        theme = rng.choice([(18, 60, 105), (120, 30, 40), (30, 95, 70), (60, 45, 110)])
        light = i % 2 == 0
        img = Image.new("RGB", (w, h), (250, 250, 252) if light else theme)
        d = ImageDraw.Draw(img)
        fg = (25, 25, 30) if light else (255, 255, 255)
        if light:
            d.rectangle([0, 0, w, 12], fill=theme)
            d.rectangle([0, h - 46, w, h], fill=(238, 238, 242))
        d.text((80, 90), title, font=_font("sans_bold", 54), fill=fg)
        d.line([80, 170, 80 + int(d.textlength(title, font=_font("sans_bold", 54))), 170],
               fill=theme if light else (255, 255, 255), width=4)
        y = 240
        for b in bullets:
            d.ellipse([90, y + 12, 106, y + 28], fill=theme if light else (255, 255, 255))
            d.text((130, y), b, font=_font("sans", 32), fill=fg)
            y += 78
        d.text((w - 60, h - 24), f"{i + 1} / {count}", font=_font("sans", 18),
               fill=(120, 120, 125) if light else (230, 230, 235), anchor="rm")
        _save(img, out / "presentation-photo" / f"SYN-slide-{i + 1:04d}.jpg")


# ------------------------------------------------------------- spreadsheets

SHEETS = [
    ("Sales by Region", ["Region", "Q1", "Q2", "Q3", "Q4", "Total"],
     ["North", "South", "East", "West", "Central", "Export"]),
    ("Inventory Levels", ["SKU", "Product", "In stock", "Reserved", "Reorder at", "Status"],
     None),
    ("Project Budget", ["Item", "Planned", "Actual", "Variance", "Owner", "Notes"],
     ["Design", "Development", "QA", "Marketing", "Infrastructure", "Contingency"]),
    ("Employee Hours", ["Employee", "Mon", "Tue", "Wed", "Thu", "Fri"], None),
]


def gen_spreadsheets(rng: random.Random, out: Path, count: int) -> None:
    for i in range(count):
        title, headers, row_names = SHEETS[i % len(SHEETS)]
        w, h = 1100, 700
        img = Image.new("RGB", (w, h), (252, 252, 253))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, w, 54], fill=(33, 115, 70))
        d.text((24, 27), f"{title}.xlsx", font=_font("sans_bold", 22), fill=(255, 255, 255), anchor="lm")
        rows, cols = 14, len(headers)
        x0, y0, cw, rh = 60, 90, (w - 100) // cols, 38
        # column letters + row numbers, like a real sheet
        for c in range(cols):
            d.text((x0 + c * cw + cw // 2, y0 - 16), chr(65 + c),
                   font=_font("sans", 14), fill=(120, 120, 125), anchor="mm")
        for r in range(rows):
            d.text((x0 - 24, y0 + r * rh + rh // 2), str(r + 1),
                   font=_font("sans", 14), fill=(120, 120, 125), anchor="mm")
        d.rectangle([x0, y0, x0 + cols * cw, y0 + rh], fill=(226, 235, 245))
        for r in range(rows + 1):
            d.line([x0, y0 + r * rh, x0 + cols * cw, y0 + r * rh], fill=(200, 202, 208))
        for c in range(cols + 1):
            d.line([x0 + c * cw, y0, x0 + c * cw, y0 + rows * rh], fill=(200, 202, 208))
        for c, hname in enumerate(headers):
            d.text((x0 + c * cw + 10, y0 + rh // 2), hname, font=_font("sans_bold", 15),
                   fill=(30, 30, 35), anchor="lm")
        for r in range(1, rows):
            for c in range(cols):
                if c == 0:
                    val = (row_names[(r - 1) % len(row_names)] if row_names
                           else f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)[0]}.")
                else:
                    val = f"{rng.randint(10, 9800):,}"
                d.text((x0 + c * cw + 10, y0 + r * rh + rh // 2), val,
                       font=_font("sans", 14), fill=(45, 45, 50), anchor="lm")
        _save(img, out / "spreadsheet-photo" / f"SYN-sheet-{i + 1:04d}.jpg")


# -------------------------------------------------------------- screenshots

APPS = [
    ("Inbox — MailPro", ["Weekly report ready for review", "Invoice #4482 from Delta Logistics",
                         "Your subscription renews soon", "Meeting notes: product sync",
                         "Security alert: new sign-in", "Lunch on Thursday?"]),
    ("Dashboard — MetricsHub", None),
    ("Tasks — FlowBoard", ["Fix login redirect bug", "Write Q4 planning doc", "Review PR #231",
                           "Update onboarding email copy", "Prepare demo environment",
                           "Ship dark mode toggle"]),
    ("Chat — TeamTalk", ["Anna: the deploy is out ✔", "Marek: metrics look stable",
                         "Sarah: can someone review my PR?", "John: on it",
                         "Ewa: standup moved to 10:30", "Bot: build #482 passed"]),
]


def gen_screenshots(rng: random.Random, out: Path, count: int) -> None:
    for i in range(count):
        title, items = APPS[i % len(APPS)]
        w, h = 1280, 800
        img = Image.new("RGB", (w, h), (243, 244, 247))
        d = ImageDraw.Draw(img)
        # window chrome
        d.rectangle([0, 0, w, 44], fill=(228, 229, 233))
        for n, color in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
            d.ellipse([16 + n * 26, 15, 30 + n * 26, 29], fill=color)
        d.text((w // 2, 22), title, font=_font("sans", 16), fill=(70, 70, 75), anchor="mm")
        # sidebar
        d.rectangle([0, 44, 230, h], fill=(37, 42, 55))
        for n, item in enumerate(["Home", "Inbox", "Projects", "Reports", "Settings"]):
            if n == i % 5:
                d.rectangle([0, 74 + n * 46, 230, 116 + n * 46], fill=(58, 66, 86))
            d.text((28, 95 + n * 46), item, font=_font("sans", 17), fill=(215, 218, 226), anchor="lm")
        if items:
            y = 80
            for item in items:
                d.rounded_rectangle([260, y, w - 40, y + 92], radius=10, fill=(255, 255, 255),
                                    outline=(222, 224, 230))
                d.text((290, y + 30), item, font=_font("sans_bold", 19), fill=(35, 35, 40), anchor="lm")
                d.text((290, y + 62), f"{rng.randint(1, 59)} min ago · {rng.choice(CITIES)}",
                       font=_font("sans", 14), fill=(130, 132, 140), anchor="lm")
                y += 108
        else:  # dashboard cards + bar chart
            for n in range(3):
                x = 260 + n * 330
                d.rounded_rectangle([x, 80, x + 300, 200], radius=10, fill=(255, 255, 255),
                                    outline=(222, 224, 230))
                d.text((x + 24, 116), ["Active users", "Revenue", "Error rate"][n],
                       font=_font("sans", 15), fill=(130, 132, 140), anchor="lm")
                d.text((x + 24, 160), [f"{rng.randint(10, 90)}k", f"${rng.randint(100, 900)}k",
                                       f"{rng.uniform(0.1, 2):.2f}%"][n],
                       font=_font("sans_bold", 32), fill=(35, 35, 40), anchor="lm")
            d.rounded_rectangle([260, 230, w - 40, h - 60], radius=10, fill=(255, 255, 255),
                                outline=(222, 224, 230))
            for n in range(12):
                bh = rng.randint(40, 380)
                x = 310 + n * 76
                d.rectangle([x, h - 110 - bh, x + 44, h - 110], fill=(86, 132, 226))
        _save(img, out / "screenshot-photo" / f"SYN-screen-{i + 1:04d}.jpg")


# -------------------------------------------------------------------- notes

NOTES = [
    ["Shopping:", "- milk, eggs, bread", "- coffee beans", "- olive oil", "- something for dinner"],
    ["Meeting w/ Marek 14:00", "- budget draft due Friday", "- ask about Q3 numbers",
     "- book room for workshop"],
    ["Ideas:", "- automate weekly report", "- refactor import script", "- blog post about caching?"],
    ["Pamiętać!", "- odebrać paczkę do 18:00", "- przelew za mieszkanie", "- zadzwonić do serwisu",
     "- kupić bilety na piątek"],
    ["Call plumber tomorrow!", "leak under kitchen sink", "warranty until Nov?",
     "receipt in the drawer"],
    ["Recipe tweaks:", "- less salt next time", "- bake 25 min not 30", "- double the sauce"],
]


def gen_notes(rng: random.Random, out: Path, count: int) -> None:
    for i in range(count):
        lines = NOTES[i % len(NOTES)]
        w, h = 760, 950
        img = Image.new("RGB", (w, h), (252, 250, 240))
        d = ImageDraw.Draw(img)
        for y in range(120, h - 40, 56):  # ruled paper
            d.line([40, y, w - 40, y], fill=(190, 205, 225), width=1)
        d.line([90, 60, 90, h - 40], fill=(230, 160, 160), width=2)
        layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        ld = ImageDraw.Draw(layer)
        ink = rng.choice([(25, 35, 110, 255), (35, 35, 40, 255), (20, 80, 45, 255)])
        y = 130
        for line in lines:
            ld.text((110 + rng.randint(-6, 10), y), line, font=_font("oblique", 30), fill=ink)
            y += 56 + rng.randint(-4, 6)
        layer = layer.rotate(rng.uniform(-2.0, 2.0), resample=Image.BICUBIC)
        img.paste(layer, (0, 0), layer)
        _save(img, out / "note-photo" / f"SYN-note-{i + 1:04d}.jpg")


# ------------------------------------------------------------------ letters

LETTERS_EN = [
    ("complaint", "Dear Sir or Madam,\n\nI am writing to complain about the dishwasher (model "
     "WX-240) purchased from your store on {date}. After only three weeks the appliance stopped "
     "draining and floods the kitchen floor on every cycle.\n\nI request a repair under warranty "
     "or a full refund within 14 days, as provided by consumer protection law. Copies of the "
     "receipt and the warranty card are enclosed.\n\nYours faithfully,\n{name}"),
    ("cover letter", "Dear Hiring Manager,\n\nI am applying for the Data Analyst position "
     "advertised on your careers page. In my current role at {company} I build reporting "
     "pipelines in Python and SQL and present monthly findings to management.\n\nI would welcome "
     "the opportunity to discuss how my experience fits your team.\n\nKind regards,\n{name}"),
    ("resignation", "Dear {name2},\n\nPlease accept this letter as formal notice of my "
     "resignation from the position of Office Manager at {company}, effective {date}. I am "
     "grateful for the opportunities for growth during my time here and will do everything to "
     "ensure a smooth handover.\n\nSincerely,\n{name}"),
    ("thank you", "Dear {name2},\n\nThank you for hosting our delegation last week. The tour of "
     "the production facility was fascinating and the discussions about a joint venture were "
     "very promising. We will send a draft memorandum by {date}.\n\nWith best regards,\n{name}"),
    ("reference", "To Whom It May Concern,\n\nI confirm that {name2} was employed at {company} "
     "as a Logistics Coordinator for three years. Their duties included route planning, customs "
     "documentation and carrier negotiations, all performed with great reliability.\n\nI can "
     "recommend them without hesitation.\n\n{name}\nOperations Director"),
    ("invitation", "Dear {name2},\n\nOn behalf of {company} I am pleased to invite you to our "
     "annual partner conference held in {city} on {date}. The agenda covers market outlook, new "
     "product lines and a networking dinner.\n\nPlease confirm attendance by e-mail.\n\nYours "
     "sincerely,\n{name}"),
    ("cancellation", "Dear Customer Service,\n\nI hereby cancel my subscription (customer number "
     "{num}) effective at the end of the current billing period. Please confirm the cancellation "
     "in writing and stop all further charges to my card.\n\nRegards,\n{name}"),
    ("inquiry", "Dear Sales Team,\n\nWe are refurbishing our office in {city} and are interested "
     "in your modular desk systems. Could you send a catalogue, bulk pricing for around 40 "
     "workstations, and typical lead times?\n\nThank you in advance,\n{name}\n{company}"),
]

LETTERS_PL = [
    ("reklamacja", "Szanowni Państwo,\n\nskładam reklamację dotyczącą ekspresu do kawy zakupionego "
     "w Państwa sklepie dnia {date}. Urządzenie po dwóch tygodniach przestało podgrzewać wodę i "
     "nie reaguje na przyciski.\n\nProszę o naprawę gwarancyjną lub zwrot pełnej kwoty w terminie "
     "14 dni. W załączeniu przesyłam kopię paragonu.\n\nZ poważaniem,\n{name}"),
    ("wypowiedzenie", "Szanowny Panie,\n\nniniejszym wypowiadam umowę najmu lokalu przy ulicy "
     "Kwiatowej 12 w mieście {city}, zawartą dnia {date}, z zachowaniem trzymiesięcznego okresu "
     "wypowiedzenia. Proszę o potwierdzenie otrzymania niniejszego pisma.\n\nZ poważaniem,\n{name}"),
    ("podanie", "Szanowna Pani Dyrektor,\n\nzwracam się z uprzejmą prośbą o wydanie duplikatu "
     "świadectwa ukończenia kursu, ponieważ oryginał uległ zniszczeniu. Numer zaświadczenia: "
     "{num}.\n\nZ góry dziękuję za pozytywne rozpatrzenie prośby.\n\nZ poważaniem,\n{name}"),
    ("zaproszenie", "Szanowni Państwo,\n\nw imieniu firmy {company} serdecznie zapraszam na "
     "uroczyste otwarcie nowego oddziału w mieście {city}, które odbędzie się dnia {date} o "
     "godzinie 17:00. Prosimy o potwierdzenie obecności.\n\nZ wyrazami szacunku,\n{name}"),
]


def gen_letters(rng: random.Random, out: Path, count: int) -> None:
    pool = LETTERS_EN + LETTERS_PL
    for i in range(count):
        kind, body = pool[i % len(pool)]
        text = body.format(
            date=_date(rng), name=_person(rng), name2=_person(rng),
            company=rng.choice(COMPANIES), city=rng.choice(CITIES),
            num=rng.randint(10 ** 5, 10 ** 6 - 1),
        )
        lang = "pl" if (kind, body) in LETTERS_PL else "en"
        path = out / "letter-txt" / f"SYN-letter-{lang}-{i + 1:04d}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {path}")


# ----------------------------------------------------- top-up: medical

LAB_TESTS = [
    ("Hemoglobin", "g/dL", 12.0, 17.5), ("WBC", "10^3/uL", 4.0, 11.0),
    ("Platelets", "10^3/uL", 150, 400), ("Glucose (fasting)", "mg/dL", 70, 99),
    ("Total cholesterol", "mg/dL", 120, 200), ("HDL", "mg/dL", 40, 90),
    ("LDL", "mg/dL", 50, 130), ("TSH", "mIU/L", 0.4, 4.0),
    ("Creatinine", "mg/dL", 0.6, 1.3), ("ALT", "U/L", 7, 56),
]


def gen_medical(rng: random.Random, out: Path, count: int) -> None:
    for i in range(count):
        w, h = 820, 1080
        img = _paper(rng, w, h)
        d = ImageDraw.Draw(img)
        lab = rng.choice(["Central Diagnostic Laboratory", "MedLab Analityka",
                          "City Hospital Laboratory", "BioTest Diagnostyka"])
        d.text((w // 2, 60), lab, font=_font("sans_bold", 26), fill=(20, 60, 110), anchor="mm")
        d.text((w // 2, 95), "LABORATORY TEST REPORT", font=_font("sans", 16),
               fill=(70, 70, 75), anchor="mm")
        d.line([50, 120, w - 50, 120], fill=(20, 60, 110), width=2)
        y = 150
        for lbl, val in [("Patient", _person(rng)), ("Date of birth", _date(rng)),
                         ("Sample collected", _date(rng)),
                         ("Order no.", str(rng.randint(10 ** 6, 10 ** 7 - 1)))]:
            d.text((60, y), f"{lbl}:", font=_font("sans", 15), fill=(90, 90, 95))
            d.text((240, y), val, font=_font("sans_bold", 15), fill=(25, 25, 30))
            y += 28
        y += 24
        cols = [(60, "Test"), (350, "Result"), (490, "Units"), (610, "Reference"), (740, "Flag")]
        d.rectangle([50, y - 6, w - 50, y + 24], fill=(226, 235, 245))
        for x, name in cols:
            d.text((x, y), name, font=_font("sans_bold", 14), fill=(30, 30, 35))
        y += 40
        for test, unit, lo, hi in rng.sample(LAB_TESTS, k=8):
            val = round(rng.uniform(lo * 0.8, hi * 1.2), 1)
            flag = "H" if val > hi else ("L" if val < lo else "")
            for (x, _), txt in zip(cols, [test, str(val), unit, f"{lo} - {hi}", flag]):
                color = (170, 30, 30) if flag and x == 740 else (35, 35, 40)
                d.text((x, y), txt, font=_font("mono", 14), fill=color)
            d.line([50, y + 24, w - 50, y + 24], fill=(215, 215, 220))
            y += 34
        d.text((60, y + 30), "Verified by: dr " + _person(rng), font=_font("oblique", 16),
               fill=(50, 50, 55))
        _save(img, out / "medical-photo" / f"SYN-labreport-{i + 1:04d}.jpg")


# ---------------------------------------------------- top-up: warranty

PRODUCTS = ["Washing machine WX-240", "Espresso machine Bar-9", "Laptop ProBook 15",
            "Cordless drill PD-18V", "Refrigerator KGN-39", "Smart TV 55Q80",
            "Robot vacuum RV-500", "Air conditioner AC-12"]


def gen_warranty(rng: random.Random, out: Path, count: int) -> None:
    for i in range(count):
        pl = i % 2 == 0
        w, h = 850, 1050
        img = _paper(rng, w, h, tint=(255, 253, 246))
        d = ImageDraw.Draw(img)
        d.rectangle([30, 30, w - 30, h - 30], outline=(140, 110, 30), width=3)
        title = "KARTA GWARANCYJNA" if pl else "WARRANTY CERTIFICATE"
        d.text((w // 2, 90), title, font=_font("serif_bold", 34), fill=(110, 85, 20), anchor="mm")
        rows = [
            ("Produkt" if pl else "Product", rng.choice(PRODUCTS)),
            ("Numer seryjny" if pl else "Serial number",
             f"SN-{rng.randint(10 ** 7, 10 ** 8 - 1)}"),
            ("Data zakupu" if pl else "Purchase date", _date(rng)),
            ("Okres gwarancji" if pl else "Warranty period",
             rng.choice(["24 miesiące", "36 miesięcy"] if pl else ["24 months", "36 months"])),
            ("Sprzedawca" if pl else "Seller", rng.choice(COMPANIES)),
        ]
        y = 170
        for lbl, val in rows:
            d.text((80, y), lbl + ":", font=_font("serif", 18), fill=(80, 70, 40))
            d.line([300, y + 22, w - 80, y + 22], fill=(190, 175, 130))
            d.text((310, y), str(val), font=_font("serif_bold", 18), fill=(30, 30, 35))
            y += 58
        terms_pl = ["Gwarancja obejmuje wady fabryczne ujawnione w okresie gwarancyjnym.",
                    "Naprawa gwarancyjna zostanie wykonana w terminie 14 dni roboczych.",
                    "Gwarancja nie obejmuje uszkodzeń mechanicznych z winy użytkownika.",
                    "Podstawą roszczeń jest karta gwarancyjna wraz z dowodem zakupu."]
        terms_en = ["This warranty covers manufacturing defects revealed during the warranty period.",
                    "Warranty repairs will be completed within 14 working days.",
                    "Mechanical damage caused by the user is not covered.",
                    "Claims require this certificate together with the proof of purchase."]
        d.text((80, y + 10), "Warunki gwarancji:" if pl else "Warranty terms:",
               font=_font("serif_bold", 18), fill=(80, 70, 40))
        y += 48
        for n, term in enumerate(terms_pl if pl else terms_en, 1):
            d.text((90, y), f"{n}. {term}", font=_font("serif", 14), fill=(45, 45, 50))
            y += 34
        d.ellipse([w - 260, h - 240, w - 110, h - 90], outline=(140, 110, 30), width=2)
        d.text((w - 185, h - 165), "OK", font=_font("serif_bold", 26), fill=(140, 110, 30),
               anchor="mm")
        d.text((110, h - 130), ("pieczęć i podpis sprzedawcy" if pl else "seller stamp and signature"),
               font=_font("oblique", 14), fill=(90, 90, 95))
        _save(img, out / "warranty-photo" / f"SYN-warranty-{i + 1:04d}.jpg")


# ----------------------------------------------------- top-up: diplomas

ACHIEVEMENTS = ["Certificate of Achievement", "Certificate of Completion",
                "Dyplom Uznania", "Certificate of Excellence", "Dyplom za Zajęcie I Miejsca"]
REASONS = ["for outstanding results in the regional mathematics olympiad",
           "for completing the Advanced Project Management course",
           "za wybitne osiągnięcia w konkursie plastycznym",
           "for exceptional contribution to the annual science fair",
           "za zajęcie pierwszego miejsca w turnieju szachowym"]


def gen_diplomas(rng: random.Random, out: Path, count: int) -> None:
    for i in range(count):
        w, h = 1100, 800
        img = _paper(rng, w, h, tint=(253, 250, 240))
        d = ImageDraw.Draw(img)
        d.rectangle([28, 28, w - 28, h - 28], outline=(150, 120, 40), width=5)
        d.rectangle([44, 44, w - 44, h - 44], outline=(150, 120, 40), width=1)
        d.text((w // 2, 150), ACHIEVEMENTS[i % len(ACHIEVEMENTS)],
               font=_font("serif_bold", 52), fill=(110, 85, 20), anchor="mm")
        d.text((w // 2, 250), "awarded to" if i % 2 else "przyznany dla",
               font=_font("oblique", 22), fill=(90, 90, 95), anchor="mm")
        d.text((w // 2, 330), _person(rng), font=_font("serif_bold", 44), fill=(30, 30, 35),
               anchor="mm")
        d.line([w // 2 - 260, 370, w // 2 + 260, 370], fill=(150, 120, 40), width=2)
        d.text((w // 2, 430), REASONS[i % len(REASONS)], font=_font("serif", 20),
               fill=(50, 50, 55), anchor="mm")
        d.text((w // 2, 500), f"{rng.choice(CITIES)}, {_date(rng)}", font=_font("serif", 18),
               fill=(80, 80, 85), anchor="mm")
        for x, who in [(220, "Organizator"), (w - 220, "Dyrektor")]:
            d.line([x - 110, h - 150, x + 110, h - 150], fill=(60, 60, 65))
            d.text((x, h - 125), who, font=_font("serif", 16), fill=(80, 80, 85), anchor="mm")
        _save(img, out / "diploma-photo" / f"SYN-diploma-{i + 1:04d}.jpg")


# ----------------------------------------------------- top-up: syllabi (PDF)

COURSES = [
    ("Introduction to Statistics", "en"), ("Databases and SQL", "en"),
    ("Organic Chemistry II", "en"), ("Historia Filozofii", "pl"),
    ("Analiza Matematyczna I", "pl"), ("Software Engineering", "en"),
    ("Makroekonomia", "pl"),
]
WEEK_TOPICS = ["Course overview and requirements", "Foundations and key definitions",
               "Core methods, part one", "Core methods, part two", "Case studies",
               "Midterm review and exam", "Advanced topics", "Applications in practice",
               "Group project work", "Guest lecture", "Revision", "Final exam"]


def gen_syllabi(rng: random.Random, out: Path, count: int) -> None:
    from fpdf import FPDF

    for i in range(count):
        course, lang = COURSES[i % len(COURSES)]
        pdf = FPDF()
        pdf.add_font("dejavu", "", str(FONTS["sans"]))
        pdf.add_font("dejavu", "B", str(FONTS["sans_bold"]))
        pdf.add_page()
        pdf.set_font("dejavu", "B", 20)
        pdf.cell(0, 12, ("Sylabus przedmiotu: " if lang == "pl" else "Course Syllabus: ") + course,
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("dejavu", "", 11)
        ects = rng.randint(2, 6)
        head = [
            (("Prowadzący" if lang == "pl" else "Instructor"), f"dr {_person(rng)}"),
            (("Semestr" if lang == "pl" else "Semester"),
             rng.choice(["zimowy 2025/26", "letni 2025/26"] if lang == "pl"
                        else ["Fall 2025", "Spring 2026"])),
            ("ECTS", str(ects)),
            (("Forma zaliczenia" if lang == "pl" else "Assessment"),
             "egzamin pisemny" if lang == "pl" else "written exam"),
        ]
        for lbl, val in head:
            pdf.cell(0, 8, f"{lbl}: {val}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        pdf.set_font("dejavu", "B", 13)
        pdf.cell(0, 10, "Program zajęć" if lang == "pl" else "Weekly schedule",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("dejavu", "", 11)
        for week, topic in enumerate(WEEK_TOPICS, 1):
            label = "Tydzień" if lang == "pl" else "Week"
            pdf.cell(0, 7, f"{label} {week}: {topic}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(4)
        pdf.set_font("dejavu", "B", 13)
        pdf.cell(0, 10, "Zasady oceniania" if lang == "pl" else "Grading",
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("dejavu", "", 11)
        exam = rng.choice([50, 60, 70])
        parts = [("egzamin" if lang == "pl" else "final exam", exam),
                 ("projekt" if lang == "pl" else "project", (100 - exam) // 2),
                 ("aktywność" if lang == "pl" else "participation", 100 - exam - (100 - exam) // 2)]
        for name, pct in parts:
            pdf.cell(0, 7, f"- {name}: {pct}%", new_x="LMARGIN", new_y="NEXT")
        path = out / "syllabus-pdf" / f"SYN-syllabus-{i + 1:04d}.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        pdf.output(str(path))
        print(f"wrote {path}")


GENERATORS = {
    "contract-photo": (gen_contracts, 12),
    "bank_statement-photo": (gen_bank_statements, 12),
    "form-photo": (gen_forms, 12),
    "business_card-photo": (gen_business_cards, 12),
    "presentation-photo": (gen_presentations, 12),
    "spreadsheet-photo": (gen_spreadsheets, 12),
    "screenshot-photo": (gen_screenshots, 12),
    "note-photo": (gen_notes, 12),
    "letter-txt": (gen_letters, 12),
    "medical-photo": (gen_medical, 8),
    "warranty-photo": (gen_warranty, 8),
    "diploma-photo": (gen_diplomas, 5),
    "syllabus-pdf": (gen_syllabi, 7),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="data", help="dataset root directory")
    parser.add_argument("--only", nargs="*", help="restrict to specific category folders")
    args = parser.parse_args()

    out = Path(args.data_dir)
    for folder, (fn, count) in GENERATORS.items():
        if args.only and folder not in args.only:
            continue
        # dedicated RNG per category: adding categories never reshuffles others
        fn(random.Random(f"{SEED}:{folder}"), out, count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
