"""Tez / makale / ödev için Türkçe bulgu raporu: Yöntem ve Bulgular metni + tez formatında tablolar.

>>> model = ek.ekk("np.log(ucret) ~ egitim + cinsiyet", veri, kovaryans="HC1")
>>> model.rapor("bulgular.docx")          # Word dosyası (ayrıca .md ve .html)
>>> print(model.rapor())                  # metin olarak

Rapor, modelin hesapladığı sayıları cümleye döker. Metni okuyup kendi çalışmana göre düzeltmek gerekir.
"""

from __future__ import annotations

import html as _html
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ._yardimci import DIL, l, sayi, tablo_sayi, yildiz, yuzde

_ALT_SIMGE = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")

KAYNAKLAR = {
    "mcfadden": "McFadden, D. (1974). Conditional logit analysis of qualitative choice behavior. P. Zarembka (Ed.), Frontiers in econometrics içinde (s. 105–142). Academic Press.",
    "df": "Dickey, D. A., & Fuller, W. A. (1979). Distribution of the estimators for autoregressive time series with a unit root. Journal of the American Statistical Association, 74(366), 427–431.",
    "kpss": "Kwiatkowski, D., Phillips, P. C. B., Schmidt, P., & Shin, Y. (1992). Testing the null hypothesis of stationarity against the alternative of a unit root. Journal of Econometrics, 54(1–3), 159–178.",
    "mackinnon96": "MacKinnon, J. G. (1996). Numerical distribution functions for unit root and cointegration tests. Journal of Applied Econometrics, 11(6), 601–618.",
    "mackinnon2010": "MacKinnon, J. G. (2010). Critical values for cointegration tests (Queen's Economics Department Working Paper No. 1227). Queen's University.",
    "eg": "Engle, R. F., & Granger, C. W. J. (1987). Co-integration and error correction: Representation, estimation, and testing. Econometrica, 55(2), 251–276.",
    "sims": "Sims, C. A. (1980). Macroeconomics and reality. Econometrica, 48(1), 1–48.",
    "lutkepohl": "Lütkepohl, H. (2005). New introduction to multiple time series analysis. Springer.",
    "llc": "Levin, A., Lin, C.-F., & Chu, C.-S. J. (2002). Unit root tests in panel data: Asymptotic and finite-sample properties. Journal of Econometrics, 108(1), 1–24.",
    "ips": "Im, K. S., Pesaran, M. H., & Shin, Y. (2003). Testing for unit roots in heterogeneous panels. Journal of Econometrics, 115(1), 53–74.",
    "maddala_wu": "Maddala, G. S., & Wu, S. (1999). A comparative study of unit root tests with panel data and a new simple test. Oxford Bulletin of Economics and Statistics, 61(S1), 631–652.",
    "choi": "Choi, I. (2001). Unit root tests for panel data. Journal of International Money and Finance, 20(2), 249–272.",
    "pesaran_cd": "Pesaran, M. H. (2004). General diagnostic tests for cross section dependence in panels (CESifo Working Paper No. 1229). CESifo.",
    "pesaran2015": "Pesaran, M. H. (2015). Testing weak cross-sectional dependence in large panels. Econometric Reviews, 34(6–10), 1089–1117.",
    "pesaran2007": "Pesaran, M. H. (2007). A simple panel unit root test in the presence of cross-section dependence. Journal of Applied Econometrics, 22(2), 265–312.",
    "bfk2012": "Baltagi, B. H., Feng, Q., & Kao, C. (2012). A Lagrange multiplier test for cross-sectional dependence in a fixed effects panel data model. Journal of Econometrics, 170(1), 164–177.",
    "zivot_andrews": "Zivot, E., & Andrews, D. W. K. (1992). Further evidence on the great crash, the oil-price shock, and the unit-root hypothesis. Journal of Business & Economic Statistics, 10(3), 251–270.",
    "toda_yamamoto": "Toda, H. Y., & Yamamoto, T. (1995). Statistical inference in vector autoregressions with possibly integrated processes. Journal of Econometrics, 66(1–2), 225–250.",
    "granger": "Granger, C. W. J. (1969). Investigating causal relations by econometric models and cross-spectral methods. Econometrica, 37(3), 424–438.",
    "pesaran": "Pesaran, M. H., Shin, Y., & Smith, R. J. (2001). Bounds testing approaches to the analysis of level relationships. Journal of Applied Econometrics, 16(3), 289–326.",
    "baltagi": "Baltagi, B. H. (2021). Econometric analysis of panel data (6. baskı). Springer.",
    "bp80": "Breusch, T. S., & Pagan, A. R. (1980). The Lagrange multiplier test and its applications to model specification in econometrics. The Review of Economic Studies, 47(1), 239–253.",
    "hausman": "Hausman, J. A. (1978). Specification tests in econometrics. Econometrica, 46(6), 1251–1271.",
    "swamy": "Swamy, P. A. V. B., & Arora, S. S. (1972). The exact finite sample properties of the estimators of coefficients in the error components regression models. Econometrica, 40(2), 261–275.",
    "staiger": "Staiger, D., & Stock, J. H. (1997). Instrumental variables regression with weak instruments. Econometrica, 65(3), 557–586.",
    "wu": "Wu, D.-M. (1973). Alternative tests of independence between stochastic regressors and disturbances. Econometrica, 41(4), 733–750.",
    "sargan": "Sargan, J. D. (1958). The estimation of economic relationships using instrumental variables. Econometrica, 26(3), 393–415.",
    "cameron": "Cameron, A. C., & Miller, D. L. (2015). A practitioner's guide to cluster-robust inference. Journal of Human Resources, 50(2), 317–372.",
    "prais": "Prais, S. J., & Winsten, C. B. (1954). Trend estimators and serial correlation (Cowles Commission Discussion Paper No. 383). Cowles Commission.",
    "corc": "Cochrane, D., & Orcutt, G. H. (1949). Application of least squares regression to relationships containing auto-correlated error terms. Journal of the American Statistical Association, 44(245), 32–61.",
    "mccullagh": "McCullagh, P. (1980). Regression models for ordinal data. Journal of the Royal Statistical Society: Series B, 42(2), 109–142.",
    "cameron_trivedi": "Cameron, A. C., & Trivedi, P. K. (2013). Regression analysis of count data (2. baskı). Cambridge University Press.",
    "tobin": "Tobin, J. (1958). Estimation of relationships for limited dependent variables. Econometrica, 26(1), 24–36.",
    "wooldridge": "Wooldridge, J. M. (2020). Introductory econometrics: A modern approach (7. baskı). Cengage Learning.",
    "white": "White, H. (1980). A heteroskedasticity-consistent covariance matrix estimator and a direct test for heteroskedasticity. Econometrica, 48(4), 817–838.",
    "mackinnon": "MacKinnon, J. G., & White, H. (1985). Some heteroskedasticity-consistent covariance matrix estimators with improved finite sample properties. Journal of Econometrics, 29(3), 305–325.",
    "newey": "Newey, W. K., & West, K. D. (1987). A simple, positive semi-definite, heteroskedasticity and autocorrelation consistent covariance matrix. Econometrica, 55(3), 703–708.",
    "bp": "Breusch, T. S., & Pagan, A. R. (1979). A simple test for heteroscedasticity and random coefficient variation. Econometrica, 47(5), 1287–1294.",
    "koenker": "Koenker, R. (1981). A note on studentizing a test for heteroscedasticity. Journal of Econometrics, 17(1), 107–112.",
    "jb": "Jarque, C. M., & Bera, A. K. (1987). A test for normality of observations and regression residuals. International Statistical Review, 55(2), 163–172.",
    "ramsey": "Ramsey, J. B. (1969). Tests for specification errors in classical linear least-squares regression analysis. Journal of the Royal Statistical Society: Series B, 31(2), 350–371.",
    "dw": "Durbin, J., & Watson, G. S. (1950). Testing for serial correlation in least squares regression: I. Biometrika, 37(3/4), 409–428.",
    "breusch": "Breusch, T. S. (1978). Testing for autocorrelation in dynamic linear models. Australian Economic Papers, 17(31), 334–355.",
    "godfrey": "Godfrey, L. G. (1978). Testing against general autoregressive and moving average error models when the regressors include lagged dependent variables. Econometrica, 46(6), 1293–1301.",
    "johansen": "Johansen, S. (1995). Likelihood-based inference in cointegrated vector autoregressive models. Oxford University Press.",
    "mackinnon1999": "MacKinnon, J. G., Haug, A. A., & Michelis, L. (1999). Numerical distribution functions of likelihood ratio tests for cointegration. Journal of Applied Econometrics, 14(5), 563–577.",
    "arellano": "Arellano, M., & Bond, S. (1991). Some tests of specification for panel data: Monte Carlo evidence and an application to employment equations. Review of Economic Studies, 58(2), 277–297.",
}


def rapor_ayari(dil: str, bicim: str):
    """Rapor dilini etkinleştirir. Dönüş değeri DIL.reset ile kapatılır."""
    from .regresyon import EkonometriHatasi

    if dil not in ("tr", "en"):
        raise EkonometriHatasi("dil 'tr' ya da 'en' olmalı.")
    if bicim not in ("apa", "izgara"):
        raise EkonometriHatasi("bicim 'apa' ya da 'izgara' olmalı.")
    return DIL.set(dil)


# --------------------------------------------------------------------------- rapor nesnesi
@dataclass
class Tablo:
    no: int
    baslik: str
    sutunlar: list[str]
    satirlar: list[list[str]]
    notu: str = ""
    sol: int = 1  # soldan kaç sütun metin (sola yaslı)


@dataclass
class Sekil_:
    """Rapor içindeki şekil: numara, başlık, PNG baytları, isteğe bağlı not."""

    no: int
    baslik: str
    png: bytes
    notu: str = ""

    def base64(self) -> str:
        import base64

        return base64.b64encode(self.png).decode()


@dataclass
class Rapor:
    """Bulgu raporu. ``print(rapor)`` metni verir; ``rapor.kaydet("bulgular.docx")`` dosyaya yazar.

    bicim: "apa" üç çizgili tez tablosu, "izgara" tüm kenarlıkları çizili tablo.
    dil: "tr" ya da "en". Tablo başlığı ve sayı biçimi bu alana bakar.
    """

    baslik: str
    bloklar: list = field(
        default_factory=list
    )  # ("baslik", metin) | ("paragraf", metin) | ("denklem", metin) | ("tablo", Tablo) | ("liste", [..])
    bicim: str = "apa"
    dil: str = "tr"

    @property
    def tablolar(self) -> list[Tablo]:
        return [b[1] for b in self.bloklar if b[0] == "tablo"]

    @property
    def paragraflar(self) -> list[str]:
        return [b[1] for b in self.bloklar if b[0] == "paragraf"]

    @property
    def sekiller(self) -> list:
        return [b[1] for b in self.bloklar if b[0] == "sekil"]

    def sekil_ekle(self, sekil, baslik: str | None = None, aciklama: str = "") -> Rapor:
        """Rapora grafik ekler (``ek.kalinti_grafigi(...)`` gibi bir ``Sekil``). Şekiller sırayla numaralanır."""
        no = len(self.sekiller) + 1
        png = sekil.png() if hasattr(sekil, "png") else bytes(sekil)
        self.bloklar.append(("sekil", Sekil_(no, baslik or getattr(sekil, "baslik", "") or "", png, aciklama)))
        return self

    def _sekil_etiketi(self) -> str:
        return "Figure" if self.dil == "en" else "Şekil"

    # ---- metin (Markdown)
    def markdown(self) -> str:
        p = [f"# {self.baslik}", ""]
        for tur, ic in self.bloklar:
            if tur == "baslik":
                p += [f"## {ic}", ""]
            elif tur == "paragraf":
                p += [ic, ""]
            elif tur == "denklem":
                p += [f"> {ic}", ""]
            elif tur == "liste":
                p += [f"- {x}" for x in ic] + [""]
            elif tur == "tablo":
                etiket = "Table" if self.dil == "en" else "Tablo"
                not_etiket = "Note." if self.dil == "en" else "Not."
                p += [f"**{etiket} {ic.no}.** *{ic.baslik}*", ""]
                p.append("| " + " | ".join(ic.sutunlar) + " |")
                p.append("|" + "|".join([":---"] * ic.sol + ["---:"] * (len(ic.sutunlar) - ic.sol)) + "|")
                p += ["| " + " | ".join(h.replace("|", "\\|").replace("*", "\\*") for h in s) + " |" for s in ic.satirlar]
                if ic.notu:
                    p += ["", f"*{not_etiket}* {ic.notu.replace('*', chr(92) + '*')}"]
                p.append("")
            elif tur == "sekil":
                p += [f"**{self._sekil_etiketi()} {ic.no}.** *{ic.baslik}*", "", f"![{ic.baslik}](data:image/png;base64,{ic.base64()})", ""]
                if ic.notu:
                    p += [f"*{'Note.' if self.dil == 'en' else 'Not.'}* {ic.notu}", ""]
        return "\n".join(p).rstrip() + "\n"

    def __str__(self) -> str:
        return self.markdown()

    def __repr__(self) -> str:
        return f"<Rapor: '{self.baslik}', {len(self.tablolar)} tablo — print(rapor) ya da rapor.kaydet('bulgular.docx')>"

    # ---- HTML
    def html(self, tam_sayfa: bool = False) -> str:
        e = _html.escape
        p = [f"<h2>{e(self.baslik)}</h2>"]
        for tur, ic in self.bloklar:
            if tur == "baslik":
                p.append(f"<h3>{e(ic)}</h3>")
            elif tur == "paragraf":
                p.append(f"<p style='text-align:justify'>{e(ic)}</p>")
            elif tur == "denklem":
                p.append(f"<p style='text-align:center'><i>{e(ic)}</i></p>")
            elif tur == "liste":
                p.append("<ul>" + "".join(f"<li>{e(x)}</li>" for x in ic) + "</ul>")
            elif tur == "tablo":
                etiket = "Table" if self.dil == "en" else "Tablo"
                not_etiket = "Note." if self.dil == "en" else "Not."
                bas = "".join(
                    f"<th style='text-align:{'left' if i < ic.sol else 'right'};border-top:2px solid;border-bottom:1px solid;padding:3px 10px'>{e(s)}</th>"
                    for i, s in enumerate(ic.sutunlar)
                )
                govde = "".join(
                    "<tr>"
                    + "".join(
                        f"<td style='text-align:{'left' if i < ic.sol else 'right'};padding:3px 10px'>{e(h)}</td>" for i, h in enumerate(s)
                    )
                    + "</tr>"
                    for s in ic.satirlar
                )
                p.append(
                    f"<p><b>{etiket} {ic.no}</b><br><i>{e(ic.baslik)}</i></p>"
                    f"<table style='border-collapse:collapse;border-bottom:2px solid'><thead><tr>{bas}</tr></thead><tbody>{govde}</tbody></table>"
                    + (f"<p><small><i>{not_etiket}</i> {e(ic.notu)}</small></p>" if ic.notu else "")
                )
            elif tur == "sekil":
                p.append(
                    f"<p><b>{self._sekil_etiketi()} {ic.no}</b><br><i>{e(ic.baslik)}</i></p>"
                    f"<img alt='{e(ic.baslik)}' style='max-width:100%' src='data:image/png;base64,{ic.base64()}'>"
                    + (f"<p><small><i>{'Note.' if self.dil == 'en' else 'Not.'}</i> {e(ic.notu)}</small></p>" if ic.notu else "")
                )
        govde = "<div>" + "".join(p) + "</div>"
        if not tam_sayfa:
            return govde
        return (
            "<!doctype html><html lang='tr'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width, initial-scale=1'>"
            f"<title>{e(self.baslik)}</title><style>body{{font-family:'Times New Roman',serif;max-width:820px;margin:2rem auto;padding:0 16px;line-height:1.5}}"
            f"table{{font-size:.95em}}</style></head><body>{govde}</body></html>"
        )

    def _repr_html_(self) -> str:
        return self.html()

    # ---- dosya
    def kaydet(self, dosya: str | Path) -> Path:
        """Raporu dosyaya yazar. Uzantıya göre biçim seçilir: .docx (Word), .tex (LaTeX), .xlsx (Excel, yalnızca tablolar), .md, .html, .txt"""
        from .regresyon import EkonometriHatasi

        yol = Path(dosya)
        uz = yol.suffix.lower()
        if uz == ".docx":
            self._docx(yol)
        elif uz in (".md", ".txt"):
            yol.write_text(self.markdown(), encoding="utf-8")
        elif uz in (".html", ".htm"):
            yol.write_text(self.html(tam_sayfa=True), encoding="utf-8")
        elif uz == ".tex":
            yol.write_text(self.latex(yol), encoding="utf-8")
        elif uz == ".xlsx":
            tablolari_excele_yaz(self.tablolar, yol, self.dil)
        else:
            raise EkonometriHatasi(f"Desteklenmeyen dosya uzantısı: {uz!r}. Seçenekler: .docx, .tex, .xlsx, .md, .html, .txt")
        return yol

    def latex(self, yol: Path | None = None) -> str:
        """pdfLaTeX ile derlenebilen tam belge. Şekiller (varsa) .tex dosyasının yanına PNG olarak yazılır."""
        g = [
            "\\documentclass[12pt]{article}",
            "\\usepackage[utf8]{inputenc}",
            "\\usepackage[T1]{fontenc}",
            f"\\usepackage[{'english' if self.dil == 'en' else 'turkish'}]{{babel}}",
            "\\usepackage{booktabs,graphicx,amsmath,float}",
            "\\begin{document}",
            f"\\section*{{{latex_kacis(self.baslik)}}}",
        ]
        for tur, ic in self.bloklar:
            if tur == "baslik":
                g.append(f"\\subsection*{{{latex_kacis(ic)}}}")
            elif tur == "paragraf":
                g += [latex_kacis(ic), ""]
            elif tur == "denklem":
                g += ["\\begin{center}", f"\\textit{{{latex_kacis(ic)}}}", "\\end{center}"]
            elif tur == "liste":
                g += ["\\begin{itemize}"] + [f"\\item {latex_kacis(x)}" for x in ic] + ["\\end{itemize}"]
            elif tur == "tablo":
                g += [tablo_latex(ic, self.dil), ""]
            elif tur == "sekil":
                ad = f"{yol.stem if yol else 'rapor'}_sekil{ic.no}.png"
                if yol is not None:
                    (yol.parent / ad).write_bytes(ic.png)
                g += [
                    "\\begin{figure}[H]",
                    f"\\caption{{{latex_kacis(ic.baslik)}}}",
                    "\\centering",
                    f"\\includegraphics[width=0.9\\textwidth]{{{ad}}}",
                    "\\end{figure}",
                ]
        g.append("\\end{document}")
        return "\n".join(g) + "\n"

    def _docx(self, yol: Path) -> None:
        try:
            from docx import Document
            from docx.enum.table import WD_TABLE_ALIGNMENT
            from docx.enum.text import WD_ALIGN_PARAGRAPH
            from docx.oxml import OxmlElement
            from docx.oxml.ns import qn
            from docx.shared import Cm, Pt, RGBColor
        except ImportError:  # pragma: no cover
            from .regresyon import EkonometriHatasi

            raise EkonometriHatasi("Word çıktısı için 'python-docx' gerekli:  pip install python-docx") from None

        YAZI = "Times New Roman"
        d = Document()
        for ad, punto in (("Normal", 12), ("Heading 1", 14), ("Heading 2", 12)):
            st = d.styles[ad]
            st.font.name, st.font.size = YAZI, Pt(punto)
            st.font.color.rgb = RGBColor(0, 0, 0)
            rpr = st.element.get_or_add_rPr()
            rf = rpr.find(qn("w:rFonts"))
            if rf is None:
                rf = OxmlElement("w:rFonts")
                rpr.append(rf)
            for oz in ("w:ascii", "w:hAnsi", "w:eastAsia", "w:cs"):
                rf.set(qn(oz), YAZI)
            for tema in ("w:asciiTheme", "w:hAnsiTheme", "w:eastAsiaTheme", "w:cstheme"):
                rf.attrib.pop(qn(tema), None)
            if ad != "Normal":
                st.font.bold = True

        def kenar(hucre, **kenarlar):
            tcPr = hucre._tc.get_or_add_tcPr()
            b = tcPr.find(qn("w:tcBorders"))
            if b is None:
                b = OxmlElement("w:tcBorders")
                tcPr.append(b)
            for yon, kalinlik in kenarlar.items():
                el = OxmlElement(f"w:{yon}")
                el.set(qn("w:val"), "single")
                el.set(qn("w:sz"), str(kalinlik))
                el.set(qn("w:space"), "0")
                el.set(qn("w:color"), "000000")
                b.append(el)

        def paragraf(metin, hizala=WD_ALIGN_PARAGRAPH.JUSTIFY, italik=False, punto=None, aralik=1.5):
            p = d.add_paragraph()
            p.alignment = hizala
            p.paragraph_format.line_spacing = aralik
            r = p.add_run(metin)
            r.italic = italik
            if punto:
                r.font.size = Pt(punto)
            return p

        d.add_heading(self.baslik, level=1)
        for tur, ic in self.bloklar:
            if tur == "baslik":
                d.add_heading(ic, level=2)
            elif tur == "paragraf":
                paragraf(ic)
            elif tur == "denklem":
                paragraf(ic, hizala=WD_ALIGN_PARAGRAPH.CENTER, italik=True)
            elif tur == "liste":
                for x in ic:
                    p = paragraf(x, hizala=WD_ALIGN_PARAGRAPH.LEFT, aralik=1.15)
                    p.paragraph_format.left_indent = Pt(36)
                    p.paragraph_format.first_line_indent = Pt(-36)  # asılı girinti (APA kaynakça)
            elif tur == "tablo":
                p = d.add_paragraph()
                p.paragraph_format.space_after = Pt(0)
                p.add_run(f"{'Table' if self.dil == 'en' else 'Tablo'} {ic.no}").bold = True
                p = d.add_paragraph()
                p.add_run(ic.baslik).italic = True
                t = d.add_table(rows=1 + len(ic.satirlar), cols=len(ic.sutunlar))
                t.alignment = WD_TABLE_ALIGNMENT.LEFT
                t.autofit = False
                # sütun genişliği içeriğe göre (toplam ~16 cm), sayılar satır atlamasın
                genislik, punto = _sutun_genislikleri(ic)
                genislik = [Cm(g) for g in genislik]
                for j, g in enumerate(genislik):
                    t.columns[j].width = g
                govde_satir = len(ic.satirlar)
                for i, satir in enumerate([ic.sutunlar] + ic.satirlar):
                    for j, h in enumerate(satir):
                        hucre = t.cell(i, j)
                        hucre.width = genislik[j]
                        if j >= ic.sol:
                            h = h.replace("-", "\u2212")  # gerçek eksi işareti: satır sonunda bölünmez
                        hp = hucre.paragraphs[0]
                        hp.alignment = WD_ALIGN_PARAGRAPH.LEFT if j < ic.sol else WD_ALIGN_PARAGRAPH.RIGHT
                        hp.paragraph_format.space_after = Pt(0)
                        run = hp.add_run(h)
                        run.font.size = Pt(punto)
                        run.bold = i == 0
                        if self.bicim == "izgara":
                            kenar(hucre, top=12 if i == 0 else 4, bottom=12 if i == govde_satir else 4, left=4, right=4)
                        else:  # APA: yalnızca üst, başlık altı ve alt çizgi
                            if i == 0:
                                kenar(hucre, top=12, bottom=6)
                            if i == govde_satir:
                                kenar(hucre, bottom=12)
                if ic.notu:
                    p = d.add_paragraph()
                    r = p.add_run("Note. " if self.dil == "en" else "Not. ")
                    r.italic = True
                    r.font.size = Pt(10)
                    p.add_run(ic.notu).font.size = Pt(10)
                else:
                    d.add_paragraph()
            elif tur == "sekil":
                import io as _io

                p = d.add_paragraph()
                p.paragraph_format.space_after = Pt(0)
                p.add_run(f"{self._sekil_etiketi()} {ic.no}").bold = True
                p = d.add_paragraph()
                p.add_run(ic.baslik).italic = True
                d.add_picture(_io.BytesIO(ic.png), width=Cm(15))
                if ic.notu:
                    p = d.add_paragraph()
                    r = p.add_run("Note. " if self.dil == "en" else "Not. ")
                    r.italic = True
                    r.font.size = Pt(10)
                    p.add_run(ic.notu).font.size = Pt(10)
        d.save(str(yol))


_LATEX_OZEL = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}
_LATEX_UNICODE = {
    "β": r"$\beta$",
    "χ": r"$\chi$",
    "ρ": r"$\rho$",
    "σ": r"$\sigma$",
    "θ": r"$\theta$",
    "Δ": r"$\Delta$",
    "Λ": r"$\Lambda$",
    "Φ": r"$\Phi$",
    "α": r"$\alpha$",
    "λ": r"$\lambda$",
    "μ": r"$\mu$",
    "ε": r"$\varepsilon$",
    "π": r"$\pi$",
    "τ": r"$\tau$",
    "²": r"$^{2}$",
    "³": r"$^{3}$",
    "⁴": r"$^{4}$",
    "¹": r"$^{1}$",
    "⁻": r"$^{-}$",
    "₀": r"$_{0}$",
    "₁": r"$_{1}$",
    "₂": r"$_{2}$",
    "₃": r"$_{3}$",
    "₄": r"$_{4}$",
    "₅": r"$_{5}$",
    "₆": r"$_{6}$",
    "₇": r"$_{7}$",
    "₈": r"$_{8}$",
    "₉": r"$_{9}$",
    "−": "$-$",
    "–": "--",
    "—": "---",
    "×": r"$\times$",
    "·": r"$\cdot$",
    "≥": r"$\geq$",
    "≤": r"$\leq$",
    "≈": r"$\approx$",
    "≠": r"$\neq$",
    "→": r"$\rightarrow$",
    "∞": r"$\infty$",
    "√": r"$\surd$",
    "ŷ": r"$\hat{y}$",
    "ᵇ": r"$^{b}$",
    "◂": "",
    "…": r"\ldots{}",
    "’": "'",
    "“": "``",
    "”": "''",
}


def latex_kacis(metin: str) -> str:
    """Düz metni pdfLaTeX'te güvenle derlenecek biçime çevirir (özel karakterler, Yunan harfleri, üst/alt simgeler)."""
    out = []
    for c in str(metin):
        out.append(_LATEX_OZEL.get(c) or _LATEX_UNICODE.get(c) or c)
    return "".join(out).replace("$$", "")


def tablo_latex(ic: Tablo, dil: str = "tr") -> str:
    """Tez tablosu: booktabs (üç çizgi), başlık ve not. Stata ``esttab using tablo.tex`` karşılığı."""
    hiza = "l" * ic.sol + "r" * (len(ic.sutunlar) - ic.sol)
    satirlar = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{latex_kacis(ic.baslik)}}}",
        f"\\begin{{tabular}}{{{hiza}}}",
        "\\toprule",
        " & ".join(latex_kacis(x) for x in ic.sutunlar) + r" \\",
        "\\midrule",
    ]
    satirlar += [
        " & ".join(latex_kacis(h).replace("-", "$-$") if j >= ic.sol and h.startswith("-") else latex_kacis(h) for j, h in enumerate(s))
        + r" \\"
        for s in ic.satirlar
    ]
    satirlar += ["\\bottomrule", "\\end{tabular}"]
    if ic.notu:
        satirlar.append(f"\\par\\smallskip\\footnotesize\\textit{{{'Note.' if dil == 'en' else 'Not.'}}} {latex_kacis(ic.notu)}")
    satirlar.append("\\end{table}")
    return "\n".join(satirlar)


def tablolari_excele_yaz(tablolar: list[Tablo], yol: Path, dil: str = "tr") -> Path:
    """Her tabloyu ayrı bir Excel sayfasına yazar (başlık, sütun başlıkları kalın, not altta)."""
    from openpyxl import Workbook
    from openpyxl.styles import Alignment, Border, Font, Side

    from .regresyon import EkonometriHatasi

    if not tablolar:
        raise EkonometriHatasi("Raporda tablo yok; Excel çıktısı yalnızca tabloları içerir.")
    kitap = Workbook()
    kitap.remove(kitap.active)
    cizgi = Side(style="thin")
    for ic in tablolar:
        sayfa = kitap.create_sheet(f"{'Table' if dil == 'en' else 'Tablo'} {ic.no}"[:31])
        sayfa.append([f"{'Table' if dil == 'en' else 'Tablo'} {ic.no}. {ic.baslik}"])
        sayfa["A1"].font = Font(bold=True)
        sayfa.append(list(ic.sutunlar))
        for hucre in sayfa[2]:
            hucre.font = Font(bold=True)
            hucre.border = Border(top=cizgi, bottom=cizgi)
        for satir in ic.satirlar:
            sayfa.append(list(satir))
        for hucre in sayfa[sayfa.max_row]:
            hucre.border = Border(bottom=cizgi)
        if ic.notu:
            sayfa.append([])
            sayfa.append([f"{'Note.' if dil == 'en' else 'Not.'} {ic.notu}"])
        for j, _ in enumerate(ic.sutunlar):
            harf = sayfa.cell(row=2, column=j + 1).column_letter
            sayfa.column_dimensions[harf].width = max(10, min(45, max(len(str(r[j])) for r in [ic.sutunlar] + ic.satirlar) + 2))
            if j >= ic.sol:
                for r in range(2, 3 + len(ic.satirlar)):
                    sayfa.cell(row=r, column=j + 1).alignment = Alignment(horizontal="right")
    kitap.save(yol)
    return yol


def _sutun_genislikleri(ic: Tablo, sayfa_cm: float = 15.6) -> tuple[list[float], float]:
    """Word tablosu için sütun genişlikleri (cm) ve yazı boyutu (punto).

    Veri hücreleri bölünmeyecek kadar geniş tutulur; başlıklar ve sola yaslı metin sütunları kelime sınırından kırılabilir.
    Tablo sayfaya sığmazsa yazı boyutu 11 punktodan en fazla 8 punktoya kadar küçültülür.
    """
    KARAKTER = 0.0185  # Times New Roman'da 1 puntoluk ortalama karakter genişliği (cm)
    BOSLUK = 0.42  # hücre iç kenar boşlukları (cm)
    gerek = []
    for j, bas in enumerate(ic.sutunlar):
        veri = [str(s[j]) for s in ic.satirlar]
        if j < ic.sol:  # metin sütunu: en uzun kelime sığsın, en uzun hücre tercihen
            en_kelime = max([len(w) for x in veri + [bas] for w in x.split()] or [1])
            gerek.append(max(en_kelime, min(max([len(x) for x in veri] or [1]), 26)))
        else:
            gerek.append(max([len(x) for x in veri] + [max((len(w) for w in bas.split()), default=1)]))
    toplam = sum(gerek)
    punto = 11.0
    while punto > 8 and toplam * punto * KARAKTER + BOSLUK * len(gerek) > sayfa_cm:
        punto -= 0.5
    genislik = [g * punto * KARAKTER + BOSLUK for g in gerek]
    olcek = sayfa_cm / sum(genislik) if sum(genislik) > sayfa_cm else 1.0
    return [g * olcek for g in genislik], punto


# --------------------------------------------------------------------------- cümle yardımcıları
def _p(p: float) -> str:
    """Akademik p gösterimi: 'p < 0,001' ya da 'p = 0,012'."""
    if p < 0.001:
        return "p < 0.001" if DIL.get() == "en" else "p < 0,001"
    return f"p = {sayi(p, 3)}"


def _p_hucre(p: float | None) -> str:
    if p is None:
        return "—"
    if p < 0.001:
        return "<0.001" if DIL.get() == "en" else "<0,001"
    return sayi(p, 3)


def _duzey(p: float) -> str:
    if DIL.get() == "en":
        if p < 0.01:
            return "statistically significant at the 1% level"
        if p < 0.05:
            return "statistically significant at the 5% level"
        if p < 0.10:
            return "statistically significant only at the 10% level"
        return "not statistically significant"
    if p < 0.01:
        return "%1 düzeyinde istatistiksel olarak anlamlıdır"
    if p < 0.05:
        return "%5 düzeyinde istatistiksel olarak anlamlıdır"
    if p < 0.10:
        return "yalnızca %10 düzeyinde istatistiksel olarak anlamlıdır"
    return "istatistiksel olarak anlamlı değildir"


class _Adlar:
    """Değişken adlarının metindeki ve tablodaki görünümü (kullanıcı etiketi varsa o kullanılır)."""

    def __init__(self, etiketler: dict[str, str] | None):
        self.e = dict(etiketler or {})

    def yalin(self, ad: str) -> str:
        return self.e.get(ad, ad)

    def metin(self, ad: str) -> str:
        # Türkçe ekler ada göre değiştiği için addan sonra hep sabit bir sözcük ("değişkeni" vb.) gelir.
        return self.e[ad] if ad in self.e else f"'{ad}'"

    def satir(self, t) -> str:
        if t.tur == "sabit":
            return l("Sabit", "Constant")
        if t.tur == "kukla":
            return f"{self.yalin(t.degisken)}: {t.seviye}"
        if t.tur == "kare":
            return f"{self.yalin(t.degisken)}²"
        if t.tur == "log":
            return f"ln({self.yalin(t.degisken)})"
        if t.tur == "surekli":
            return self.yalin(t.degisken)
        if t.parcalar:
            return " × ".join(self.satir(p) for p in t.parcalar)
        return t.sutun


def _ist(b: float, se: float, t: float, p: float, ad: str = "t") -> str:
    se_ad = "SE" if DIL.get() == "en" else "SH"
    return f"(β = {tablo_sayi(b)}; {se_ad} = {tablo_sayi(se)}; {ad} = {sayi(t, 2)}; {_p(p)})"


# --------------------------------------------------------------------------- bölümler
def _tanimlayici(sonuc, adlar: _Adlar, no: int) -> tuple[list, int]:
    bloklar: list = []
    v = getattr(sonuc, "_veri", None)
    if v is None or v.shape[1] == 0:
        v = pd.concat([sonuc.y, sonuc.X.drop(columns=[c for c in sonuc.X.columns if c == "Intercept"])], axis=1)
    kuklalar = {t.degisken for t in sonuc._terimler if t.tur == "kukla"}  # C(x) ile kategorik kullanılan sayısal sütunlar dâhil
    sayisal = [c for c in v.columns if c not in kuklalar and pd.api.types.is_numeric_dtype(v[c]) and not pd.api.types.is_bool_dtype(v[c])]
    kategorik = [c for c in v.columns if c not in sayisal]
    atif = []
    if sayisal:
        satirlar = [
            [
                adlar.yalin(c),
                sayi(int(v[c].count()), 0),
                sayi(float(v[c].mean()), 2),
                sayi(float(v[c].std(ddof=1)), 2),
                sayi(float(v[c].min()), 2),
                sayi(float(v[c].max()), 2),
            ]
            for c in sayisal
        ]
        bloklar.append(
            (
                "tablo",
                Tablo(
                    no,
                    l("Sürekli değişkenlere ilişkin tanımlayıcı istatistikler", "Descriptive statistics for continuous variables"),
                    [
                        l("Değişken", "Variable"),
                        "N",
                        l("Ortalama", "Mean"),
                        l("Std. sapma", "Std. dev."),
                        l("En küçük", "Minimum"),
                        l("En büyük", "Maximum"),
                    ],
                    satirlar,
                ),
            )
        )
        atif.append(l(f"Tablo {no}", f"Table {no}"))
        no += 1
    if kategorik:
        satirlar = []
        for c in kategorik:
            say = v[c].value_counts(sort=False)
            if isinstance(v[c].dtype, pd.CategoricalDtype):
                say = say.reindex(v[c].cat.categories).dropna()
            else:
                say = say.sort_index()
            for i, (kat, n) in enumerate(say.items()):
                satirlar.append([adlar.yalin(c) if i == 0 else "", str(kat), sayi(int(n), 0), sayi(100 * n / say.sum(), 1)])
        bloklar.append(
            (
                "tablo",
                Tablo(
                    no,
                    l("Kategorik değişkenlerin frekans dağılımı", "Frequency distribution of categorical variables"),
                    [l("Değişken", "Variable"), l("Kategori", "Category"), "n", "%"],
                    satirlar,
                    sol=2,
                ),
            )
        )
        atif.append(l(f"Tablo {no}", f"Table {no}"))
        no += 1
    from .regresyon import _log_mu

    y_ad = _log_mu(sonuc.bagimli) or sonuc.bagimli
    giris = l(
        f"Analiz {sayi(sonuc.n, 0)} gözlem üzerinden gerçekleştirilmiştir. Modelde yer alan değişkenlere ilişkin tanımlayıcı istatistikler aşağıda sunulmuştur ({'; '.join(atif)}).",
        f"The analysis uses {sayi(sonuc.n, 0)} observations. Descriptive statistics for the variables in the model are reported below ({'; '.join(atif)}).",
    )
    if y_ad in sayisal:
        giris += l(
            f" Bağımlı değişken olan {adlar.metin(y_ad)} değişkeninin ortalaması {sayi(float(v[y_ad].mean()), 2)} "
            f"(SS = {sayi(float(v[y_ad].std(ddof=1)), 2)}) olarak hesaplanmıştır.",
            f" The dependent variable {adlar.metin(y_ad)} has a mean of {sayi(float(v[y_ad].mean()), 2)} "
            f"(SD = {sayi(float(v[y_ad].std(ddof=1)), 2)}).",
        )
    return [("baslik", l("Tanımlayıcı İstatistikler", "Descriptive statistics")), ("paragraf", giris)] + bloklar, no


def _regresyon(sonuc, adlar: _Adlar, no: int) -> tuple[list, int]:
    from .regresyon import _log_mu

    y_log = _log_mu(sonuc.bagimli)
    y_ad = y_log or sonuc.bagimli
    Y = adlar.metin(y_ad)
    b, se, tt, pp = sonuc.katsayilar, sonuc.standart_hatalar, sonuc.t_degerleri, sonuc.p_degerleri
    ga = sonuc.guven_araliklari(0.95)

    # ---- tablo
    satirlar = [
        [
            adlar.satir(t),
            tablo_sayi(b[t.sutun]) + yildiz(pp[t.sutun]),
            tablo_sayi(se[t.sutun]),
            sayi(tt[t.sutun], 2),
            _p_hucre(pp[t.sutun]),
            f"[{tablo_sayi(ga.loc[t.sutun, 'alt'])}; {tablo_sayi(ga.loc[t.sutun, 'üst'])}]",
        ]
        for t in sonuc._terimler
    ]
    f_metin = f"F({sonuc.f_serbestlik[0]}, {sonuc.f_serbestlik[1]}) = {sayi(sonuc.f_istatistigi, 2)}"
    referanslar = {}
    for t in sonuc._terimler:
        if t.tur == "kukla":
            referanslar.setdefault(t.degisken, t.referans)
    notu = (
        f"N = {sayi(sonuc.n, 0)}; R² = {sayi(sonuc.r2, 3)}; {l('düzeltilmiş R²', 'adjusted R²')} = {sayi(sonuc.duzeltilmis_r2, 3)}; {f_metin} ({_p(sonuc.f_p_degeri)}). "
        f"{l('Bağımlı değişken', 'Dependent variable')}: {'ln(' + adlar.yalin(y_ad) + ')' if y_log else adlar.yalin(y_ad)}. "
        f"{l('Standart hatalar', 'Standard errors')}: {sonuc.kovaryans_aciklama}. "
        + (
            (l("Referans gruplar: ", "Reference groups: ") + "; ".join(f"{adlar.yalin(k)} = {r}" for k, r in referanslar.items()) + ". ")
            if referanslar
            else ""
        )
        + l(
            "SH: standart hata; GA: güven aralığı. *** p < 0,01; ** p < 0,05; * p < 0,10.",
            "SE: standard error; CI: confidence interval. *** p < 0.01; ** p < 0.05; * p < 0.10.",
        )
    )
    tablo = Tablo(
        no,
        l("EKK regresyon sonuçları", "OLS regression results"),
        [l("Değişken", "Variable"), "β", l("SH", "SE"), "t", "p", l("%95 GA", "95% CI")],
        satirlar,
        notu,
    )

    # ---- model geneli
    if DIL.get() == "en":
        anlam = (
            "The model is jointly statistically significant"
            if sonuc.f_p_degeri < 0.05
            else "The model is not jointly statistically significant"
        )
        hedef = f"the logarithm of {Y}" if y_log else Y
        genel = (
            f"OLS estimates are reported in Table {no}. {anlam} ({f_metin}; {_p(sonuc.f_p_degeri)}). "
            f"The regressors account for {yuzde(sonuc.r2 * 100)} of the variation in {hedef} "
            f"(R² = {sayi(sonuc.r2, 3)}; adjusted R² = {sayi(sonuc.duzeltilmis_r2, 3)})."
        )
    else:
        genel = (
            f"EKK tahmin sonuçları Tablo {no} ile sunulmuştur. "
            + (
                f"Model bir bütün olarak istatistiksel olarak anlamlıdır ({f_metin}; {_p(sonuc.f_p_degeri)}). "
                if sonuc.f_p_degeri < 0.05
                else f"Model bir bütün olarak istatistiksel olarak anlamlı bulunmamıştır ({f_metin}; {_p(sonuc.f_p_degeri)}). "
            )
            + f"Açıklayıcı değişkenler, {Y} değişkeninin {'logaritmasındaki ' if y_log else ''}değişimin {yuzde(sonuc.r2 * 100)} kadarını açıklamaktadır "
            f"(R² = {sayi(sonuc.r2, 3)}; düzeltilmiş R² = {sayi(sonuc.duzeltilmis_r2, 3)})."
        )

    bloklar = [("baslik", l("Regresyon Bulguları", "Regression results")), ("paragraf", genel), ("tablo", tablo)] + katsayi_paragraflari(
        sonuc, adlar
    )
    return bloklar, no + 1


def katsayi_paragraflari(sonuc, adlar: _Adlar) -> list:
    """Katsayıların akademik Türkçe yorumu (EKK, panel ve araç değişken modelleri için ortak)."""
    from .regresyon import _log_mu

    y_log = _log_mu(sonuc.bagimli)
    y_ad = y_log or sonuc.bagimli
    Y = adlar.metin(y_ad)
    varsayilan_birim = "unit" if DIL.get() == "en" else "birim"
    y_birim = sonuc._birimler.get(y_ad, varsayilan_birim)
    b, se, tt, pp = sonuc.katsayilar, sonuc.standart_hatalar, sonuc.t_degerleri, sonuc.p_degerleri
    ia = getattr(sonuc, "ist_adi", "t")

    # ---- katsayılar
    kareler = {t.degisken: t for t in sonuc._terimler if t.tur == "kare"}
    surekli_c: list[str] = []
    etkilesim_c: list[str] = []
    kukla_gruplari: dict[str, list] = {}
    for t in sonuc._terimler:
        if t.tur in ("sabit", "kare"):
            continue
        s = t.sutun
        bi, pi = float(b[s]), float(pp[s])
        ist = _ist(bi, float(se[s]), float(tt[s]), pi, ia)
        x = t.degisken
        X = adlar.metin(x)
        xb = sonuc._birimler.get(x, varsayilan_birim)
        if t.tur == "kukla":
            kukla_gruplari.setdefault(x, []).append((t, bi, pi, ist))
            continue
        if t.tur == "etkilesim":
            from .yorum import etkilesim_cumlesi

            ec = etkilesim_cumlesi(t, bi, Y, ad=adlar.metin, akademik=True)
            ad_et = adlar.satir(t)
            if ec and pi < 0.10:
                etkilesim_c.append(l(f"{ec}; bu fark {_duzey(pi)} {ist}.", f"{ec}; this difference is {_duzey(pi)} {ist}."))
            elif pi >= 0.10:
                etkilesim_c.append(
                    l(
                        f"{ad_et} etkileşim teriminin katsayısı istatistiksel olarak anlamlı bulunmamıştır {ist}; etkinin diğer değişkene göre farklılaştığına dair kanıt elde edilememiştir.",
                        f"The coefficient on the {ad_et} interaction is not statistically significant {ist}; there is no evidence that the effect differs.",
                    )
                )
            else:
                etkilesim_c.append(
                    l(
                        f"{ad_et} etkileşim teriminin katsayısı {_duzey(pi)} {ist}.",
                        f"The coefficient on the {ad_et} interaction is {_duzey(pi)} {ist}.",
                    )
                )
            continue
        if t.tur == "diger":
            surekli_c.append(
                l(
                    f"{adlar.metin(s)} teriminin katsayısı {_duzey(pi)} {ist}.",
                    f"The coefficient on {adlar.metin(s)} is {_duzey(pi)} {ist}.",
                )
            )
            continue
        if t.tur == "surekli" and x in kareler:
            k = kareler[x].sutun
            b2, p2 = float(b[k]), float(pp[k])
            donum = -bi / (2 * b2) if b2 != 0 else float("nan")
            kare_ist = _ist(b2, float(se[k]), float(tt[k]), p2, ia)
            if DIL.get() == "en":
                sekil = "an inverted-U" if b2 < 0 else "a U-shaped"
                c = (
                    f"The relationship between {X} and {Y} is modeled as nonlinear: the linear term is "
                    f"{'positive' if bi > 0 else 'negative'} {ist}, and the squared term is "
                    f"{'positive' if b2 > 0 else 'negative'} {kare_ist}. "
                )
                if p2 < 0.10:
                    c += f"This is consistent with {sekil} relationship, with a turning point of about {sayi(donum, 2)} {xb}."
                else:
                    c += "The squared term is not statistically significant, so the evidence for a nonlinear relationship is weak."
            else:
                sekil = "ters U" if b2 < 0 else "U"
                c = (
                    f"{X} değişkeni ile {Y} değişkeni arasındaki ilişki doğrusal olmayan biçimde modellenmiştir: doğrusal terimin katsayısı "
                    f"{'pozitif' if bi > 0 else 'negatif'} {ist}, kare terimin katsayısı ise {'pozitif' if b2 > 0 else 'negatif'} bulunmuştur "
                    f"{kare_ist}. "
                )
                if p2 < 0.10:
                    c += f"Bu bulgu {sekil} biçimli bir ilişkiye işaret etmekte olup dönüm noktası yaklaşık {sayi(donum, 2)} {xb} olarak hesaplanmıştır."
                else:
                    c += "Kare terim istatistiksel olarak anlamlı olmadığından doğrusal olmayan bir ilişkiye dair kanıt zayıftır."
            surekli_c.append(c)
            continue
        if pi >= 0.10:
            surekli_c.append(
                l(
                    f"{X} değişkeninin {Y} değişkeni üzerindeki etkisi istatistiksel olarak anlamlı bulunmamıştır {ist}.",
                    f"The effect of {X} on {Y} is not statistically significant {ist}.",
                )
            )
            continue
        if DIL.get() == "en":
            direction = "increases" if bi > 0 else "decreases"
            olcu = "units" if y_birim == "unit" else y_birim
            if t.tur == "log":
                kosul = f"a 1% increase in {X}"
                buyukluk = f"by about {yuzde(abs(bi), 2)}" if y_log else f"by about {sayi(abs(bi) / 100, 4)} {olcu}"
            else:
                kosul = f"an increase of one {xb} in {X}"
                buyukluk = f"by about {yuzde(abs(100 * (np.exp(bi) - 1)), 2)}" if y_log else f"by about {sayi(abs(bi), 3)} {olcu}"
            surekli_c.append(f"Holding other variables fixed, {kosul} {direction} {Y} {buyukluk}; the effect is {_duzey(pi)} {ist}.")
            continue
        yon = "artmaktadır" if bi > 0 else "azalmaktadır"
        if t.tur == "log":
            kosul = f"{X} değişkeni %1 arttığında"
            buyukluk = f"yaklaşık {yuzde(abs(bi), 2)}" if y_log else f"yaklaşık {sayi(abs(bi) / 100, 4)} {y_birim}"
        else:
            kosul = f"{X} değişkeni 1 {xb} arttığında"
            buyukluk = f"ortalama {yuzde(abs(100 * (np.exp(bi) - 1)), 2)}" if y_log else f"ortalama {sayi(abs(bi), 3)} {y_birim}"
        surekli_c.append(f"Diğer değişkenler sabitken, {kosul} {Y} değişkeni {buyukluk} {yon}; bu etki {_duzey(pi)} {ist}.")

    kukla_c: list[str] = []
    for x, uyeler in kukla_gruplari.items():
        X = adlar.metin(x)
        ref = uyeler[0][0].referans
        parcalar_anlamli, parcalar_anlamsiz = [], []
        for t, bi, pi, ist in uyeler:
            if pi < 0.10:
                if y_log:
                    fark = yuzde(abs(100 * (np.exp(bi) - 1)))
                else:
                    olcu = "units" if DIL.get() == "en" and y_birim == "unit" else y_birim
                    fark = f"{sayi(abs(bi), 3)} {olcu}"
                if DIL.get() == "en":
                    parcalar_anlamli.append(f"on average {fark} {'higher' if bi > 0 else 'lower'} in the '{t.seviye}' group {ist}")
                else:
                    parcalar_anlamli.append(f"'{t.seviye}' grubunda ortalama {fark} daha {'yüksek' if bi > 0 else 'düşük'} {ist}")
            else:
                parcalar_anlamsiz.append(f"'{t.seviye}' {ist}")
        if DIL.get() == "en":
            c = f"The reference group for {X} is '{ref}'."
            if parcalar_anlamli:
                c += f" Holding other variables fixed, relative to the reference group, {Y} is " + "; ".join(parcalar_anlamli) + "."
            if parcalar_anlamsiz:
                etiket = "group" if len(parcalar_anlamsiz) == 1 else "groups"
                c += (
                    f" The difference from the reference group is not statistically significant for the following {etiket}: "
                    + "; ".join(parcalar_anlamsiz)
                    + "."
                )
        else:
            c = f"{X} değişkeni için referans grup '{ref}' olarak alınmıştır."
            if parcalar_anlamli:
                c += f" Diğer değişkenler sabitken {Y} düzeyi, referans gruba kıyasla " + "; ".join(parcalar_anlamli) + " bulunmuştur."
            if parcalar_anlamsiz:
                c += (
                    " Referans grupla aralarındaki fark istatistiksel olarak anlamlı bulunmayan "
                    + ("grup" if len(parcalar_anlamsiz) == 1 else "gruplar")
                    + ": "
                    + "; ".join(parcalar_anlamsiz)
                    + "."
                )
        kukla_c.append(c)

    bloklar: list = []
    if surekli_c:
        bloklar.append(("paragraf", " ".join(surekli_c)))
    bloklar += [("paragraf", c) for c in kukla_c]
    if etkilesim_c:
        bloklar.append(
            (
                "paragraf",
                " ".join(etkilesim_c)
                + l(
                    " Etkileşime giren değişkenlerin ana etki katsayıları, diğer değişkenin sıfır olduğu ya da referans grupta bulunduğu durumu yansıtmaktadır.",
                    " The main-effect coefficients apply when the other variable is zero or in the reference group.",
                ),
            )
        )
    return bloklar


def _varsayim(sonuc, vr, no: int) -> tuple[list, int]:
    def bul(on):
        return next((t for t in vr.testler if t.ad.startswith(on)), None)

    def karar(t) -> str:
        if t.p_degeri is None:
            return l("Sorun görüldü" if t.sorun_var else "Sorun görülmedi", "Problem detected" if t.sorun_var else "No problem detected")
        return l("H₀ reddedildi" if t.p_degeri < 0.05 else "H₀ reddedilemedi", "Rejected" if t.p_degeri < 0.05 else "Not rejected")

    varsayim_en = {
        "Sabit varyans": "Homoskedasticity",
        "Otokorelasyon yok": "No autocorrelation",
        "Normal hatalar": "Normal errors",
        "Çoklu doğrusal bağlantı yok": "No multicollinearity",
        "Doğru fonksiyonel biçim": "Correct functional form",
    }
    satirlar = [
        [t.ad, l(t.varsayim, varsayim_en.get(t.varsayim, t.varsayim)), sayi(t.istatistik, 3), _p_hucre(t.p_degeri), karar(t)]
        for t in vr.testler
    ]
    tablo = Tablo(
        no,
        l("Varsayım testleri", "Assumption tests"),
        [l("Test", "Test"), l("Sınanan varsayım", "Assumption"), l("İstatistik", "Statistic"), "p", l("Karar", "Decision")],
        satirlar,
        l(
            "H₀ her testte ilgili varsayımın sağlandığını ifade eder; kararlar %5 anlamlılık düzeyine göredir. Durbin-Watson ve VIF için p-değeri hesaplanmaz.",
            "In each test, H₀ states that the assumption holds; decisions use the 5% significance level. A p-value is not computed for Durbin–Watson or VIF.",
        ),
        sol=2,
    )
    kov = sonuc.kovaryans_turu
    c: list[str] = [
        l(
            f"Modelin varsayımlarına ilişkin test sonuçları Tablo {no} ile sunulmuştur.",
            f"Tests of the model assumptions are reported in Table {no}.",
        )
    ]

    bp, wh = bul("Breusch-Pagan"), bul("White")
    if bp and wh:
        c.append(
            l(
                f"Değişen varyans, Breusch-Pagan (LM = {sayi(bp.istatistik, 2)}; {_p(bp.p_degeri)}) ve White (LM = {sayi(wh.istatistik, 2)}; {_p(wh.p_degeri)}) testleriyle sınanmıştır.",
                f"Heteroskedasticity was tested with Breusch-Pagan (LM = {sayi(bp.istatistik, 2)}; {_p(bp.p_degeri)}) and White (LM = {sayi(wh.istatistik, 2)}; {_p(wh.p_degeri)}).",
            )
        )
        if bp.p_degeri < 0.05 or wh.p_degeri < 0.05:
            ikisi = bp.p_degeri < 0.05 and wh.p_degeri < 0.05
            c.append(
                l(
                    ("Her iki test de" if ikisi else "Testlerden biri") + " sabit varyans varsayımının sağlanmadığını göstermektedir.",
                    "Both tests indicate that the homoskedasticity assumption fails."
                    if ikisi
                    else "One of the tests indicates that the homoskedasticity assumption fails.",
                )
            )
            if kov in ("HC0", "HC1", "HC2", "HC3"):
                c.append(
                    l(
                        f"Bu nedenle çıkarımlar değişen varyansa dayanıklı standart hatalara ({kov}) dayandırılmıştır.",
                        f"Inference is therefore based on heteroskedasticity-robust {kov} standard errors.",
                    )
                )
            elif kov == "HAC":
                c.append(
                    l(
                        "Bu nedenle çıkarımlar değişen varyans ve otokorelasyona dayanıklı Newey-West (HAC) standart hatalarına dayandırılmıştır.",
                        "Inference is therefore based on Newey-West (HAC) standard errors, which are robust to heteroskedasticity and autocorrelation.",
                    )
                )
            else:
                c.append(
                    l(
                        "Katsayı tahminleri bu durumda yansız kalmakla birlikte klasik standart hatalar güvenilir değildir; modelin değişen varyansa dayanıklı standart hatalarla yeniden tahmin edilmesi gerekir.",
                        "The coefficient estimates remain unbiased, but the classical standard errors are not reliable; the model should be re-estimated with heteroskedasticity-robust standard errors.",
                    )
                )
        else:
            c.append(
                l(
                    "Her iki test de sabit varyans varsayımının reddedilemediğini göstermektedir.",
                    "Neither test rejects the homoskedasticity assumption.",
                )
            )

    jb = bul("Jarque-Bera")
    if jb:
        jb_sayi = f"JB = {sayi(jb.istatistik, 2)}; {_p(jb.p_degeri)}"
        if jb.p_degeri < 0.05:
            c.append(
                l(
                    f"Jarque-Bera testi kalıntıların normal dağılmadığına işaret etmektedir ({jb_sayi}).",
                    f"The Jarque-Bera test indicates that the residuals are not normally distributed ({jb_sayi}).",
                )
            )
            c.append(
                l(
                    f"Bununla birlikte örneklem büyüklüğü (N = {sayi(sonuc.n, 0)}) dikkate alındığında, merkezî limit teoremi uyarınca t ve F testleri yaklaşık olarak geçerlidir."
                    if sonuc.n >= 100
                    else "Örneklem küçük olduğundan t ve F testlerinin sonuçları temkinli yorumlanmalıdır.",
                    f"Given the sample size (N = {sayi(sonuc.n, 0)}), the central limit theorem implies that the t and F tests are approximately valid."
                    if sonuc.n >= 100
                    else "Because the sample is small, the t and F tests should be interpreted with caution.",
                )
            )
        else:
            c.append(
                l(
                    f"Jarque-Bera testi kalıntıların normal dağıldığı hipotezinin reddedilemediğini göstermektedir ({jb_sayi}).",
                    f"The Jarque-Bera test does not reject normality of the residuals ({jb_sayi}).",
                )
            )

    rs = bul("Ramsey")
    if rs:
        q, sd = rs.detay.get("serbestlik", ("", ""))
        ist = f"(F({q}, {sd}) = {sayi(rs.istatistik, 2)}; {_p(rs.p_degeri)})"
        c.append(
            l(
                f"Ramsey RESET testi fonksiyonel biçim hatasına ya da atlanmış değişkene işaret etmektedir {ist}; model kurgusunun gözden geçirilmesi gerekir."
                if rs.p_degeri < 0.05
                else f"Ramsey RESET testi, modelde fonksiyonel biçim hatası bulunmadığı hipotezinin reddedilemediğini göstermektedir {ist}.",
                f"The Ramsey RESET test indicates a functional-form error or an omitted variable {ist}; the specification should be reviewed."
                if rs.p_degeri < 0.05
                else f"The Ramsey RESET test does not reject the hypothesis of a correct functional form {ist}.",
            )
        )

    vf = bul("VIF")
    if vf:
        vif_sayi = sayi(vf.istatistik, 2)
        if vf.sorun_var:
            c.append(
                l(
                    f"Varyans şişirme faktörleri bazı değişkenlerde yüksek çoklu doğrusal bağlantıya işaret etmektedir (en yüksek VIF = {vif_sayi}); ilgili katsayıların standart hataları temkinli yorumlanmalıdır.",
                    f"Variance inflation factors indicate high multicollinearity for some variables (highest VIF = {vif_sayi}); the standard errors of the affected coefficients should be read with caution.",
                )
            )
        elif vf.istatistik > 10:
            c.append(
                l(
                    "Yüksek VIF değerleri yalnızca bir değişken ile kendi kare terimi arasında gözlenmiştir; bu beklenen bir durumdur ve çoklu doğrusal bağlantı sorunu olarak değerlendirilmemiştir.",
                    "High VIF values appear only between a variable and its square; this is expected and is not treated as a multicollinearity problem.",
                )
            )
        else:
            c.append(
                l(
                    f"Varyans şişirme faktörleri 10 eşik değerinin altındadır (en yüksek VIF = {vif_sayi}); çoklu doğrusal bağlantı sorunu görülmemiştir.",
                    f"Variance inflation factors are below the threshold of 10 (highest VIF = {vif_sayi}); multicollinearity is not a concern.",
                )
            )

    dw, bg = bul("Durbin-Watson"), bul("Breusch-Godfrey")
    if dw and bg:
        c.append(
            l(
                f"Otokorelasyon, Durbin-Watson (DW = {sayi(dw.istatistik, 2)}) ve Breusch-Godfrey (LM = {sayi(bg.istatistik, 2)}; {_p(bg.p_degeri)}) testleriyle sınanmıştır.",
                f"Autocorrelation was tested with Durbin-Watson (DW = {sayi(dw.istatistik, 2)}) and Breusch-Godfrey (LM = {sayi(bg.istatistik, 2)}; {_p(bg.p_degeri)}).",
            )
        )
        if bg.p_degeri < 0.05:
            c.append(
                l(
                    "Breusch-Godfrey testi kalıntılarda otokorelasyon bulunduğunu göstermektedir.",
                    "The Breusch-Godfrey test indicates autocorrelation in the residuals.",
                )
            )
            c.append(
                l(
                    "Bu nedenle çıkarımlar otokorelasyona dayanıklı Newey-West (HAC) standart hatalarına dayandırılmıştır."
                    if kov == "HAC"
                    else "Bu durumda standart hatalar güvenilir değildir; modelin Newey-West (HAC) standart hatalarıyla yeniden tahmin edilmesi gerekir.",
                    "Inference is therefore based on autocorrelation-robust Newey-West (HAC) standard errors."
                    if kov == "HAC"
                    else "The standard errors are not reliable in this case; the model should be re-estimated with Newey-West (HAC) standard errors.",
                )
            )
        else:
            c.append(
                l(
                    "Breusch-Godfrey testi kalıntılarda otokorelasyon bulunmadığı hipotezinin reddedilemediğini göstermektedir.",
                    "The Breusch-Godfrey test does not reject the hypothesis of no autocorrelation in the residuals.",
                )
            )

    return [("baslik", l("Varsayım Testleri", "Assumption tests")), ("paragraf", " ".join(c)), ("tablo", tablo)], no + 1


def _yontem(sonuc, vr, adlar: _Adlar) -> tuple[list, list[str]]:
    from . import __version__
    from .regresyon import _log_mu

    y_log = _log_mu(sonuc.bagimli)
    y_ad = y_log or sonuc.bagimli
    kay = ["wooldridge"]
    sag = []
    i = 0
    for t in sonuc._terimler:
        if t.tur == "sabit":
            sag.append("β₀")
        else:
            i += 1
            sag.append(f"β{str(i).translate(_ALT_SIMGE)}·({adlar.satir(t)})")
    sol = f"ln({adlar.yalin(y_ad)})" if y_log else adlar.yalin(y_ad)
    denklem = f"{sol} = " + " + ".join(sag) + " + u"

    c = [
        l(
            f"Bu çalışmada {adlar.metin(y_ad)} değişkeninin belirleyicileri çoklu doğrusal regresyon modeliyle incelenmiş, model En Küçük Kareler (EKK) yöntemiyle tahmin edilmiştir (Wooldridge, 2020). Tahmin edilen model aşağıdaki gibidir:",
            f"The determinants of {adlar.metin(y_ad)} were examined with a multiple linear regression estimated by ordinary least squares (Wooldridge, 2020). The estimated model is:",
        )
    ]
    d = []
    if y_log:
        d.append(
            l(
                "Bağımlı değişkenin doğal logaritması kullanıldığından katsayılar yüzde değişim olarak yorumlanmış; düzey değişkenleri ve kukla değişkenler için yüzde etkiler 100·(e^β − 1) dönüşümüyle hesaplanmıştır.",
                "Because the dependent variable is in natural logarithms, coefficients are read as percentage changes; for level and dummy regressors the percentage effect is 100·(e^β − 1).",
            )
        )
    if any(t.tur == "kukla" for t in sonuc._terimler):
        d.append(
            l(
                "Kategorik değişkenler modele kukla değişkenler aracılığıyla dâhil edilmiş, her kategorik değişkenin bir kategorisi referans grup olarak alınmıştır.",
                "Categorical variables enter as dummies, with one category of each variable omitted as the reference group.",
            )
        )
    kov = sonuc.kovaryans_turu
    if kov == "klasik":
        d.append(
            l(
                "Standart hatalar klasik (sabit varyans varsayımına dayalı) yöntemle hesaplanmıştır.",
                "Standard errors are the classical homoskedastic standard errors.",
            )
        )
    elif kov in ("HAC", "newey"):
        d.append(
            l(
                "Standart hatalar, değişen varyans ve otokorelasyona dayanıklı Newey-West (HAC) tahmincisiyle hesaplanmıştır (Newey ve West, 1987).",
                "Standard errors are Newey-West (HAC) standard errors, robust to heteroskedasticity and autocorrelation (Newey and West, 1987).",
            )
        )
        kay.append("newey")
    elif kov == "kume":
        acik = sonuc.kovaryans_aciklama.split(";")[0]
        govde = acik.removeprefix("kümelenmiş, ")
        if " küme (" in govde and govde.endswith(")"):
            n_kume, kume_ad = govde.split(" küme (", 1)
            en_kume = f"on {kume_ad[:-1]} ({n_kume} clusters)"
        else:
            en_kume = f"({acik})"
        d.append(
            l(
                f"Standart hatalar, gözlemlerin küme içi bağımlılığına ve değişen varyansa dayanıklı kümelenmiş tahminciyle hesaplanmıştır ({acik}); "
                "anlamlılık testlerinde küme sayısının bir eksiği serbestlik derecesi olarak kullanılmıştır (Cameron ve Miller, 2015).",
                f"Standard errors are clustered {en_kume} and are robust to heteroskedasticity and within-cluster dependence; "
                "significance tests use one less than the number of clusters as the degrees of freedom (Cameron and Miller, 2015).",
            )
        )
        kay.append("cameron")
    else:
        atif = "White, 1980" if kov in ("HC0", "HC1") else "MacKinnon ve White, 1985"
        atif_en = "White, 1980" if kov in ("HC0", "HC1") else "MacKinnon and White, 1985"
        d.append(
            l(
                f"Standart hatalar, değişen varyansa dayanıklı {kov} tahmincisiyle hesaplanmıştır ({atif}).",
                f"Standard errors are heteroskedasticity-robust {kov} standard errors ({atif_en}).",
            )
        )
        kay.append("white" if kov in ("HC0", "HC1") else "mackinnon")
    if hasattr(sonuc, "rho"):
        d.append(
            l(
                f"Hata teriminde birinci dereceden otokorelasyon bulunduğundan model, yinelemeli {sonuc.model_turu} genelleştirilmiş en küçük kareler yöntemiyle "
                f"tahmin edilmiştir (Prais ve Winsten, 1954; Cochrane ve Orcutt, 1949); tahmin edilen otokorelasyon katsayısı ρ = {sayi(sonuc.rho, 4)} olup "
                f"Durbin-Watson istatistiği {sayi(sonuc.dw_ilk, 3)} değerinden dönüştürülmüş modelde {sayi(sonuc.dw_donusmus, 3)} değerine gelmiştir.",
                f"Because the errors are first-order autocorrelated, the model was estimated by iterated {sonuc.model_turu} GLS "
                f"(Prais and Winsten, 1954; Cochrane and Orcutt, 1949). The estimated autocorrelation is ρ = {sayi(sonuc.rho, 4)}; "
                f"the Durbin-Watson statistic moved from {sayi(sonuc.dw_ilk, 3)} to {sayi(sonuc.dw_donusmus, 3)} in the transformed model.",
            )
        )
        kay += ["prais", "corc"]
    agirlik = getattr(sonuc, "agirlik", None)
    if agirlik:
        tur_ad = {"aw": "analitik ağırlıklar", "fw": "frekans ağırlıkları", "pw": "örneklem (olasılık) ağırlıkları"}[agirlik[0]]
        tur_en = {"aw": "analytic weights", "fw": "frequency weights", "pw": "sampling (probability) weights"}[agirlik[0]]
        d.append(
            l(
                f"Model, {adlar.metin(agirlik[1])} değişkeni {tur_ad} olarak kullanılarak ağırlıklı en küçük kareler yöntemiyle tahmin edilmiştir.",
                f"The model was estimated by weighted least squares, using {adlar.metin(agirlik[1])} as {tur_en}.",
            )
        )
    if vr is None:
        d.append(
            l(
                f"Hesaplar ekonometri-tr {__version__} ile yapılmıştır.",
                f"The calculations were done with ekonometri-tr {__version__}.",
            )
        )
        bloklar = [("baslik", l("Yöntem", "Method")), ("paragraf", c[0]), ("denklem", denklem), ("paragraf", " ".join(d))]
        return bloklar, list(dict.fromkeys(kay))
    t = (
        "Modelin varsayımları; değişen varyans için Breusch-Pagan (Breusch ve Pagan, 1979; Koenker, 1981) ve White (1980) testleri, "
        "normallik için Jarque-Bera testi (Jarque ve Bera, 1987), fonksiyonel biçim için RESET testi (Ramsey, 1969)"
    )
    kay += ["bp", "koenker", "white", "jb", "ramsey"]
    if any(x.ad == "VIF" for x in vr.testler):
        t += ", çoklu doğrusal bağlantı için varyans şişirme faktörleri (VIF)"
    if any(x.ad.startswith("Durbin") for x in vr.testler):
        t += ", otokorelasyon için Durbin-Watson (Durbin ve Watson, 1950) ve Breusch-Godfrey (Breusch, 1978; Godfrey, 1978) testleri"
        kay += ["dw", "breusch", "godfrey"]
    if DIL.get() == "en":
        parcalar = [
            "heteroskedasticity (Breusch and Pagan, 1979; Koenker, 1981; White, 1980)",
            "normality (Jarque and Bera, 1987)",
            "functional form (Ramsey, 1969)",
        ]
        if any(x.ad == "VIF" for x in vr.testler):
            parcalar.append("multicollinearity (VIF)")
        if any(x.ad.startswith("Durbin") for x in vr.testler):
            parcalar.append("autocorrelation (Durbin and Watson, 1950; Breusch, 1978; Godfrey, 1978)")
        d.append("The assumptions were tested with " + ", ".join(parcalar) + ".")
    else:
        d.append(t + " ile sınanmıştır.")
    d.append(
        l(
            f"Hesaplar ekonometri-tr {__version__} ile yapılmıştır.",
            f"The calculations were done with ekonometri-tr {__version__}.",
        )
    )
    bloklar = [("baslik", l("Yöntem", "Method")), ("paragraf", c[0]), ("denklem", denklem), ("paragraf", " ".join(d))]
    return bloklar, list(dict.fromkeys(kay))


# --------------------------------------------------------------------------- ana fonksiyon
def rapor_olustur(
    sonuc,
    baslik: str = "Bulgular",
    etiketler: dict[str, str] | None = None,
    tablo_no: int = 1,
    yontem: bool = True,
    kaynakca: bool = True,
    zaman_serisi: bool | None = None,
    dil: str = "tr",
    bicim: str = "apa",
) -> Rapor:
    from .regresyon import EkonometriHatasi

    if dil not in ("tr", "en"):
        raise EkonometriHatasi("dil 'tr' ya da 'en' olmalı.")
    if dil == "en" and baslik == "Bulgular":
        baslik = "Findings"
    if bicim not in ("apa", "izgara"):
        raise EkonometriHatasi("bicim 'apa' ya da 'izgara' olmalı.")
    token = DIL.set(dil)
    try:
        return _rapor_olustur(sonuc, baslik, etiketler, tablo_no, yontem, kaynakca, zaman_serisi, dil, bicim)
    finally:
        DIL.reset(token)


def _rapor_olustur(sonuc, baslik, etiketler, tablo_no, yontem, kaynakca, zaman_serisi, dil, bicim) -> Rapor:
    adlar = _Adlar({**getattr(sonuc, "_etiketler", {}), **(etiketler or {})})
    vr = None if (getattr(sonuc, "agirlik", None) or hasattr(sonuc, "rho")) else sonuc.varsayimlar(zaman_serisi=zaman_serisi)
    r = Rapor(baslik, bicim=bicim, dil=dil)
    y_blok, kaynaklar = _yontem(sonuc, vr, adlar)
    if yontem:
        r.bloklar += y_blok
    no = tablo_no
    for bolum in (_tanimlayici, _regresyon):
        b, no = bolum(sonuc, adlar, no)
        r.bloklar += b
    if vr is not None:
        b, no = _varsayim(sonuc, vr, no)
        r.bloklar += b
    r.bloklar.append(
        (
            "paragraf",
            l(
                "Bulgular değişkenler arasındaki ilişkiyi göstermektedir; nedensel bir yorum, araştırma tasarımının buna izin verdiği ölçüde yapılabilir.",
                "The findings describe associations among the variables; a causal reading is warranted only to the extent that the research design supports it.",
            ),
        )
    )
    if yontem and kaynakca:
        r.bloklar += [("baslik", l("Kaynakça", "References")), ("liste", sorted(KAYNAKLAR[k] for k in kaynaklar))]
    return r


__all__ = ["Rapor", "Tablo", "rapor_olustur"]


def test_raporu(
    baslik: str, paragraflar: list[str], tablo: Tablo, kaynaklar: list[str], dosya=None, dil: str = "tr", bicim: str = "apa"
) -> Rapor:
    """Tek tablolu test raporu (yöntem + bulgu paragrafı, tablo, kaynakça). Çağıran DIL ayarını yapmış olmalı."""
    r = Rapor(
        baslik,
        [
            ("paragraf", " ".join(paragraflar)),
            ("tablo", tablo),
            ("baslik", "References" if dil == "en" else "Kaynakça"),
            ("liste", sorted(set(kaynaklar))),
        ],
        bicim=bicim,
        dil=dil,
    )
    if dosya is not None:
        r.kaydet(dosya)
    return r
