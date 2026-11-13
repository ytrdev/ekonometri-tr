"""Stata komutlarıyla çalış: Stata bilen kullanıcı aynı komutları yazar, sonuçlar Stata ile aynıdır, çıktılar Türkçedir.

>>> s = ek.Stata()
>>> s("use anket.dta")                       # ya da: s = ek.Stata(veri)
>>> s("gen lnw = ln(ucret)")
>>> s("regress lnw egitim c.tecrube##c.tecrube i.cinsiyet, vce(robust)")
>>> s("estat hettest")
>>> s("rapor bulgular.docx")                 # Türkçe tez raporu (Stata'da olmayan ek komut)
>>> s.do("analiz.do")                        # do dosyasını çalıştır

Desteklenen komutlar için: ``print(ek.Stata.KOMUTLAR)``. Yalnızca klasik ekonometri komutları desteklenir;
desteklenmeyen bir komut yazıldığında Türkçe hata mesajı verilir.
"""

from __future__ import annotations

import ast
import fnmatch
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ._yardimci import p_yaz, sayi
from .regresyon import EkonometriHatasi

KOMUTLAR = """
Veri         : use, sysuse, import excel, import delimited, save, describe, list, count
Değişken     : generate, replace, egen (mean/total/sum/sd/min/max/count/median, by()), drop, keep, rename, order,
               label variable / define / values, encode, decode, destring, recode, duplicates, sort, tsset, xtset
Birleştirme  : merge 1:1 / m:1 / 1:m, append, reshape long / wide, collapse
Tanımlayıcı  : summarize [, detail], tabulate (tek / iki yönlü), correlate, pwcorr [, sig]
Tahmin       : regress, newey, prais, logit, probit, ologit, oprobit, mlogit [, baseoutcome(#)],
               poisson / nbreg [, exposure() irr], tobit [, ll(#) ul(#)], xtreg (fe / re), xtabond, ivregress 2sls
Zaman serisi : dfuller, zandrews [, break(intercept|trend|both) lagmethod(aic|bic|ttest) lags(#) maxlags(#) trim(#)],
               var [, lags(1/#)], varsoc [, maxlag(#)], varstable, vargranger, irf graph|table oirf [, step(#)],
               ardl y x [, lags(#) maxlags(#) aic] + estat ectest, vecrank [, lags(#) trend()]
Panel        : xtunitroot llc | ips | fisher (dfuller) [, lags(# | aic #) trend demean], pescadf [, lags(#) trend],
               xtcd (değişken), xtcsd (xtreg sonrası), xttest0, hausman
Testler      : ttest, oneway, tabulate …, chi2 (V, lrchi2)
Sonrası      : lincom, predict, margins [faktör] [, at(x=(1 2 3)) dydx()], test, estimates store, esttab [using dosya.docx|.tex|.xlsx],
               estat (hettest, imtest white, ovtest, vif, dwatson, bgodfrey, firststage, endogenous, overid,
               classification, gof [, group(10)], abond, sargan, ectest), display
Grafik       : rvfplot, qnorm, histogram [, normal], scatter, twoway (scatter)(lfit), tsline, coefplot, marginsplot, lroc,
               irf graph, graph export dosya.png [, replace]
Türkçe ekler : yorum, varsayimlar, rapor "dosya.docx", todayamamoto y x [, dmax(1) lags(#) maxlag(#) sbic trend]
Seçenekler   : vce(robust | cluster değişken | hc2 | hc3), robust, noconstant, small, fe, re, lags(#), trend, detail
Değişken dili: i.x, ib#.x, ib(first|last).x, c.x, x#z, x##z, L. L#. D. F., x*, x1-x5, if, in
""".strip()

_KISALTMA = {
    "reg": "regress",
    "regr": "regress",
    "regre": "regress",
    "regres": "regress",
    "su": "summarize",
    "sum": "summarize",
    "summ": "summarize",
    "summa": "summarize",
    "summar": "summarize",
    "summari": "summarize",
    "summariz": "summarize",
    "ta": "tabulate",
    "tab": "tabulate",
    "tabu": "tabulate",
    "tabul": "tabulate",
    "tabula": "tabulate",
    "tabulat": "tabulate",
    "cor": "correlate",
    "corr": "correlate",
    "corre": "correlate",
    "correl": "correlate",
    "correla": "correlate",
    "correlat": "correlate",
    "g": "generate",
    "gen": "generate",
    "gene": "generate",
    "gener": "generate",
    "genera": "generate",
    "generat": "generate",
    "d": "describe",
    "de": "describe",
    "des": "describe",
    "desc": "describe",
    "descr": "describe",
    "descri": "describe",
    "describ": "describe",
    "l": "list",
    "li": "list",
    "lis": "list",
    "ren": "rename",
    "rena": "rename",
    "renam": "rename",
    "ivreg": "ivregress",
    "ivregr": "ivregress",
    "ivregre": "ivregress",
    "ivregres": "ivregress",
    "est": "estimates",
    "esti": "estimates",
    "estim": "estimates",
    "estima": "estimates",
    "estimat": "estimates",
    "estimate": "estimates",
    "di": "display",
    "dis": "display",
    "disp": "display",
    "displ": "display",
    "displa": "display",
    "qui": "quietly",
    "quie": "quietly",
    "quiet": "quietly",
    "quietl": "quietly",
    "lab": "label",
    "labe": "label",
    "pred": "predict",
    "predi": "predict",
    "predic": "predict",
    "cou": "count",
    "coun": "count",
}


def _hata(m: str) -> EkonometriHatasi:
    return EkonometriHatasi(m)


def _tsayi(x) -> str:
    """Tablo sayısı: bilimsel gösterim olmadan, Türkçe ayırıcılarla (büyük değerlerde 2, küçüklerde 4 ondalık)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "."
    basamak = 2 if abs(x) >= 1000 else 4
    return f"{x:,.{basamak}f}".replace(",", "_").replace(".", ",").replace("_", ".")


# =========================================================================== metin ayrıştırma yardımcıları
def _dis_dizin(metin: str, hedef: str) -> int:
    """Tırnak ve parantez dışındaki ilk 'hedef' karakterinin yeri (yoksa -1)."""
    derinlik, tirnak = 0, None
    for i, c in enumerate(metin):
        if tirnak:
            if c == tirnak:
                tirnak = None
            continue
        if c in "\"'":
            tirnak = c
        elif c in "([":
            derinlik += 1
        elif c in ")]":
            derinlik -= 1
        elif c == hedef and derinlik == 0:
            return i
    return -1


def _kelime_dis(metin: str, kelime: str) -> int:
    """Tırnak ve parantez dışında, ayrı kelime olarak geçen 'kelime'nin yeri."""
    derinlik, tirnak = 0, None
    for i, c in enumerate(metin):
        if tirnak:
            if c == tirnak:
                tirnak = None
            continue
        if c in "\"'":
            tirnak = c
        elif c in "([":
            derinlik += 1
        elif c in ")]":
            derinlik -= 1
        elif derinlik == 0 and metin.startswith(kelime, i):
            once = metin[i - 1] if i else " "
            sonra = metin[i + len(kelime)] if i + len(kelime) < len(metin) else " "
            if once.isspace() and (sonra.isspace() or sonra == "("):
                return i
    return -1


def _secenekler(metin: str) -> dict[str, str | bool]:
    """'vce(cluster il) fe lags(2) sheet("A")' -> {'vce': 'cluster il', 'fe': True, 'lags': '2', 'sheet': 'A'}"""
    sonuc: dict[str, str | bool] = {}
    i, n = 0, len(metin)
    while i < n:
        while i < n and metin[i].isspace():
            i += 1
        j = i
        while j < n and (metin[j].isalnum() or metin[j] in "_"):
            j += 1
        ad = metin[i:j].lower()
        if not ad:
            i += 1
            continue
        if j < n and metin[j] == "(":
            derinlik, k = 0, j
            while k < n:
                if metin[k] == "(":
                    derinlik += 1
                elif metin[k] == ")":
                    derinlik -= 1
                    if derinlik == 0:
                        break
                k += 1
            sonuc[ad] = metin[j + 1 : k].strip().strip('"').strip("'")
            i = k + 1
        else:
            sonuc[ad] = True
            i = j
    return sonuc


def _secenek(sec: dict, ad: str, en_kisa: int | None = None):
    """Stata gibi kısaltılmış seçenek adlarını tanı: noc / nocons / noconstant."""
    en_kisa = en_kisa or len(ad)
    for k, v in sec.items():
        if len(k) >= en_kisa and ad.startswith(k):
            return v
    return None


def _sayi_listesi(metin: str) -> list[float]:
    """Stata numlist: "8 12 16", "0(5)20", "1/4", "10 20(10)50" → sayı listesi."""
    sonuc: list[float] = []
    for parca in metin.replace(",", " ").split():
        m = re.fullmatch(r"(-?[\d.]+)\((-?[\d.]+)\)(-?[\d.]+)", parca)
        m2 = re.fullmatch(r"(-?\d+)/(-?\d+)", parca)
        try:
            if m:
                a, h, b = (float(x) for x in m.groups())
                if h == 0:
                    raise _hata(f"Adım sıfır olamaz: {parca}")
                sonuc += list(np.arange(a, b + h / 2, h))
            elif m2:
                a, b = (int(x) for x in m2.groups())
                sonuc += list(range(a, b + 1))
            else:
                sonuc.append(float(parca))
        except ValueError:
            raise _hata(f"Sayı listesi okunamadı: {metin!r}") from None
    return [int(x) if float(x).is_integer() else float(x) for x in sonuc]


def _yorumlari_temizle(metin: str) -> list[str]:
    """Stata yorumlarını (*, //, /* */) atar, /// ile bölünmüş satırları birleştirir."""
    metin = re.sub(r"/\*.*?\*/", " ", metin, flags=re.S)
    satirlar, birikim = [], ""
    for ham in metin.splitlines():
        satir = ham
        # tırnak dışındaki // ve /// yorumları
        tirnak, kes = None, None
        for i, c in enumerate(satir):
            if tirnak:
                if c == tirnak:
                    tirnak = None
                continue
            if c in '"':
                tirnak = c
            elif satir.startswith("//", i) and (i == 0 or satir[i - 1].isspace()):
                kes = i
                break
        devam = False
        if kes is not None:
            devam = satir.startswith("///", kes)
            satir = satir[:kes]
        if not birikim and satir.strip().startswith("*"):
            continue
        birikim += " " + satir.strip()
        if not devam:
            if birikim.strip():
                satirlar.append(birikim.strip())
            birikim = ""
    if birikim.strip():
        satirlar.append(birikim.strip())
    return satirlar


# =========================================================================== Stata ifadeleri (gen / replace / if / display)
_ISLEVLER = {
    "ln",
    "log",
    "log10",
    "exp",
    "sqrt",
    "abs",
    "round",
    "floor",
    "ceil",
    "int",
    "missing",
    "inlist",
    "inrange",
    "cond",
    "max",
    "min",
    "normal",
    "invnormal",
}


def _stata_tarih(yil: str, tur: str, deger: str) -> int:
    """Stata %tq / %tm / %th sayısı: 1960q1 = 0 (2010q1 = 200)."""
    carpan = {"q": 4, "m": 12, "h": 2}[tur.lower()]
    return (int(yil) - 1960) * carpan + int(deger) - 1


def _stata_zaman_serisi(x: pd.Series) -> pd.Series:
    """Dönem (Period) sütununu Stata'nın sayısal tarih değerine çevir: %ty yıl, %tq/%tm/%th 1960'tan beri geçen dönem."""
    frek = x.dtype.freq
    if frek.name[0].upper() in ("Y", "A"):
        return x.dt.year.astype(float)
    baslangic = pd.Period("1960-01-01", freq=frek).ordinal
    return pd.Series([np.nan if pd.isna(p) else float(p.ordinal - baslangic) for p in x], index=x.index)


def _python_ifadesi(ifade: str) -> str:
    """Stata ifadesini Python sözdizimine çevirir (yalnızca tırnak dışında)."""
    parcalar = re.split(r"(\"[^\"]*\")", ifade)
    out = []
    for i, p in enumerate(parcalar):
        if i % 2 == 1:
            out.append(p)
            continue
        p = p.replace("~=", "!=")
        p = re.sub(r"\b(?:t[qmh])\(\s*(\d{4}[qmh]\d{1,2})\s*\)", r"\1", p, flags=re.I)  # tq(2010q1) → 2010q1
        p = re.sub(r"\b(\d{4})([qmh])(\d{1,2})\b", lambda m: str(_stata_tarih(*m.groups())), p, flags=re.I)
        p = re.sub(r"!(?!=)", " not ", p)
        p = p.replace("^", "**")
        p = re.sub(r"(?<![\w.])\.(?![\w.])", " _EKSIK ", p)  # Stata eksik değeri '.'
        p = re.sub(r"(?<![&|])&(?![&|])", " and ", p)
        p = re.sub(r"(?<![&|])\|(?![&|])", " or ", p)
        out.append(p)
    return "".join(out)


class _Degerlendirici:
    """Stata ifadesini güvenli biçimde (eval kullanmadan) değerlendirir. Eksik değer Stata'daki gibi +∞ sayılır."""

    def __init__(self, veri: pd.DataFrame, deger_etiketleri: dict | None = None, son_model=None):
        self.v = veri
        self.de = deger_etiketleri or {}
        self.son = son_model
        self.n = len(veri)

    def __call__(self, ifade: str):
        try:
            agac = ast.parse(_python_ifadesi(ifade).strip(), mode="eval")
        except SyntaxError:
            raise _hata(f"İfade anlaşılamadı: {ifade!r}") from None
        return self._d(agac.body)

    # --- yardımcılar
    def _seri(self, x):
        if isinstance(x, pd.Series):
            return x
        return pd.Series([x] * self.n, index=self.v.index)

    @staticmethod
    def _dogru(x):
        """Stata mantığı: sıfır olmayan (eksik dahil) değer doğrudur."""
        if isinstance(x, pd.Series):
            if x.dtype == bool:
                return x
            return x.fillna(1).astype(float) != 0
        return bool(x != 0) if x == x else True

    @staticmethod
    def _sonsuz(x):
        if isinstance(x, pd.Series) and pd.api.types.is_numeric_dtype(x) and x.dtype != bool:
            return x.astype(float).fillna(np.inf)
        if isinstance(x, float) and math.isnan(x):
            return np.inf
        return x

    def _d(self, d):
        if isinstance(d, ast.Constant):
            return d.value
        if isinstance(d, ast.Name):
            ad = d.id
            if ad == "_EKSIK":
                return np.nan
            if ad == "_n":
                return pd.Series(np.arange(1, self.n + 1, dtype=float), index=self.v.index)
            if ad == "_N":
                return float(self.n)
            if ad == "_pi":
                return math.pi
            if ad in self.v.columns:
                x = self.v[ad]
                return _stata_zaman_serisi(x) if isinstance(x.dtype, pd.PeriodDtype) else x
            raise _hata(f"Değişken bulunamadı: {ad!r}")
        if isinstance(d, ast.BinOp):
            a, b = self._d(d.left), self._d(d.right)
            islem = {
                ast.Add: lambda x, y: x + y,
                ast.Sub: lambda x, y: x - y,
                ast.Mult: lambda x, y: x * y,
                ast.Div: lambda x, y: x / y,
                ast.Pow: lambda x, y: x**y,
            }
            for t, f in islem.items():
                if isinstance(d.op, t):
                    with np.errstate(all="ignore"):
                        sonuc = f(a, b)
                    if isinstance(d.op, ast.Div):  # Stata: sıfıra bölme eksik değer verir
                        sonuc = (
                            sonuc.replace([np.inf, -np.inf], np.nan)
                            if isinstance(sonuc, pd.Series)
                            else (np.nan if not np.isfinite(sonuc) else sonuc)
                        )
                    return sonuc
            raise _hata("Desteklenmeyen işlem.")
        if isinstance(d, ast.UnaryOp):
            a = self._d(d.operand)
            if isinstance(d.op, ast.USub):
                return -a
            if isinstance(d.op, ast.UAdd):
                return a
            if isinstance(d.op, ast.Not):
                return ~self._dogru(self._seri(a))
        if isinstance(d, ast.BoolOp):
            degerler = [self._dogru(self._seri(self._d(x))) for x in d.values]
            sonuc = degerler[0]
            for x in degerler[1:]:
                sonuc = (sonuc & x) if isinstance(d.op, ast.And) else (sonuc | x)
            return sonuc
        if isinstance(d, ast.Compare):
            sol_dugum, sol = d.left, self._d(d.left)
            sonuc = None
            for op, sag_dugum in zip(d.ops, d.comparators):
                sag = self._d(sag_dugum)
                sol2, sag2 = self._etiket_eslestir(sol_dugum, sol, sag), self._etiket_eslestir(sag_dugum, sag, sol)
                a, b = self._sonsuz(sol2), self._sonsuz(sag2)
                f = {
                    ast.Eq: lambda x, y: x == y,
                    ast.NotEq: lambda x, y: x != y,
                    ast.Lt: lambda x, y: x < y,
                    ast.LtE: lambda x, y: x <= y,
                    ast.Gt: lambda x, y: x > y,
                    ast.GtE: lambda x, y: x >= y,
                }[type(op)]
                r = self._seri(f(a, b))
                r = r.fillna(False).astype(bool) if r.dtype != bool else r
                sonuc = r if sonuc is None else (sonuc & r)
                sol_dugum, sol = sag_dugum, sag
            return sonuc
        if isinstance(d, ast.Call) and isinstance(d.func, ast.Name):
            return self._islev(d.func.id, [self._d(a) for a in d.args], d.args)
        if isinstance(d, ast.Subscript) and isinstance(d.value, ast.Name):
            return self._alt_simge(d)
        raise _hata("İfade desteklenmiyor (yalnızca aritmetik, karşılaştırma, mantık işlemleri ve temel Stata işlevleri).")

    def _etiket_eslestir(self, dugum, deger, karsi):
        """Değer etiketli değişkende 'cinsiyet == 2' gibi kod karşılaştırmasını etikete çevir."""
        if isinstance(dugum, ast.Constant) and isinstance(deger, (int, float)) and isinstance(karsi, pd.Series):
            ad = karsi.name
            if ad in self.de and deger in self.de[ad]:
                return self.de[ad][deger]
        return deger

    def _islev(self, ad: str, a: list, dugumler):
        with np.errstate(all="ignore"):
            if ad in ("ln", "log"):
                x = self._seri(a[0]).astype(float)
                return np.log(x.where(x > 0))
            if ad == "log10":
                x = self._seri(a[0]).astype(float)
                return np.log10(x.where(x > 0))
            if ad == "exp":
                return np.exp(a[0])
            if ad == "sqrt":
                x = self._seri(a[0]).astype(float)
                return np.sqrt(x.where(x >= 0))
            if ad == "abs":
                return np.abs(a[0])
            if ad == "floor":
                return np.floor(a[0])
            if ad == "ceil":
                return np.ceil(a[0])
            if ad == "int":
                return np.trunc(a[0])
            if ad == "round":
                y = a[1] if len(a) > 1 else 1.0
                return (
                    np.floor(np.asarray(a[0], dtype=float) / y + 0.5) * y
                    if not isinstance(a[0], pd.Series)
                    else np.floor(a[0] / y + 0.5) * y
                )
            if ad == "missing":
                sonuc = None
                for x in a:
                    sx = self._seri(x)
                    metin_mi = not pd.api.types.is_numeric_dtype(sx) and not isinstance(sx.dtype, pd.CategoricalDtype)
                    m = (sx.isna() | (sx.astype(str).str.strip() == "")) if metin_mi else sx.isna()
                    sonuc = m if sonuc is None else (sonuc | m)
                return sonuc
            if ad == "inlist":
                x = a[0]
                adaylar = [self._etiket_eslestir(dg, dv, x) for dg, dv in zip(dugumler[1:], a[1:])]
                return self._seri(x).isin(adaylar)
            if ad == "inrange":
                x = self._sonsuz(self._seri(a[0]))
                return (x >= a[1]) & (x <= a[2])
            if ad == "cond":
                kosul = self._dogru(self._seri(a[0]))
                return pd.Series(np.where(kosul, self._seri(a[1]), self._seri(a[2])), index=self.v.index)
            if ad in ("max", "min"):
                tablo = pd.concat([self._seri(x).astype(float) for x in a], axis=1)
                return tablo.max(axis=1) if ad == "max" else tablo.min(axis=1)
            if ad == "normal":
                return pd.Series(stats.norm.cdf(self._seri(a[0]).astype(float)), index=self.v.index)
            if ad == "invnormal":
                return pd.Series(stats.norm.ppf(self._seri(a[0]).astype(float)), index=self.v.index)
        raise _hata(f"Desteklenmeyen işlev: {ad}(). Desteklenenler: {', '.join(sorted(_ISLEVLER))}")

    def _alt_simge(self, d):
        ad = d.value.id
        icerik = d.slice
        if ad in ("_b", "_se"):
            if self.son is None:
                raise _hata("_b[] / _se[] için önce bir model tahmin et.")
            terim = ast.unparse(icerik).replace(" ", "")
            terim = "Intercept" if terim == "_cons" else terim
            seri = self.son.katsayilar if ad == "_b" else self.son.standart_hatalar
            if terim not in seri.index:
                raise _hata(f"Modelde {terim!r} terimi yok. Terimler: {list(seri.index)}")
            return float(seri[terim])
        if ad in self.v.columns:  # x[_n-1] gibi açık alt simge
            konum = self._seri(self._d(icerik)).astype(float)
            degerler = self.v[ad].to_numpy()
            out = [degerler[int(k) - 1] if k == k and 1 <= k <= self.n else np.nan for k in konum]
            return pd.Series(out, index=self.v.index)
        raise _hata(f"Alt simge anlaşılamadı: {ad}[...]")


# =========================================================================== oturum
class Stata:
    """Stata komutlarını çalıştıran oturum. Bellekte bir veri seti ve son tahmin tutulur (Stata'daki gibi)."""

    KOMUTLAR = KOMUTLAR

    def __init__(self, veri: pd.DataFrame | None = None, yazdir: bool = True):
        self.veri = None if veri is None else self._kopya(veri)
        self.son = None  # son tahmin (e())
        self.saklanan: dict[str, object] = {}
        self.yazdir = yazdir
        self._sessiz = False
        self._donusumler: dict[str, str] = {}  # gen ile oluşturulan log / kare değişkenler: 'lnw' -> 'np.log(ucret)'
        self._son_sekil = None  # son çizilen grafik (graph export için)
        self._son_marj = None  # son margins sonucu (marginsplot için)

    # ------------------------------------------------------------------ genel
    @staticmethod
    def _kopya(veri: pd.DataFrame) -> pd.DataFrame:
        k = veri.copy()
        k.attrs = {key: (dict(v) if isinstance(v, dict) else v) for key, v in veri.attrs.items()}
        return k

    def _yaz(self, x) -> None:
        if self.yazdir and not self._sessiz and x is not None:
            print(x)

    def __call__(self, metin: str):
        sonuc = None
        for komut in _yorumlari_temizle(metin):
            sonuc = self._calistir(komut)
        return sonuc

    def do(self, dosya: str | Path):
        """Stata do dosyasını çalıştırır (yorumlar ve /// satır devamı desteklenir)."""
        yol = Path(dosya)
        if not yol.exists():
            raise _hata(f"Do dosyası bulunamadı: {yol}")
        return self(yol.read_text(encoding="utf-8"))

    def _veri_gerekli(self) -> pd.DataFrame:
        if self.veri is None:
            raise _hata("Bellekte veri yok. Önce: use dosya.dta / import excel / sysuse ucret")
        return self.veri

    def _calistir(self, komut: str):
        komut = komut.strip().rstrip(";")
        if not komut:
            return None
        mk = re.match(r"^(\w+)(.*)$", komut, re.S)
        if not mk:
            raise _hata(f"Komut anlaşılamadı: {komut!r}")
        ilk, kalan = mk.group(1), mk.group(2)
        ad = _KISALTMA.get(ilk.lower(), ilk.lower())
        if ad == "quietly":
            onceki, self._sessiz = self._sessiz, True
            try:
                return self._calistir(kalan.lstrip(": "))
            finally:
                self._sessiz = onceki
        if ad in ("by", "bys", "bysort"):
            return self._by(kalan)
        if ad == "capture":
            try:
                return self._calistir(kalan)
            except EkonometriHatasi:
                return None
        islev = getattr(self, f"_k_{ad}", None)
        if islev is None:
            raise _hata(f"'{ilk}' komutu desteklenmiyor. Desteklenen komutlar:\n{KOMUTLAR}")
        return islev(kalan.strip())

    # ------------------------------------------------------------------ ayrıştırma
    _AGIRLIK = re.compile(r"\[\s*(aw|aweight|fw|fweight|pw|pweight|iw|iweight)\s*=\s*([^\]]+?)\s*\]", re.I)

    def _parcala(self, metin: str) -> tuple[str, str | None, str | None, dict]:
        """'varlist [ağırlık] if ... in ..., seçenekler' -> (ana, if, in, seçenekler); ağırlık sec['_agirlik'] = (tür, değişken)"""
        agirlik = None
        m = self._AGIRLIK.search(metin)
        if m:
            tur = m.group(1).lower()[:1] + "w"
            if tur == "iw":
                raise _hata("iweight desteklenmez; aw, fw ya da pw kullan.")
            agirlik = (tur, m.group(2).strip())
            metin = metin[: m.start()] + " " + metin[m.end() :]
        virgul = _dis_dizin(metin, ",")
        ana, sec = (metin[:virgul], metin[virgul + 1 :]) if virgul >= 0 else (metin, "")
        ana = " " + ana + " "
        kosul = aralik = None
        i_in = _kelime_dis(ana, "in")
        i_if = _kelime_dis(ana, "if")
        if i_in >= 0 and (i_if < 0 or i_in < i_if):
            son = i_if if i_if > i_in else len(ana)
            aralik = ana[i_in + 2 : son].strip()
            ana = ana[:i_in] + ana[son:]
            i_if = _kelime_dis(ana, "if")
        if i_if >= 0:
            kosul = ana[i_if + 2 :].strip()
            ana = ana[:i_if]
            i_in = _kelime_dis(" " + kosul + " ", "in")
            if i_in >= 0 and re.match(r"^\s*-?\w+\s*/\s*-?\w+\s*$", (" " + kosul + " ")[i_in + 2 :]):
                aralik = (" " + kosul + " ")[i_in + 2 :].strip()
                kosul = (" " + kosul + " ")[:i_in].strip()
        secenekler = _secenekler(sec)
        if agirlik:
            secenekler["_agirlik"] = (agirlik[0], self._ad_coz(agirlik[1]))
        return ana.strip(), kosul, aralik, secenekler

    def _ornek(self, kosul: str | None, aralik: str | None) -> pd.Series:
        """if / in koşulunu sağlayan satırlar (mantıksal seri)."""
        v = self._veri_gerekli()
        m = pd.Series(True, index=v.index)
        if aralik:
            a, _, b = aralik.partition("/")

            def konum(x):
                x = x.strip().lower()
                return {"f": 1, "l": len(v)}.get(x) or (len(v) + int(x) + 1 if int(x) < 0 else int(x))

            bas, son = konum(a), konum(b or a)
            m &= pd.Series((np.arange(1, len(v) + 1) >= bas) & (np.arange(1, len(v) + 1) <= son), index=v.index)
        if kosul:
            m &= self._degerlendir(kosul)
        return m

    def _degerlendir(self, ifade: str, veri: pd.DataFrame | None = None):
        v = self._veri_gerekli() if veri is None else veri
        sonuc = _Degerlendirici(v, v.attrs.get("deger_etiketleri", {}), self.son)(ifade)
        if isinstance(sonuc, pd.Series) and sonuc.dtype == bool:
            return sonuc
        return sonuc

    def _degiskenler(self, metin: str) -> list[str]:
        """Değişken listesi: joker (x*), aralık (x1-x5) ve tam adlar."""
        v = self._veri_gerekli()
        cikti = []
        for t in metin.split():
            if "*" in t or "?" in t:
                eslesen = [c for c in v.columns if fnmatch.fnmatchcase(c, t)]
                if not eslesen:
                    raise _hata(f"{t!r} ile eşleşen değişken yok.")
                cikti += eslesen
            elif "-" in t and t.split("-")[0] in v.columns and t.split("-")[1] in v.columns:
                a, b = t.split("-")
                sutunlar = list(v.columns)
                cikti += sutunlar[sutunlar.index(a) : sutunlar.index(b) + 1]
            else:
                cikti.append(self._ad_coz(t))
        return cikti

    def _ad_coz(self, ad: str) -> str:
        v = self._veri_gerekli()
        if ad in v.columns:
            return ad
        adaylar = [c for c in v.columns if c.startswith(ad)]  # Stata değişken adı kısaltması
        if len(adaylar) == 1:
            return adaylar[0]
        if len(adaylar) > 1:
            raise _hata(f"'{ad}' kısaltması belirsiz: {adaylar}")
        raise _hata(f"Değişken bulunamadı: {ad!r}. Mevcut değişkenler: {list(v.columns)}")

    # ------------------------------------------------------------------ değişken dili → formül
    _ZS = re.compile(r"^((?:[LDF]\d*)+)\.(.+)$", re.I)
    _ZS_ARALIK = re.compile(r"^([LF])\((\d+)/(\d+)\)\.(.+)$", re.I)

    def _zaman_sutunu(self, d: pd.DataFrame, islemler: str, ad: str) -> str:
        """L., L2., D., LD., F. işlemleriyle yeni sütun üretir (tüm veride hesaplanır, Stata gibi)."""
        from .girdi import _zaman_islemi

        temel = self._ad_coz(ad)
        anahtar = f"{islemler.upper()}_{temel}"
        if anahtar in d.columns:
            return anahtar
        seri = self.veri[temel].astype(float)
        for op, sayi_ in reversed(re.findall(r"([LDF])(\d*)", islemler.upper())):
            k = int(sayi_ or 1)
            gecici = self.veri.drop(columns=[temel]).assign(**{temel: seri})
            gecici.attrs = dict(self.veri.attrs)
            if op == "L":
                seri = _zaman_islemi(gecici, temel, k)
            elif op == "F":
                seri = _zaman_islemi(gecici, temel, -k)
            else:
                for _ in range(k):
                    gecici[temel] = seri
                    seri = seri - _zaman_islemi(gecici, temel, 1)
        d[anahtar] = seri.loc[d.index].values
        etiket = f"{islemler.upper()}.{temel}"
        d.attrs.setdefault("etiketler", {})[anahtar] = etiket
        return anahtar

    def _terim(self, t: str, d: pd.DataFrame) -> list[str]:
        """Stata faktör / zaman serisi değişken dilini patsy terimlerine çevirir."""
        if "##" in t:
            parcalar = t.split("##")
            from itertools import combinations

            out = []
            for r in range(1, len(parcalar) + 1):
                for grup in combinations(parcalar, r):
                    out += self._terim("#".join(grup), d)
            return list(dict.fromkeys(out))
        if "#" in t:
            parcalar = [self._tekil(p, d, etkilesim=True) for p in t.split("#")]
            surekli = [p for p in parcalar if not p.startswith("C(")]
            if len(set(surekli)) == 1 and len(surekli) == len(parcalar):
                return [f"I({surekli[0]}**{len(surekli)})"]  # c.x#c.x → kare terim (yorumda dönüm noktası)
            return [":".join(parcalar)]
        m = self._ZS_ARALIK.match(t)
        if m:
            op, a, b, ad = m.groups()
            return [self._tekil(f"{op}{k}.{ad}" if k else ad, d) for k in range(int(a), int(b) + 1)]
        if "*" in t or "?" in t or ("-" in t and not t.startswith(("L", "D", "F"))):
            return [self._tekil(x, d) for x in self._degiskenler(t)]
        return [self._tekil(t, d)]

    def _tekil(self, t: str, d: pd.DataFrame, etkilesim: bool = False) -> str:
        on = ""
        m = re.match(r"^(i|c|ib\d+|ib\(first\)|ib\(last\)|ib\(freq\))\.(.+)$", t, re.I)
        if m:
            on, t = m.group(1).lower(), m.group(2)
        islemler = ""
        mz = self._ZS.match(t)
        while mz:  # L.D.y → LD.y (zincirli zaman işlemcileri)
            islemler += mz.group(1)
            t = mz.group(2)
            mz = self._ZS.match(t)
        if islemler:
            ad = self._zaman_sutunu(d, islemler, t)
        else:
            ad = self._ad_coz(t)
            if ad not in d.columns:
                d[ad] = self.veri.loc[d.index, ad]
        if on.startswith("ib"):
            self._temel_duzey(d, ad, on)
            return f"C({ad})"
        if on == "i":
            return f"C({ad})"
        if on == "c" or etkilesim:
            return self._donusumler.get(ad, ad) if not etkilesim else ad
        if not pd.api.types.is_numeric_dtype(d[ad]) and not isinstance(d[ad].dtype, pd.CategoricalDtype):
            return f"C({ad})"  # metin değişkeni (Stata'da önce encode gerekir)
        return self._donusumler.get(ad, ad)

    def _temel_duzey(self, d: pd.DataFrame, ad: str, on: str) -> None:
        x = d[ad]
        kat = list(x.cat.categories) if isinstance(x.dtype, pd.CategoricalDtype) else sorted(x.dropna().unique())
        if on == "ib(first)":
            temel = kat[0]
        elif on == "ib(last)":
            temel = kat[-1]
        elif on == "ib(freq)":
            temel = x.value_counts().idxmax()
        else:
            kod = float(on[2:])
            de = self.veri.attrs.get("deger_etiketleri", {}).get(ad, {})
            temel = de.get(int(kod), kod) if de else (int(kod) if float(kod).is_integer() and int(kod) in kat else kod)
            if temel not in kat:
                raise _hata(f"{ad} değişkeninde {on[2:]} düzeyi yok. Düzeyler: {kat}")
        d[ad] = pd.Categorical(x, categories=[temel] + [k for k in kat if k != temel])

    def _formul(self, degiskenler: str, kosul, aralik, sabit: bool = True) -> tuple[str, pd.DataFrame]:
        v = self._veri_gerekli()
        maske = self._ornek(kosul, aralik)
        d = v.loc[maske, []].copy()
        d.attrs = {k: (dict(x) if isinstance(x, dict) else x) for k, x in v.attrs.items()}
        tokenlar = degiskenler.split()
        if not tokenlar:
            raise _hata("Bağımlı değişken belirtilmedi.")
        bagimli = self._terim(tokenlar[0], d)[0]
        bagimli = self._donusumler.get(bagimli, bagimli)
        sag = []
        for t in tokenlar[1:]:
            sag += self._terim(t, d)
        formul = f"{bagimli} ~ " + (" + ".join(sag) if sag else "1") + ("" if sabit else " - 1")
        # dönüşüm kaynağı sütunlar (np.log(ucret) gibi) çalışma verisinde olmalı
        for c in v.columns:
            if c not in d.columns and re.search(rf"(?<![\w.]){re.escape(c)}(?!\w)", formul):
                d[c] = v.loc[d.index, c]
        return formul, d

    @staticmethod
    def _vce(sec: dict) -> tuple[str, str | None]:
        """vce(...) / robust / cluster(...) seçeneklerini (tür, küme) çiftine çevirir."""
        vce = _secenek(sec, "vce", 3)
        if isinstance(vce, str):
            parcalar = vce.split()
            tur = parcalar[0].lower()
            if "robust".startswith(tur) and len(tur) >= 1:
                return "saglam", None
            if "cluster".startswith(tur) and len(tur) >= 2:
                if len(parcalar) < 2:
                    raise _hata("vce(cluster) için küme değişkenini yaz: vce(cluster il)")
                return "kume", parcalar[1]
            if tur in ("hc2", "hc3"):
                return tur.upper(), None
            if tur in ("ols", "oim"):
                return "klasik", None
            raise _hata(f"Desteklenmeyen vce({vce}). Seçenekler: robust, cluster değişken, hc2, hc3")
        if _secenek(sec, "robust", 1) is True:
            return "saglam", None
        kume = _secenek(sec, "cluster", 2)
        if isinstance(kume, str):
            return "kume", kume
        return "klasik", None

    # ------------------------------------------------------------------ veri komutları
    def _k_use(self, metin: str):
        ana, _, _, sec = self._parcala(metin)
        dosya = ana.replace("using", "").strip().strip('"').strip("'")
        from .girdi import oku

        if not Path(dosya).suffix:
            dosya += ".dta"
        self.veri = oku(dosya)
        self._donusumler = {}
        self._yaz(f"({sayi(len(self.veri), 0)} gözlem, {self.veri.shape[1]} değişken yüklendi: {dosya})")

    def _k_sysuse(self, metin: str):
        from . import veri as hazir

        ad = self._parcala(metin)[0]
        self.veri = hazir.yukle(ad)
        self.veri.attrs.setdefault("etiketler", {})
        if isinstance(self.veri.index, pd.PeriodIndex):
            ad_zaman = self.veri.index.name or "donem"
            self.veri = self.veri.reset_index()
            self.veri.attrs["tsset"] = ad_zaman
        self._donusumler = {}
        self._yaz(f"({len(self.veri)} gözlem, {self.veri.shape[1]} değişken: {ad} — simüle eğitim verisi)")

    def _k_import(self, metin: str):
        tur, _, kalan = metin.partition(" ")
        tur = tur.lower()
        from .girdi import oku

        ana, _, _, sec = self._parcala(kalan)
        dosya = ana.replace("using", "").strip().strip('"').strip("'")
        if "excel".startswith(tur) and len(tur) >= 3:
            sayfa = sec.get("sheet")
            self.veri = oku(dosya, sayfa=sayfa if isinstance(sayfa, str) else None)
        elif "delimited".startswith(tur) and len(tur) >= 3:
            ay = sec.get("delimiters") or sec.get("delimiter")
            self.veri = oku(dosya, ayirici=ay if isinstance(ay, str) else None)
        else:
            raise _hata("import excel ya da import delimited kullan.")
        self._donusumler = {}
        self._yaz(f"({len(self.veri)} gözlem, {self.veri.shape[1]} değişken yüklendi: {dosya})")

    def _k_save(self, metin: str):
        from .girdi import kaydet

        ana, _, _, sec = self._parcala(metin)
        dosya = ana.strip().strip('"')
        if not Path(dosya).suffix:
            dosya += ".dta"
        if Path(dosya).exists() and not sec.get("replace"):
            raise _hata(f"{dosya} zaten var; üzerine yazmak için: save {dosya}, replace")
        kaydet(self._veri_gerekli(), dosya)
        self._yaz(f"dosya {dosya} kaydedildi")

    def _k_describe(self, metin: str):
        v = self._veri_gerekli()
        sutunlar = self._degiskenler(self._parcala(metin)[0]) if self._parcala(metin)[0] else list(v.columns)
        et = v.attrs.get("etiketler", {})
        satirlar = [f"{'Değişken':<20}{'Tür':<14}Etiket", "─" * 60]
        for c in sutunlar:
            tur = (
                "kategorik"
                if isinstance(v[c].dtype, pd.CategoricalDtype)
                else ("sayısal" if pd.api.types.is_numeric_dtype(v[c]) else "metin")
            )
            satirlar.append(f"{c:<20}{tur:<14}{et.get(c, '')}")
        self._yaz(f"Gözlem: {len(v)}  Değişken: {v.shape[1]}\n" + "\n".join(satirlar))

    def _k_list(self, metin: str):
        ana, kosul, aralik, _ = self._parcala(metin)
        v = self._veri_gerekli()
        sutunlar = self._degiskenler(ana) if ana else list(v.columns)
        if not kosul and not aralik:
            aralik = "1/20" if len(v) > 20 else None
        t = v.loc[self._ornek(kosul, aralik), sutunlar]
        self._yaz(t.to_string())
        return t

    def _k_count(self, metin: str):
        _, kosul, aralik, _ = self._parcala(metin)
        n = int(self._ornek(kosul, aralik).sum())
        self._yaz(f"  {n:,}".replace(",", "."))
        return n

    # ------------------------------------------------------------------ değişken komutları
    def _k_generate(self, metin: str, degistir: bool = False):
        v = self._veri_gerekli()
        esit = _dis_dizin(metin, "=")
        if esit < 0:
            raise _hata("Söz dizimi: generate yeni = ifade [if koşul]")
        sol = metin[:esit].split()
        ad = sol[-1]
        if not degistir and ad in v.columns:
            raise _hata(f"{ad} zaten tanımlı (Stata: 'variable {ad} already defined'). Değiştirmek için replace kullan.")
        if degistir:
            ad = self._ad_coz(ad)
        ifade, kosul, aralik, _ = self._parcala(metin[esit + 1 :])
        deger = self._degerlendir(ifade)
        if not isinstance(deger, pd.Series):
            deger = pd.Series([deger] * len(v), index=v.index)
        if deger.dtype == bool:
            deger = deger.astype(float)
        maske = self._ornek(kosul, aralik)
        if degistir:
            eski = v[ad]
            yeni = eski.where(~maske, deger)
            degisen = int((~((eski == yeni) | (eski.isna() & yeni.isna()))).sum())
            v[ad] = yeni
            self._yaz(f"({degisen} gerçek değişiklik yapıldı)")
        else:
            v[ad] = deger.where(maske)
            eksik = int(v[ad].isna().sum())
            if eksik:
                self._yaz(f"({eksik} eksik değer üretildi)")
        # yorum için dönüşümleri hatırla: ln(x) → log, x^2 → kare
        sade = ifade.replace(" ", "")
        m = re.fullmatch(r"(?:ln|log)\((\w+)\)", sade)
        if m and m.group(1) in v.columns and not kosul:
            self._donusumler[ad] = f"np.log({m.group(1)})"
            v.attrs.setdefault("etiketler", {})
        m = re.fullmatch(r"(\w+)\^2|(\w+)\*\2", sade) or re.fullmatch(r"(\w+)\*(\w+)", sade)
        if m and not kosul:
            g = [x for x in m.groups() if x]
            if len(set(g)) == 1 and g[0] in v.columns:
                self._donusumler[ad] = f"I({g[0]}**2)"
        if degistir:
            self._donusumler.pop(ad, None)

    def _k_replace(self, metin: str):
        return self._k_generate(metin, degistir=True)

    def _k_egen(self, metin: str, by: list[str] | None = None):
        v = self._veri_gerekli()
        esit = _dis_dizin(metin, "=")
        ad = metin[:esit].split()[-1]
        ifade, kosul, aralik, sec = self._parcala(metin[esit + 1 :])
        m = re.fullmatch(r"\s*(\w+)\((.*)\)\s*", ifade)
        if not m:
            raise _hata("Söz dizimi: egen yeni = mean(x) [if], by(grup)")
        islev, ic = m.group(1).lower(), m.group(2)
        islev = {
            "total": "sum",
            "sum": "sum",
            "mean": "mean",
            "sd": "std",
            "min": "min",
            "max": "max",
            "count": "count",
            "median": "median",
        }.get(islev)
        if islev is None:
            raise _hata("egen için desteklenen işlevler: mean, total, sum, sd, min, max, count, median")
        by = by or (self._degiskenler(sec["by"]) if isinstance(sec.get("by"), str) else None)
        maske = self._ornek(kosul, aralik)
        x = self._degerlendir(ic)
        x = (x if isinstance(x, pd.Series) else pd.Series(x, index=v.index)).where(maske)
        if by:
            sonuc = x.groupby([v[b] for b in by], observed=True).transform(islev)
        else:
            sonuc = pd.Series(getattr(x, islev)(), index=v.index)
        v[ad] = sonuc.where(maske)

    def _by(self, metin: str):
        iki = _dis_dizin(metin, ":")
        if iki < 0:
            raise _hata("Söz dizimi: bysort grup: egen yeni = mean(x)")
        gruplar = [g for g in re.sub(r"\(.*?\)", " ", metin[:iki]).split()]
        komut = metin[iki + 1 :].strip()
        ilk, _, kalan = komut.partition(" ")
        if _KISALTMA.get(ilk, ilk) != "egen":
            raise _hata("by / bysort öneki yalnızca egen ile desteklenir (ör. bysort il: egen ort = mean(gsyh)).")
        return self._k_egen(kalan, by=self._degiskenler(" ".join(gruplar)))

    def _k_drop(self, metin: str, tut: bool = False):
        v = self._veri_gerekli()
        ana, kosul, aralik, _ = self._parcala(metin)
        if kosul or aralik:
            maske = self._ornek(kosul, aralik)
            yeni = v.loc[maske if tut else ~maske]
            self._yaz(f"({int((~maske).sum() if tut else maske.sum())} gözlem silindi)")
        else:
            sutunlar = self._degiskenler(ana)
            yeni = v[sutunlar] if tut else v.drop(columns=sutunlar)
        yeni = yeni.copy()
        yeni.attrs = v.attrs
        self.veri = yeni

    def _k_keep(self, metin: str):
        return self._k_drop(metin, tut=True)

    def _k_rename(self, metin: str):
        eski, yeni = metin.split()[:2]
        eski = self._ad_coz(eski)
        v = self._veri_gerekli()
        attrs = v.attrs
        self.veri = v.rename(columns={eski: yeni})
        self.veri.attrs = attrs
        et = attrs.get("etiketler", {})
        if eski in et:
            et[yeni] = et.pop(eski)

    def _k_label(self, metin: str):
        metin = metin.strip()
        alt = metin.split()[0].lower() if metin else ""
        if alt.startswith("var"):
            m = re.match(r'^var(?:iable)?\s+(\S+)\s+"(.*)"\s*$', metin, re.I)
            if not m:
                raise _hata('Söz dizimi: label variable x "Etiket"')
            ad = self._ad_coz(m.group(1))
            self._veri_gerekli().attrs.setdefault("etiketler", {})[ad] = m.group(2)
            return None
        if alt.startswith("def"):
            m = re.match(r"^def\w*\s+(\w+)\s+(.*?)(?:,\s*(\w[\w ]*))?$", metin, re.I | re.S)
            if not m:
                raise _hata('Söz dizimi: label define ad 1 "Erkek" 2 "Kadın" [, replace add]')
            ad, govde, sec = m.group(1), m.group(2), (m.group(3) or "").lower()
            ciftler = re.findall(r'(-?\d+(?:\.\d+)?)\s+"([^"]*)"', govde)
            if not ciftler:
                raise _hata('label define için kod–etiket çiftleri yaz: 1 "Erkek" 2 "Kadın"')
            tanimlar = self._veri_gerekli().attrs.setdefault("etiket_tanimlari", {})
            if ad in tanimlar and not ("replace" in sec or "add" in sec):
                raise _hata(f"label {ad} already defined — üzerine yazmak için: , replace")
            yeni = {} if ("replace" in sec or ad not in tanimlar) else dict(tanimlar[ad])
            for k, e in ciftler:
                yeni[int(float(k)) if float(k).is_integer() else float(k)] = e
            tanimlar[ad] = yeni
            return None
        if alt.startswith("val"):
            parca = metin.split()
            if len(parca) < 3:
                raise _hata("Söz dizimi: label values değişken(ler) etiket_adı")
            v = self._veri_gerekli()
            tanim = v.attrs.get("etiket_tanimlari", {}).get(parca[-1])
            if tanim is None:
                raise _hata(f'label {parca[-1]} not found — önce: label define {parca[-1]} 1 "..." 2 "..."')
            for ad in self._degiskenler(" ".join(parca[1:-1])):
                x = pd.to_numeric(v[ad], errors="coerce")
                kodlar = sorted(set(x.dropna().unique()) | set(tanim))
                etiket = {k: tanim.get(k, str(int(k)) if float(k).is_integer() else str(k)) for k in kodlar}
                v[ad] = pd.Categorical(x.map(etiket), categories=[etiket[k] for k in kodlar if k in set(x.dropna().unique())])
                v.attrs.setdefault("deger_etiketleri", {})[ad] = etiket
            return None
        raise _hata("Desteklenen: label variable, label define, label values")

    def _k_order(self, metin: str):
        v = self._veri_gerekli()
        ilk = self._degiskenler(self._parcala(metin)[0])
        attrs = v.attrs
        self.veri = v[ilk + [c for c in v.columns if c not in ilk]]
        self.veri.attrs = attrs

    # ------------------------------------------------------------------ veri birleştirme ve dönüştürme
    def _dosya_oku(self, ad: str) -> pd.DataFrame:
        from .girdi import oku

        ad = ad.strip().strip('"').strip("'")
        if not Path(ad).suffix:
            ad += ".dta"
        return oku(ad)

    def _k_merge(self, metin: str):
        v = self._veri_gerekli()
        virgul = _dis_dizin(metin, ",")
        govde, sec = (metin[:virgul], _secenekler(metin[virgul + 1 :])) if virgul >= 0 else (metin, {})
        m = re.match(r"^\s*(1:1|m:1|1:m|m:m)\s+(.+?)\s+using\s+(.+?)\s*$", govde, re.I)
        if not m:
            raise _hata("Söz dizimi: merge 1:1 | m:1 | 1:m anahtar(lar) using dosya [, keep(match) nogenerate keepusing(değişkenler)]")
        tur, anahtar_metni, dosya = m.group(1).lower(), m.group(2), m.group(3)
        if tur == "m:m":
            raise _hata("merge m:m desteklenmez (Stata da önermez); joinby ya da doğru anahtarla 1:m / m:1 kullan.")
        anahtar = self._degiskenler(anahtar_metni)
        kul = self._dosya_oku(dosya)
        eksik = [a for a in anahtar if a not in kul.columns]
        if eksik:
            raise _hata(f"Anahtar değişken(ler) using dosyasında yok: {eksik}")
        if tur in ("1:1", "1:m") and v.duplicated(anahtar).any():
            raise _hata(
                f"{' '.join(anahtar)} değişken(ler)i master verideki gözlemleri tekil olarak tanımlamıyor (Stata: variable does not uniquely identify observations in the master data)."
            )
        if tur in ("1:1", "m:1") and kul.duplicated(anahtar).any():
            raise _hata(f"{' '.join(anahtar)} değişken(ler)i using verisindeki gözlemleri tekil olarak tanımlamıyor.")
        if isinstance(sec.get("keepusing"), str):
            kul = kul[anahtar + [c for c in kul.columns if c in sec["keepusing"].split() and c not in anahtar]]
        uret = not sec.get("nogenerate") and not sec.get("nogen")
        if uret and "_merge" in v.columns:
            raise _hata("_merge değişkeni zaten var (Stata: variable _merge already defined); önce drop _merge.")
        ortak = [c for c in kul.columns if c in v.columns and c not in anahtar]
        kul_s = kul.drop(columns=ortak)  # Stata: ortak değişkenlerde master değeri korunur
        birlesik = v.merge(kul_s, on=anahtar, how="left", indicator=True)
        birlesik["_merge"] = np.where(birlesik["_merge"] == "both", 3, 1)
        anahtar_master = set(map(tuple, v[anahtar].astype(object).values.tolist()))
        sadece_using = kul[[tuple(r) not in anahtar_master for r in kul[anahtar].astype(object).values.tolist()]].copy()
        sadece_using["_merge"] = 2
        sonuc = pd.concat([birlesik, sadece_using], ignore_index=True, sort=False)
        kosul = sec.get("keep")
        if isinstance(kosul, str):
            kodlar = {"master": 1, "match": 3, "using": 2, "1": 1, "2": 2, "3": 3, "mas": 1, "mat": 3, "us": 2}
            tut = {kodlar[k.lower()] for k in kosul.split() if k.lower() in kodlar}
            sonuc = sonuc[sonuc["_merge"].isin(tut)]
        sayim = sonuc["_merge"].value_counts()
        attrs = dict(v.attrs)
        attrs["etiketler"] = {**kul.attrs.get("etiketler", {}), **v.attrs.get("etiketler", {})}
        attrs["deger_etiketleri"] = {**kul.attrs.get("deger_etiketleri", {}), **v.attrs.get("deger_etiketleri", {})}
        if not uret:
            sonuc = sonuc.drop(columns="_merge")
        sonuc = sonuc.reset_index(drop=True)
        sonuc.attrs = attrs
        self.veri = sonuc
        n1, n2, n3 = (int(sayim.get(i, 0)) for i in (1, 2, 3))
        self._yaz(
            f"    Sonuç                      Gözlem\n    {'─' * 36}\n    Eşleşmeyen               {n1 + n2:>9}\n"
            f"        master'dan           {n1:>9}  (_merge==1)\n        using'den            {n2:>9}  (_merge==2)\n\n"
            f"    Eşleşen                  {n3:>9}  (_merge==3)"
        )
        return sayim

    def _k_append(self, metin: str):
        v = self._veri_gerekli()
        govde = metin.split(",")[0]
        m = re.match(r"^\s*using\s+(.+)$", govde, re.I)
        if not m:
            raise _hata("Söz dizimi: append using dosya1 [dosya2 ...]")
        dosyalar = re.findall(r'"[^"]+"|\S+', m.group(1))
        parcalar = [v] + [self._dosya_oku(d) for d in dosyalar]
        attrs = dict(v.attrs)
        for p in parcalar[1:]:
            attrs["etiketler"] = {**p.attrs.get("etiketler", {}), **attrs.get("etiketler", {})}
        self.veri = pd.concat(parcalar, ignore_index=True, sort=False)
        self.veri.attrs = attrs
        self._yaz(f"({len(self.veri) - len(v)} gözlem eklendi; toplam {len(self.veri)})")

    def _k_reshape(self, metin: str):
        v = self._veri_gerekli()
        yon, _, kalan = metin.strip().partition(" ")
        ana, _, _, sec = self._parcala(kalan)
        if not isinstance(sec.get("i"), str) or not isinstance(sec.get("j"), str):
            raise _hata("Söz dizimi: reshape long|wide kök(ler), i(kimlik) j(değişken) [string]")
        i_ = self._degiskenler(sec["i"])
        j = sec["j"].split()[0]
        kokler = ana.split()
        attrs = dict(v.attrs)
        attrs.pop("xtset", None)
        attrs.pop("tsset", None)
        if yon.lower() == "long":
            sonek = r".+" if sec.get("string") else r"-?\d+"
            sutunlar = [c for c in v.columns if any(re.fullmatch(re.escape(k) + sonek, c) for k in kokler)]
            if not sutunlar:
                raise _hata(f"{kokler} ile başlayan geniş biçimli değişken bulunamadı (ör. gelir2020 gelir2021).")
            if v.duplicated(i_).any():
                raise _hata(f"i({' '.join(i_)}) gözlemleri tekil olarak tanımlamıyor.")
            uzun = pd.wide_to_long(v, stubnames=kokler, i=i_, j=j, sep="", suffix=sonek).reset_index()
            uzun = uzun.sort_values(i_ + [j]).reset_index(drop=True)
            if not sec.get("string"):
                uzun[j] = pd.to_numeric(uzun[j])
            uzun = uzun[i_ + [j] + [c for c in uzun.columns if c not in i_ + [j]]]
            self.veri = uzun
        elif yon.lower() == "wide":
            if j not in v.columns:
                raise _hata(f"j({j}) değişkeni veride yok.")
            if v.duplicated(i_ + [j]).any():
                raise _hata(f"{j} değerleri i({' '.join(i_)}) içinde tekil değil.")
            diger = [c for c in v.columns if c not in i_ + [j] + kokler]
            sabitmi = v.groupby(i_, observed=True)[diger].nunique(dropna=False).le(1).all() if diger else pd.Series(dtype=bool)
            degisen = [c for c in diger if not sabitmi.get(c, True)]
            if degisen:
                raise _hata(
                    f"{degisen} değişken(ler)i i içinde sabit değil (Stata: variable not constant within i); reshape listesine ekle ya da sil."
                )
            genis = v.pivot(index=i_, columns=j, values=kokler)
            genis.columns = [f"{k}{jj}" for k, jj in genis.columns]
            genis = genis.reset_index()
            if diger:
                genis = genis.merge(v.groupby(i_, observed=True, as_index=False)[diger].first(), on=i_, how="left")
            self.veri = genis
        else:
            raise _hata("reshape long ya da reshape wide yaz.")
        self.veri.attrs = attrs
        self._yaz(f"(reshape {yon.lower()}: {len(v)} → {len(self.veri)} gözlem, {v.shape[1]} → {self.veri.shape[1]} değişken)")

    def _k_collapse(self, metin: str):
        v = self._veri_gerekli()
        ana, kosul, aralik, sec = self._parcala(metin)
        istat = "mean"
        hedefler: list[tuple[str, str, str]] = []
        for t in re.findall(r"\(\w+\)|\S+", ana):
            if t.startswith("("):
                istat = t[1:-1].lower()
                continue
            yeni, _, kaynak = t.partition("=")
            if not kaynak:
                for c in self._degiskenler(yeni):
                    hedefler.append((c, c, istat))
            else:
                hedefler.append((yeni, self._ad_coz(kaynak), istat))
        islev = {
            "mean": "mean",
            "sum": "sum",
            "rawsum": "sum",
            "count": "count",
            "sd": "std",
            "min": "min",
            "max": "max",
            "median": "median",
            "p50": "median",
            "first": "first",
            "last": "last",
            "firstnm": "first",
            "lastnm": "last",
        }
        bilinmeyen = {i for _, _, i in hedefler} - set(islev)
        if bilinmeyen:
            raise _hata(f"Desteklenmeyen collapse istatistiği: {bilinmeyen}. Desteklenenler: {', '.join(islev)}")
        by = self._degiskenler(sec["by"]) if isinstance(sec.get("by"), str) else []
        alt = v.loc[self._ornek(kosul, aralik)]
        sutunlar = {}
        if by:
            g = alt.groupby(by, observed=True, sort=True)
            for yeni, kaynak, i in hedefler:
                sutunlar[yeni] = g[kaynak].agg(islev[i])
            sonuc = pd.DataFrame(sutunlar).reset_index()
        else:
            sonuc = pd.DataFrame({yeni: [alt[kaynak].agg(islev[i])] for yeni, kaynak, i in hedefler})
        et = v.attrs.get("etiketler", {})
        attrs = {"etiketler": {yeni: f"({i}) {et.get(k, k)}" for yeni, k, i in hedefler}}
        self.veri = sonuc
        self.veri.attrs = attrs

    def _k_encode(self, metin: str):
        v = self._veri_gerekli()
        ana, _, _, sec = self._parcala(metin)
        kaynak = self._ad_coz(ana.split()[0])
        yeni = sec.get("generate") or sec.get("gen")
        if not isinstance(yeni, str):
            raise _hata("Söz dizimi: encode metin_değişkeni, gen(yeni_değişken)")
        if yeni in v.columns:
            raise _hata(f"{yeni} zaten tanımlı.")
        degerler = sorted(str(x) for x in v[kaynak].dropna().unique())  # Stata: alfabetik sıra → 1, 2, ...
        v[yeni] = pd.Categorical(v[kaynak].astype("string").astype(object).where(v[kaynak].notna()), categories=degerler)
        v.attrs.setdefault("deger_etiketleri", {})[yeni] = {i + 1: e for i, e in enumerate(degerler)}

    def _k_decode(self, metin: str):
        v = self._veri_gerekli()
        ana, _, _, sec = self._parcala(metin)
        kaynak = self._ad_coz(ana.split()[0])
        yeni = sec.get("generate") or sec.get("gen")
        if not isinstance(yeni, str):
            raise _hata("Söz dizimi: decode kategorik_değişken, gen(yeni_değişken)")
        v[yeni] = v[kaynak].astype(object).where(v[kaynak].notna())

    def _k_destring(self, metin: str):
        v = self._veri_gerekli()
        ana, _, _, sec = self._parcala(metin)
        adlar = (
            self._degiskenler(ana)
            if ana
            else [c for c in v.columns if not pd.api.types.is_numeric_dtype(v[c]) and not isinstance(v[c].dtype, pd.CategoricalDtype)]
        )
        yeniler = (sec.get("generate") or sec.get("gen") or "").split() if isinstance(sec.get("generate") or sec.get("gen"), str) else None
        if not sec.get("replace") and not yeniler:
            raise _hata("destring için replace ya da gen(yeni adlar) seçeneğini belirt.")
        if yeniler and len(yeniler) != len(adlar):
            raise _hata("gen() içindeki ad sayısı değişken sayısına eşit olmalı.")
        yok_say = sec.get("ignore") if isinstance(sec.get("ignore"), str) else ""
        for i, ad in enumerate(adlar):
            s = v[ad].astype("string")
            for ch in yok_say:
                s = s.str.replace(ch, "", regex=False)
            if sec.get("dpcomma"):
                s = s.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
            s = s.str.strip()
            sayi_ = pd.to_numeric(s.replace({"": None, ".": None}), errors="coerce")
            bozuk = sayi_.isna() & s.notna() & ~s.isin(["", "."])
            if bozuk.any() and not sec.get("force"):
                self._yaz(f"{ad}: sayısal olmayan karakterler içeriyor ({int(bozuk.sum())} gözlem); değiştirilmedi (force ile zorla)")
                continue
            v[yeniler[i] if yeniler else ad] = sayi_.astype(float)
            self._yaz(f"{ad}: {'yeni değişken ' + yeniler[i] if yeniler else 'sayıya çevrildi'}")

    def _k_recode(self, metin: str):
        v = self._veri_gerekli()
        ana, kosul, aralik, sec = self._parcala(metin)
        m = re.match(r"^\s*(\S+)\s+(.*)$", ana)
        if not m:
            raise _hata("Söz dizimi: recode x (1 2 = 1) (3/5 = 2) (else = .) [, gen(yeni)]")
        ad = self._ad_coz(m.group(1))
        kurallar = re.findall(r"\(([^()]*)=([^()]*)\)", m.group(2))
        if not kurallar:
            raise _hata("recode kuralları parantez içinde yazılmalı: (1 2 = 1) (3/5 = 2)")
        x = pd.to_numeric(v[ad], errors="coerce")
        sonuc = x.copy()
        atanmis = pd.Series(False, index=v.index)
        maske = self._ornek(kosul, aralik)

        def deger(t):
            t = t.strip()
            return np.nan if t == "." else float(t)

        for sol, sag in kurallar:
            hedef = deger(sag)
            sec_ = pd.Series(False, index=v.index)
            for tok in sol.split():
                tl = tok.lower()
                if tl == "else":
                    sec_ |= ~atanmis
                elif tl in ("missing", "miss"):
                    sec_ |= x.isna()
                elif tl in ("nonmissing", "nonm"):
                    sec_ |= x.notna()
                elif "/" in tok:
                    a, b = tok.split("/")
                    alt = -np.inf if a.lower() == "min" else float(a)
                    ust = np.inf if b.lower() == "max" else float(b)
                    sec_ |= (x >= alt) & (x <= ust)
                elif tok == ".":
                    sec_ |= x.isna()
                else:
                    sec_ |= x == float(tok)
            sec_ &= ~atanmis & maske
            sonuc[sec_] = hedef
            atanmis |= sec_
        yeni = sec.get("generate") or sec.get("gen")
        v[yeni if isinstance(yeni, str) else ad] = sonuc
        self._yaz(f"({int(((sonuc != x) & ~(sonuc.isna() & x.isna())).sum())} değişiklik yapıldı)")

    def _k_duplicates(self, metin: str):
        v = self._veri_gerekli()
        alt, _, kalan = metin.strip().partition(" ")
        ana, _, _, sec = self._parcala(kalan)
        sutunlar = self._degiskenler(ana) if ana else list(v.columns)
        tekrar = v.duplicated(sutunlar, keep="first")
        if alt.lower().startswith("rep"):
            kopya = v.groupby(sutunlar, dropna=False, observed=True).size()
            tablo = kopya.value_counts().sort_index()
            self._yaz("Kopya sayısı  Gözlem  Fazlalık\n" + "\n".join(f"{k:>12} {k * n:>7} {(k - 1) * n:>9}" for k, n in tablo.items()))
            return int(tekrar.sum())
        if alt.lower() == "drop":
            if ana and not sec.get("force"):
                raise _hata("Değişken listesiyle duplicates drop için force seçeneği gerekir (Stata ile aynı).")
            attrs = v.attrs
            self.veri = v.loc[~tekrar].reset_index(drop=True)
            self.veri.attrs = attrs
            self._yaz(f"({int(tekrar.sum())} kopya gözlem silindi)")
            return int(tekrar.sum())
        raise _hata("Desteklenen: duplicates report [değişkenler], duplicates drop [değişkenler, force]")

    def _k_sort(self, metin: str):
        v = self._veri_gerekli()
        attrs = v.attrs
        self.veri = v.sort_values(self._degiskenler(self._parcala(metin)[0])).reset_index(drop=True)
        self.veri.attrs = attrs

    def _k_tsset(self, metin: str):
        from .girdi import tsset

        ad = self._ad_coz(self._parcala(metin)[0].split()[0])
        v = self._veri_gerekli()
        sirali = v.sort_values(ad).reset_index(drop=True)
        if sirali[ad].duplicated().any():
            raise _hata(f"{ad} tekrarlanan değerler içeriyor (Stata: repeated time values in sample).")
        sirali.attrs = {**v.attrs, "tsset": ad}
        sirali.attrs.pop("xtset", None)
        self.veri = sirali
        if not self._sessiz and self.yazdir:
            tsset(sirali, ad)

    def _k_xtset(self, metin: str):
        from .girdi import xtset

        adlar = self._parcala(metin)[0].split()
        if len(adlar) != 2:
            raise _hata("Söz dizimi: xtset birim zaman")
        self.veri = xtset(self._veri_gerekli(), self._ad_coz(adlar[0]), self._ad_coz(adlar[1]), sessiz=self._sessiz or not self.yazdir)
        self.veri.attrs.pop("tsset", None)

    # ------------------------------------------------------------------ tanımlayıcı
    def _k_summarize(self, metin: str):
        from .karsilastirma import tanimlayici

        ana, kosul, aralik, sec = self._parcala(metin)
        v = self._veri_gerekli()
        sutunlar = self._degiskenler(ana) if ana else [c for c in v.columns if pd.api.types.is_numeric_dtype(v[c])]
        ag = sec.get("_agirlik")
        if ag and ag[0] == "pw":
            raise _hata("summarize pweight kabul etmez (Stata ile aynı); aw ya da fw kullan.")
        alt = v.loc[self._ornek(kosul, aralik)]
        t = tanimlayici(
            alt, sutunlar, detay=bool(_secenek(sec, "detail", 1)), agirlik=alt[ag[1]] if ag else None, agirlik_turu=ag[0] if ag else "aw"
        )
        if not _secenek(sec, "detail", 1):
            t = t[["N", "Ortalama", "Std. sapma", "En küçük", "En büyük"]]
        self._yaz(t.to_string(float_format=_tsayi))
        return t

    def _k_tabulate(self, metin: str):
        ana, kosul, aralik, sec = self._parcala(metin)
        v = self._veri_gerekli().loc[self._ornek(kosul, aralik)]
        adlar = self._degiskenler(ana)
        if len(adlar) == 1:
            f = v[adlar[0]].value_counts(sort=False).sort_index()
            t = pd.DataFrame({"Frekans": f, "Yüzde": 100 * f / f.sum(), "Birikimli": 100 * f.cumsum() / f.sum()})
            t.loc["Toplam"] = [f.sum(), 100.0, np.nan]
            self._yaz(t.to_string(float_format=lambda x: sayi(x, 2)))
            return t
        if len(adlar) == 2:
            if sec.get("chi2") or sec.get("ch") or sec.get("v") or sec.get("lrchi2"):
                from .klasik import ki_kare

                sonuc = ki_kare(v, adlar[0], adlar[1])
                self._yaz(sonuc)
                return sonuc
            t = pd.crosstab(v[adlar[0]], v[adlar[1]], margins=True, margins_name="Toplam")
            self._yaz(t.to_string())
            return t
        raise _hata("tabulate en fazla iki değişkenle kullanılır.")

    def _k_correlate(self, metin: str, cift: bool = False):
        from .karsilastirma import korelasyon

        ana, kosul, aralik, sec = self._parcala(metin)
        v = self._veri_gerekli()
        sutunlar = self._degiskenler(ana) if ana else None
        k = korelasyon(v.loc[self._ornek(kosul, aralik)], sutunlar, eksik="cift" if cift else "liste")
        self._yaz(k if (cift and sec) or not cift else k.r.round(4).to_string())
        return k

    def _k_pwcorr(self, metin: str):
        return self._k_correlate(metin, cift=True)

    # ------------------------------------------------------------------ tahmin komutları
    def _kaydet_sonuc(self, m, komut: str):
        m._stata_komutu = komut
        self.son = m
        self._yaz(m)
        return m

    def _k_regress(self, metin: str):
        from .regresyon import ekk

        ana, kosul, aralik, sec = self._parcala(metin)
        formul, d = self._formul(ana, kosul, aralik, sabit=_secenek(sec, "noconstant", 3) is None)
        tur, kume = self._vce(sec)
        if kume:
            kume = self._ad_coz(kume)
            d[kume] = self.veri.loc[d.index, kume]
        tur = {"saglam": "HC1"}.get(tur, tur)
        ag = sec.get("_agirlik")
        if ag:
            d[ag[1]] = self.veri.loc[d.index, ag[1]]
        m = ekk(formul, d, kovaryans=tur, kume=kume, agirlik=ag[1] if ag else None, agirlik_turu=ag[0] if ag else "aw")
        m._zaman_serisi = "tsset" in self.veri.attrs
        return self._kaydet_sonuc(m, "regress " + metin)

    def _ikili(self, metin: str, tur: str):
        from .ikili import _tahmin_et

        ana, kosul, aralik, sec = self._parcala(metin)
        formul, d = self._formul(ana, kosul, aralik, sabit=_secenek(sec, "noconstant", 3) is None)
        kov, kume = self._vce(sec)
        if kov in ("HC2", "HC3"):
            raise _hata(f"{tur} için vce({kov.lower()}) yok; vce(robust) ya da vce(cluster) kullan.")
        if kume:
            kume = self._ad_coz(kume)
            d[kume] = self.veri.loc[d.index, kume]
        ag = sec.get("_agirlik")
        if ag:
            d[ag[1]] = self.veri.loc[d.index, ag[1]]
        m = _tahmin_et(formul, d, tur, kov, None, kume=kume, agirlik=ag[1] if ag else None, agirlik_turu=ag[0] if ag else "pw")
        return self._kaydet_sonuc(m, f"{tur} " + metin)

    def _k_logit(self, metin: str):
        return self._ikili(metin, "logit")

    def _k_probit(self, metin: str):
        return self._ikili(metin, "probit")

    def _k_xtreg(self, metin: str):
        from .panel import f_testi, panel

        ana, kosul, aralik, sec = self._parcala(metin)
        xt = self._veri_gerekli().attrs.get("xtset")
        if xt is None:
            raise _hata("xtreg için önce panel yapısını tanımla: xtset birim zaman")
        formul, d = self._formul(ana, kosul, aralik)
        for c in xt:
            d[c] = self.veri.loc[d.index, c]
        if sec.get("be") or sec.get("mle") or sec.get("pa"):
            raise _hata("xtreg için yalnızca fe ve re desteklenir.")
        yontem = "sabit" if sec.get("fe") else "rassal"  # Stata varsayılanı re
        kov, kume = self._vce(sec)
        if kov == "saglam" or kume:
            if kume and self._ad_coz(kume) != xt[0]:
                raise _hata(f"xtreg'de yalnızca panel değişkenine ({xt[0]}) göre kümeleme desteklenir.")
            kov = "kume"
        elif kov != "klasik":
            raise _hata("xtreg için vce(robust) ya da vce(cluster panel) kullan.")
        m = panel(formul, d, birim=xt[0], zaman=xt[1], yontem=yontem, kovaryans=kov, _derinlik=2)
        self._kaydet_sonuc(m, "xtreg " + metin)
        if yontem == "sabit" and kov == "klasik":
            h = panel(formul, d, birim=xt[0], zaman=xt[1], yontem="havuz", _derinlik=2)
            ft = f_testi(h, m)
            self._yaz(
                f"Tüm u_i = 0 F testi: F({ft.serbestlik[0]}, {ft.serbestlik[1]}) = {sayi(ft.istatistik, 2)}   Prob > F = {sayi(ft.p_degeri, 4)}"
            )
        return m

    def _k_ivregress(self, metin: str):
        from .arac import iv2sls

        yontem, _, kalan = metin.strip().partition(" ")
        if yontem.lower() != "2sls":
            raise _hata("Yalnızca ivregress 2sls desteklenir.")
        ana, kosul, aralik, sec = self._parcala(kalan)
        m = re.search(r"\(([^()]*)=([^()]*)\)", ana)
        if not m:
            raise _hata("Söz dizimi: ivregress 2sls y x1 x2 (içsel = araç1 araç2)")
        dis = (ana[: m.start()] + " " + ana[m.end() :]).split()
        formul, d = self._formul(" ".join(dis), kosul, aralik, sabit=_secenek(sec, "noconstant", 3) is None)
        icsel = [t for x in m.group(1).split() for t in self._terim(x, d)]
        araclar = [t for x in m.group(2).split() for t in self._terim(x, d)]
        sag = formul.split("~", 1)[1].strip()
        sag = "" if sag == "1" else sag + " + "
        iv_formul = f"{formul.split('~', 1)[0].strip()} ~ {sag}[{' + '.join(icsel)} ~ {' + '.join(araclar)}]"
        kov, kume = self._vce(sec)
        if kov in ("HC2", "HC3"):
            raise _hata("ivregress için vce(robust) ya da vce(cluster) kullan.")
        if kume:
            kume = self._ad_coz(kume)
            d[kume] = self.veri.loc[d.index, kume]
        sonuc = iv2sls(iv_formul, d, kovaryans=kov, kucuk_orneklem=bool(sec.get("small")), kume=kume)
        self._kaydet_sonuc(sonuc, "ivregress 2sls " + kalan)
        if sec.get("first"):
            self._k_estat("firststage")
        return sonuc

    def _k_dfuller(self, metin: str):
        from .zaman import adf

        ana, kosul, aralik, sec = self._parcala(metin)
        v = self._veri_gerekli()
        if "tsset" not in v.attrs:
            raise _hata("dfuller için önce zaman yapısını tanımla: tsset zaman")
        maske = self._ornek(kosul, aralik)
        d = v.loc[maske].copy()
        d.attrs = v.attrs
        t = ana.split()[0]
        mz = self._ZS.match(t)
        seri = d[self._zaman_sutunu(d, mz.group(1), mz.group(2))] if mz else d[self._ad_coz(t)]
        det = "trend" if sec.get("trend") else ("yok" if _secenek(sec, "noconstant", 3) else "sabit")
        lags = _secenek(sec, "lags", 1)
        sonuc = adf(seri.dropna().astype(float), det, gecikme=int(lags) if isinstance(lags, str) else 0, ad=t)
        self._yaz(sonuc)
        self._yaz("Not: p-değeri MacKinnon (1996) yaklaşık p-değeridir (Stata ile aynı); kritik değerler MacKinnon (2010) tablosundan.")
        return sonuc

    # ------------------------------------------------------------------ tahmin sonrası
    def _model_gerekli(self, *turler: type):
        if self.son is None:
            raise _hata("Önce bir model tahmin et (regress, logit, xtreg, ivregress ...).")
        if turler and not isinstance(self.son, turler):
            raise _hata(f"Bu komut son tahmin edilen modelle ({type(self.son).__name__}) kullanılamaz.")
        return self.son

    def _k_predict(self, metin: str):
        from .ikili import IkiliSonuc
        from .regresyon import EKKSonuc
        from .secim import MLSonuc

        m = self._model_gerekli()
        ana, kosul, aralik, sec = self._parcala(metin)
        v = self._veri_gerekli()
        if isinstance(m, MLSonuc):
            return self._ml_predict(m, ana.split(), kosul, aralik, sec)
        ad = ana.split()[-1]
        if ad in v.columns:
            raise _hata(f"{ad} zaten tanımlı.")
        if isinstance(m, EKKSonuc) and _secenek(sec, "residuals", 1):
            seri = m.kalintilar
        elif isinstance(m, IkiliSonuc) and not sec.get("xb"):
            seri = m.olasiliklar
        else:
            X = getattr(m, "X", None)
            if X is None or not isinstance(getattr(m, "katsayilar", None), pd.Series):
                raise _hata(
                    f"predict bu model için ({type(m).__name__}) desteklenmiyor; regress, logit, probit ve xtreg sonrasında kullan."
                )
            seri = pd.Series(X.values @ m.katsayilar[X.columns].values, index=X.index)
        v[ad] = seri.reindex(v.index)
        if kosul or aralik:
            v[ad] = v[ad].where(self._ornek(kosul, aralik))

    def _ml_predict(self, m, adlar: list[str], kosul, aralik, sec):
        """ologit/mlogit: predict p1 p2 … (sonuç başına) ya da predict p, outcome(#); poisson/nbreg: n; tobit: xb."""
        v = self._veri_gerekli()
        for ad in adlar:
            if ad in v.columns:
                raise _hata(f"{ad} zaten tanımlı.")
        t = m.tahminler("xb" if sec.get("xb") else None)
        if m.tur in ("ologit", "oprobit", "mlogit") and not sec.get("xb"):
            sonuc = _secenek(sec, "outcome", 3)
            if isinstance(sonuc, str):
                # Tam ad: "1" soneki "11" ve "21" sütunlarını da tutuyordu.
                hedef = f"Pr({m.bagimli}={sonuc.strip().lstrip('#')})"
                if hedef not in t.columns:
                    raise _hata(f"outcome({sonuc}) bulunamadı. Sonuçlar: {m.kategoriler}")
                t = t[[hedef]]
            elif len(adlar) == len(t.columns):
                pass
            elif len(adlar) == 1:
                t = t.iloc[:, :1]  # Stata: tek değişken → ilk sonucun olasılığı
            else:
                raise _hata(f"{len(t.columns)} sonuç var; {len(t.columns)} değişken adı ver (predict p1 p2 …) ya da outcome(#) kullan.")
        if len(adlar) != t.shape[1]:
            raise _hata(f"{t.shape[1]} yeni değişken adı gerekli, {len(adlar)} verildi.")
        maske = self._ornek(kosul, aralik) if (kosul or aralik) else None
        for ad, c in zip(adlar, t.columns):
            v[ad] = t[c].reindex(v.index)
            if maske is not None:
                v[ad] = v[ad].where(maske)
            self._yaz(f"{ad}: {c}")
        return t

    def _k_margins(self, metin: str):
        from .ikili import IkiliSonuc
        from .marjinal import marjlar
        from .regresyon import EKKSonuc
        from .secim import MLSonuc

        m = self._model_gerekli()
        ana, _, _, sec = self._parcala(metin)
        dydx = sec.get("dydx")
        dydx = dydx if isinstance(dydx, str) else None
        at = self._at_coz(sec["at"]) if isinstance(sec.get("at"), str) else {}
        grup = [self._ad_coz(x) for x in self._degiskenler(ana)] if ana.strip() else []
        if isinstance(m, MLSonuc):
            if at or grup or not dydx:
                raise _hata(f"{m.tur} sonrasında yalnızca margins, dydx(*) desteklenir.")
            ame = m.marjinal_etkiler()
            self._yaz("Ortalama marjinal etkiler (delta yöntemi)\n" + ame.to_string(index=False, float_format=lambda x: sayi(x, 4)))
            return ame
        if not isinstance(m, (EKKSonuc, IkiliSonuc)):
            raise _hata("margins regress, logit, probit, ologit, oprobit, mlogit, poisson, nbreg ve tobit sonrasında kullanılabilir.")
        if len(grup) > 1:
            raise _hata("margins için en fazla bir faktör değişkeni ver: margins cinsiyet, at(egitim=(8 12 16))")
        if isinstance(m, IkiliSonuc) and dydx and not at and not grup:
            ame = m.marjinal_etkiler()  # Stata: dydx() açıklayıcılar için AME (kategorik değişkenlerde düzey farkı)
            if dydx.strip() not in ("*", "_all"):
                ame = ame[ame["değişken"].isin(set(self._degiskenler(dydx)))]
            self._yaz(
                "Ortalama marjinal etkiler (delta yöntemi)\n"
                + ame[["AME", "std. hata", "z", "p"]].to_string(float_format=lambda x: sayi(x, 4))
            )
            return ame
        if dydx:
            adlar = self._degiskenler(dydx) if dydx.strip() not in ("*", "_all") else None
            if adlar is None:
                sag = m.formul.split("~", 1)[1]
                adlar = [c for c in m._veri.columns if re.search(rf"(?<![\w.]){re.escape(c)}(?!\w)", sag) and c not in at and c not in grup]
            sonuclar = [marjlar(m, at=at, grup=grup[0] if grup else None, dydx=self._ad_coz(a)) for a in adlar]
            tablo = pd.concat([s.tablo for s in sonuclar], ignore_index=True)
            from .marjinal import MarjSonucu

            sonuc = MarjSonucu(tablo, "etki", sonuclar[0].aciklama if len(sonuclar) == 1 else "Ortalama marjinal etkiler — delta yöntemi")
        else:
            sonuc = marjlar(m, at=at, grup=grup[0] if grup else None)
        self._son_marj = sonuc
        self._yaz(sonuc)
        return sonuc

    def _at_coz(self, metin: str) -> dict[str, list]:
        """at(x=(8 12 16) z=(0(5)20) w=3) → {"x": [8, 12, 16], "z": [0, 5, 10, 15, 20], "w": [3]}"""
        sonuc: dict[str, list] = {}
        for ad, deger in re.findall(r"([\w.]+)\s*=\s*(\([^()]*(?:\([^()]*\)[^()]*)*\)|[^\s()]+)", metin):
            sonuc[self._ad_coz(ad)] = _sayi_listesi(deger.strip()[1:-1] if deger.startswith("(") else deger)
        if not sonuc:
            raise _hata("at() söz dizimi: at(egitim=(8 12 16)) ya da at(tecrube=(0(5)30))")
        return sonuc

    def _k_estat(self, metin: str):
        from . import testler as T
        from .arac import IVSonuc
        from .regresyon import EKKSonuc

        mt = re.match(r"\s*(\w+)(.*)$", metin, re.S)
        if not mt:
            raise _hata("Söz dizimi: estat hettest | imtest, white | ovtest | vif | dwatson | bgodfrey | firststage | endogenous | overid")
        alt, kalan = mt.group(1).lower(), mt.group(2)
        ana, _, _, sec = self._parcala(kalan)
        m = self._model_gerekli()
        if alt in ("hettest", "hett"):
            self._model_gerekli(EKKSonuc)
            degisken = self._degiskenler(ana) if ana else ("aciklayici" if sec.get("rhs") else "tahmin")
            sonuc = T.breusch_pagan(m, degisken, koenker=bool(sec.get("iid")))
        elif alt in ("imtest", "imt"):
            self._model_gerekli(EKKSonuc)
            sonuc = T.white(m)
        elif alt in ("ovtest", "ovt"):
            self._model_gerekli(EKKSonuc)
            sonuc = T.reset(m, us=4)
        elif alt == "vif":
            self._model_gerekli(EKKSonuc)
            sonuc = T.vif(m)
            from .regresyon import gorunen_ad

            sonuc = sonuc.assign(**{"1/VIF": 1 / sonuc["VIF"]}).sort_values("VIF", ascending=False)
            sonuc.index = [gorunen_ad(m, i) for i in sonuc.index]
            self._yaz(sonuc.to_string(float_format=lambda x: sayi(x, 3)) + f"\nOrtalama VIF: {sayi(float(sonuc['VIF'].mean()), 2)}")
            return sonuc
        elif alt in ("dwatson", "dwa"):
            sonuc = T.durbin_watson(self._model_gerekli(EKKSonuc))
        elif alt in ("bgodfrey", "bgo"):
            lags = _secenek(sec, "lags", 1)
            sonuc = T.breusch_godfrey(self._model_gerekli(EKKSonuc), int(lags) if isinstance(lags, str) else 1)
        elif alt in ("firststage", "first"):
            iv = self._model_gerekli(IVSonuc)
            satir = [
                f"{ic}: F({b['sd'][0]}, {b['sd'][1]}) = {sayi(b['F'], 2)}  Prob > F = {sayi(b['p'], 4)}  kısmi R² = {sayi(b['kismi_r2'], 4)}"
                for ic, b in iv.birinci_asama.items()
            ]
            sonuc = [t for t in iv.tanilar() if t.ad.startswith("Birinci")]
            self._yaz("Birinci aşama özet istatistikleri\n" + "\n".join(satir))
            return sonuc
        elif alt in ("endogenous", "endog"):
            sonuc = [t for t in self._model_gerekli(IVSonuc).tanilar() if t.ad.startswith(("Durbin", "Wu-Hausman"))]
            self._yaz("\n".join(str(t) for t in sonuc))
            return sonuc
        elif alt == "overid":
            sonuc = [t for t in self._model_gerekli(IVSonuc).tanilar() if t.ad.startswith(("Sargan", "Basmann"))]
            if not sonuc:
                raise _hata("Model tam tanımlı (araç sayısı = içsel değişken sayısı); aşırı tanımlama testi yapılamaz.")
            self._yaz("\n".join(str(t) for t in sonuc))
            return sonuc
        elif alt in ("classification", "clas", "class", "classi", "classif"):
            from .ikili import IkiliSonuc

            cutoff = _secenek(sec, "cutoff", 3)
            sonuc = self._model_gerekli(IkiliSonuc).siniflandirma(float(cutoff) if isinstance(cutoff, str) else 0.5)
            a = sonuc.attrs
            self._yaz(
                sonuc.to_string()
                + "\n"
                + "\n".join(
                    f"{ad:<28}{sayi(100 * a[k], 2)}%"
                    for ad, k in (
                        ("Duyarlılık  Pr(+|D)", "duyarlilik"),
                        ("Özgüllük    Pr(−|~D)", "ozgulluk"),
                        ("Pozitif öngörü Pr(D|+)", "pozitif_ongoru"),
                        ("Negatif öngörü Pr(~D|−)", "negatif_ongoru"),
                        ("Doğru sınıflandırma", "dogruluk"),
                    )
                    if k in a
                )
            )
            return sonuc
        elif alt == "gof":
            from .ikili import IkiliSonuc

            g = _secenek(sec, "group", 2)
            if not isinstance(g, str):
                raise _hata("Stata'daki gibi grup sayısını ver: estat gof, group(10)  (Hosmer-Lemeshow testi)")
            hl = self._model_gerekli(IkiliSonuc).hosmer_lemeshow(int(g))
            self._yaz(
                f"Hosmer-Lemeshow uyum iyiliği testi ({len(hl['gruplar'])} grup)\n"
                f"   χ²({hl['sd']}) = {sayi(hl['ki_kare'], 2)}   Prob > χ² = {sayi(hl['p'], 4)}"
            )
            return hl
        elif alt == "abond":
            from .dinamik import DinamikSonuc

            d = self._model_gerekli(DinamikSonuc)
            self._yaz(
                "Arellano-Bond fark kalıntılarında otokorelasyon testi (H0: otokorelasyon yok)\n"
                + "\n".join(f"   AR({j}): z = {sayi(z, 4)}   Prob > z = {sayi(pv, 4)}" for j, (z, pv) in sorted(d.ar_testleri.items()))
            )
            return d.ar_testleri
        elif alt in ("sargan", "sarg"):
            from .dinamik import DinamikSonuc

            d = self._model_gerekli(DinamikSonuc)
            self._yaz(
                f"Sargan aşırı tanımlama testi (H0: araçlar geçerli)\n   χ²({d.sargan_sd}) = {sayi(d.sargan, 4)}   Prob > χ² = {sayi(d.sargan_p, 4)}"
            )
            return d.sargan
        elif alt in ("ectest", "ect"):
            from .ardl import ARDLSinirSonucu

            sonuc = self._model_gerekli(ARDLSinirSonucu)
        else:
            raise _hata(
                "Desteklenen estat komutları: hettest, imtest white, ovtest, vif, dwatson, bgodfrey, firststage, endogenous, overid, "
                "classification, gof, abond, sargan, ectest"
            )
        self._yaz(sonuc)
        return sonuc

    def _k_test(self, metin: str):
        from .regresyon import EKKSonuc

        m = self._model_gerekli(EKKSonuc)
        ana = self._parcala(metin)[0]
        if "=" in ana:
            kisit = ana.replace("==", "=")
        else:
            kisit = ["Intercept" if x == "_cons" else self._donusumler.get(x, x) for x in ana.split()]
            eksik = [x for x in kisit if x not in m.katsayilar.index]
            if eksik:
                raise _hata(f"Modelde olmayan terim(ler): {eksik}. Terimler: {list(m.katsayilar.index)}")
        sonuc = m.wald_testi(kisit)
        self._yaz(sonuc)
        return sonuc

    def _k_estimates(self, metin: str):
        alt, _, kalan = metin.strip().partition(" ")
        alt = alt.lower()
        if alt in ("store", "sto"):
            ad = kalan.strip()
            self.saklanan[ad] = self._model_gerekli()
            return None
        if alt in ("table", "tab"):
            return self._k_esttab(kalan)
        raise _hata("Desteklenen: estimates store ad, estimates table ad1 ad2")

    def _k_esttab(self, metin: str):
        from .karsilastirma import karsilastir

        ana, _, _, sec = self._parcala(metin)
        dosya = None
        i = _kelime_dis(" " + ana + " ", "using")
        if i >= 0:
            dosya = Path((" " + ana + " ")[i + 5 :].strip().strip('"'))
            ana = (" " + ana + " ")[:i]
        adlar = ana.split() or list(self.saklanan)
        if not adlar:
            raise _hata("Saklanmış tahmin yok. Önce: estimates store m1")
        eksik = [a for a in adlar if a not in self.saklanan]
        if eksik:
            raise _hata(f"Saklanmamış tahmin(ler): {eksik}. Önce: estimates store ad")
        k = karsilastir(*[self.saklanan[a] for a in adlar], adlar=adlar)
        if dosya is not None:
            if dosya.suffix.lower() not in (".docx", ".tex", ".xlsx", ".csv", ".md", ".html", ".rtf"):
                raise _hata("esttab using dosya.docx | .tex | .xlsx | .csv | .md | .html")
            if dosya.suffix.lower() == ".rtf":
                raise _hata(".rtf yerine .docx kullan (Word'de doğrudan açılır).")
            if dosya.exists() and not sec.get("replace"):
                raise _hata(f"{dosya} zaten var; üzerine yazmak için: esttab ... using {dosya}, replace")
            k.kaydet(dosya)
            self._yaz(f"Tablo kaydedildi: {dosya}")
            return k
        self._yaz(k)
        return k

    _k_etable = _k_esttab

    def _k_hausman(self, metin: str):
        from .panel import PanelSonuc, hausman

        ana, _, _, sec = self._parcala(metin)
        adlar = ana.split()
        if len(adlar) != 2:
            raise _hata("Söz dizimi: hausman tutarlı_model etkin_model  (ör. estimates store fe ... hausman fe re)")
        a, b = (self.saklanan.get(x) for x in adlar)
        if a is None or b is None:
            raise _hata(f"Saklanmamış tahmin: {adlar}. Önce: estimates store ad")
        if not (isinstance(a, PanelSonuc) and isinstance(b, PanelSonuc)):
            raise _hata("hausman yalnızca xtreg fe / xtreg re tahminleri için desteklenir.")
        sonuc = hausman(a, b, sigmamore=bool(sec.get("sigmamore")))
        self._yaz(sonuc)
        return sonuc

    def _k_display(self, metin: str):
        v = self.veri if self.veri is not None else pd.DataFrame(index=[0])
        deger = _Degerlendirici(v.iloc[:1] if len(v) else pd.DataFrame(index=[0]), {}, self.son)(metin)
        if isinstance(deger, pd.Series):
            deger = deger.iloc[0]
        if isinstance(deger, (bool, np.bool_)):
            deger = int(deger)
        if isinstance(deger, (int, float, np.floating, np.integer)):
            self._yaz(f"{float(deger):.8g}".replace(".", ",") if deger == deger else ".")
        else:
            self._yaz(deger)
        return deger

    # ------------------------------------------------------------------ klasik testler
    def _k_ttest(self, metin: str):
        from .klasik import ttest

        ana, kosul, aralik, sec = self._parcala(metin)
        v = self._veri_gerekli().loc[self._ornek(kosul, aralik)]
        if "==" in ana or ("=" in ana and "==" not in ana):
            sol, sag = [x.strip() for x in ana.replace("==", "=").split("=", 1)]
            sol = self._ad_coz(sol)
            try:
                deger = float(sag)
            except ValueError:
                sag = self._ad_coz(sag)
                if sec.get("unpaired"):
                    uzun = pd.DataFrame({"x": pd.concat([v[sol], v[sag]], ignore_index=True), "g": [sol] * len(v) + [sag] * len(v)})
                    sonuc = ttest(uzun, "x", by="g", esit_varyans=not sec.get("unequal"))
                else:
                    sonuc = ttest(v, sol, ikinci=sag)
            else:
                sonuc = ttest(v, sol, deger=deger)
        else:
            by = sec.get("by")
            if not isinstance(by, str):
                raise _hata("Söz dizimi: ttest x == 0 | ttest x == y | ttest x, by(grup) [unequal]")
            sonuc = ttest(v, self._ad_coz(ana.split()[0]), by=self._ad_coz(by), esit_varyans=not sec.get("unequal"))
        self._yaz(sonuc)
        return sonuc

    def _k_oneway(self, metin: str):
        from .klasik import anova

        ana, kosul, aralik, _ = self._parcala(metin)
        adlar = self._degiskenler(ana)
        if len(adlar) != 2:
            raise _hata("Söz dizimi: oneway bağımlı grup")
        sonuc = anova(self._veri_gerekli().loc[self._ornek(kosul, aralik)], adlar[0], adlar[1])
        self._yaz(sonuc)
        return sonuc

    def _k_newey(self, metin: str):
        from .regresyon import ekk

        ana, kosul, aralik, sec = self._parcala(metin)
        if "tsset" not in self._veri_gerekli().attrs:
            raise _hata("newey için önce zaman yapısını tanımla: tsset zaman")
        lag = _secenek(sec, "lag", 1)
        if not isinstance(lag, str):
            raise _hata("newey için gecikme sayısı zorunlu: newey y x, lag(2)")
        formul, d = self._formul(ana, kosul, aralik, sabit=_secenek(sec, "noconstant", 3) is None)
        m = ekk(formul, d, kovaryans="newey", gecikme=int(lag))
        m._zaman_serisi = True
        return self._kaydet_sonuc(m, "newey " + metin)

    def _k_prais(self, metin: str):
        from .klasik import prais

        ana, kosul, aralik, sec = self._parcala(metin)
        if "tsset" not in self._veri_gerekli().attrs:
            raise _hata("prais için önce zaman yapısını tanımla: tsset zaman")
        formul, d = self._formul(ana, kosul, aralik, sabit=_secenek(sec, "noconstant", 3) is None)
        kov, _ = self._vce(sec)
        m = prais(formul, d, corc=bool(sec.get("corc")), kovaryans="HC1" if kov == "saglam" else "klasik")
        self._kaydet_sonuc(m, "prais " + metin)
        self._yaz(
            f"ρ = {sayi(m.rho, 4)} ({m.yineleme} yineleme)   Durbin-Watson: ilk = {sayi(m.dw_ilk, 4)}, dönüştürülmüş = {sayi(m.dw_donusmus, 4)}"
        )
        return m

    def _k_xttest0(self, metin: str):
        from .panel import PanelSonuc, xttest0

        sonuc = xttest0(self._model_gerekli(PanelSonuc))
        self._yaz(sonuc)
        return sonuc

    # ------------------------------------------------------------------ seçim, sayım ve sansürlü modeller
    def _ml_model(self, metin: str, ad: str, **ek):
        from . import secim

        ana, kosul, aralik, sec = self._parcala(metin)
        formul, d = self._formul(ana, kosul, aralik)
        kov, kume = self._vce(sec)
        if kov in ("HC2", "HC3"):
            raise _hata(f"{ad} için vce(robust) ya da vce(cluster değişken) kullan.")
        if kume:
            kume = self._ad_coz(kume)
            d[kume] = self.veri.loc[d.index, kume]
        if ad in ("poisson", "nbreg"):
            maruz = _secenek(sec, "exposure", 3)
            if isinstance(maruz, str):
                maruz = self._ad_coz(maruz)
                d[maruz] = self.veri.loc[d.index, maruz]
                ek["maruziyet"] = maruz
        if ad == "mlogit":
            temel = _secenek(sec, "baseoutcome", 2)
            if isinstance(temel, str):
                ek["temel"] = float(temel) if re.fullmatch(r"-?[\d.]+", temel) else temel
        if ad == "tobit":
            y = d.eval(formul.split("~")[0]) if formul.split("~")[0].strip() in d.columns else None
            for anahtar, hedef, uc in (("ll", "alt", "min"), ("ul", "ust", "max")):
                v = sec.get(anahtar)
                if v is True:
                    if y is None:
                        raise _hata(f"{anahtar} için sınırı sayı olarak yaz: {anahtar}(0)")
                    ek[hedef] = float(getattr(y, uc)())
                elif isinstance(v, str):
                    ek[hedef] = float(v)
                elif hedef == "alt":
                    ek[hedef] = None
            if ek.get("alt") is None and ek.get("ust") is None:
                raise _hata("tobit için sansür sınırını ver: tobit y x, ll(0)  ya da  ul(100)")
        m = getattr(secim, ad)(formul, d, kovaryans=kov, kume=kume, **ek)
        if sec.get("or") or sec.get("rrr") or sec.get("irr"):
            self._kaydet_sonuc(m, f"{ad} " + metin)
            self._yaz(m.oranlar().to_string(float_format=lambda x: sayi(x, 4)))
            return m
        return self._kaydet_sonuc(m, f"{ad} " + metin)

    def _k_ologit(self, metin: str):
        return self._ml_model(metin, "ologit")

    def _k_oprobit(self, metin: str):
        return self._ml_model(metin, "oprobit")

    def _k_mlogit(self, metin: str):
        return self._ml_model(metin, "mlogit")

    def _k_poisson(self, metin: str):
        return self._ml_model(metin, "poisson")

    def _k_nbreg(self, metin: str):
        return self._ml_model(metin, "nbreg")

    def _k_tobit(self, metin: str):
        return self._ml_model(metin, "tobit")

    # ------------------------------------------------------------------ zaman serisi
    def _ts_verisi(self, kosul=None, aralik=None) -> pd.DataFrame:
        v = self._veri_gerekli()
        if "tsset" not in v.attrs:
            raise _hata("Önce zaman yapısını tanımla: tsset zaman")
        d = v.loc[self._ornek(kosul, aralik)].copy()  # tsset veriyi zamana göre sıralı tutar
        d.attrs = v.attrs
        return d

    @staticmethod
    def _gecikme_listesi(deger) -> int | None:
        """lags(1/2) ya da lags(2) → en büyük gecikme; lags(1 2) gibi ardışık olmayanlar desteklenmez."""
        if not isinstance(deger, str):
            return None
        liste = _sayi_listesi(deger)
        if sorted(liste) != list(range(1, int(max(liste)) + 1)) and len(liste) > 1:
            raise _hata("Yalnızca ardışık gecikmeler desteklenir: lags(1/2)")
        return int(max(liste))

    def _k_var(self, metin: str):
        from .vektor import var

        ana, kosul, aralik, sec = self._parcala(metin)
        d = self._ts_verisi(kosul, aralik)
        p = self._gecikme_listesi(_secenek(sec, "lags", 3)) or 2  # Stata varsayılanı lags(1/2)
        m = var(d, [self._ad_coz(x) for x in self._degiskenler(ana)], gecikme=p, sabit=_secenek(sec, "noconstant", 3) is None)
        return self._kaydet_sonuc(m, "var " + metin)

    def _k_varsoc(self, metin: str):
        from .vektor import var_gecikme_secimi

        ana, kosul, aralik, sec = self._parcala(metin)
        mx = _secenek(sec, "maxlag", 4)
        t = var_gecikme_secimi(
            self._ts_verisi(kosul, aralik), [self._ad_coz(x) for x in self._degiskenler(ana)], int(mx) if isinstance(mx, str) else 4
        )
        g = t.copy()
        for olcut, lag in t.attrs["secim"].items():
            if olcut in g.columns:
                g[olcut] = [sayi(x, 4) + ("*" if i == lag else "") for i, x in zip(g.index, t[olcut])]
        g["LL"] = [sayi(x, 3) for x in t["LL"]]
        g["LR"] = ["" if np.isnan(x) else sayi(x, 3) + ("*" if i == t.attrs["secim"]["LR"] else "") for i, x in zip(t.index, t["LR"])]
        g["sd"] = ["" if np.isnan(x) else str(int(x)) for x in t["sd"]]
        g["p"] = ["" if np.isnan(x) else p_yaz(x) for x in t["p"]]
        self._yaz(f"Gecikme seçim ölçütleri (örneklem T = {t.attrs['T']}); * ölçütün seçtiği gecikme\n" + g.to_string())
        return t

    def _k_varstable(self, metin: str):
        from .vektor import VARSonucu

        t = self._model_gerekli(VARSonucu).kararlilik()
        self._yaz(
            "Eşlik matrisinin özdeğerleri\n"
            + t.to_string(float_format=lambda x: sayi(x, 4))
            + (
                "\nTüm özdeğerler birim çemberin içinde: VAR kararlılık koşulunu sağlıyor."
                if t.attrs["kararli"]
                else "\nVAR kararlı DEĞİL."
            )
        )
        return t

    def _k_vargranger(self, metin: str):
        from .vektor import VARSonucu
        from .zaman import var_granger

        v = self._model_gerekli(VARSonucu)
        d = self._ts_verisi()
        if len(v.sutunlar) == 2:
            from .zaman import granger

            a, b = v.sutunlar
            sonuc = granger(d[a], d[b], gecikme=v.gecikme, cift=True)
        else:
            sonuc = var_granger(d, v.sutunlar, gecikme=v.gecikme)
        self._yaz(sonuc)
        return sonuc

    def _k_irf(self, metin: str):
        from .vektor import VARSonucu

        alt, _, kalan = metin.strip().partition(" ")
        v = self._model_gerekli(VARSonucu)
        ana, _, _, sec = self._parcala(kalan)
        adim = _secenek(sec, "step", 3)
        adim = int(adim) if isinstance(adim, str) else 8  # Stata varsayılanı step(8)
        tur = (ana.split() or ["oirf"])[0].lower()
        if tur not in ("oirf", "irf"):
            raise _hata("Desteklenen: irf graph oirf | irf graph irf | irf table oirf  [, step(#)]")
        if alt.lower() in ("graph", "gr"):
            self._son_sekil = v.etki_tepki_grafigi(adim, ortogonal=tur == "oirf")
            self._yaz(f"Grafik hazır ({tur}, {adim} adım). Kaydetmek için: graph export dosya.png")
            return self._son_sekil
        if alt.lower() in ("table", "tab"):
            it = v.etki_tepki(adim, ortogonal=tur == "oirf")
            t = pd.DataFrame({f"{e} → {r}": s for (e, r), s in it.items()})
            t.index.name = "adım"
            self._yaz(t.to_string(float_format=lambda x: sayi(x, 4)))
            return t
        raise _hata("Söz dizimi: var y1 y2 … ardından irf graph oirf [, step(10)]  (irf create gerekmez)")

    def _k_ardl(self, metin: str):
        from .ardl import ardl_sinir

        ana, kosul, aralik, sec = self._parcala(metin)
        d = self._ts_verisi(kosul, aralik)
        adlar = [self._ad_coz(x) for x in self._degiskenler(ana)]
        if len(adlar) < 2:
            raise _hata("Söz dizimi: ardl y x1 x2 [, lags(#) maxlags(#) aic]")
        lags = _secenek(sec, "lags", 3)
        mx = _secenek(sec, "maxlags", 4)
        m = ardl_sinir(
            f"{adlar[0]} ~ {' + '.join(adlar[1:])}",
            d,
            gecikme=int(lags.split()[0]) if isinstance(lags, str) else None,
            en_fazla_gecikme=int(mx) if isinstance(mx, str) else 4,
            bilgi_olcutu="aic" if sec.get("aic") else "bic",  # Stata ardl varsayılanı BIC
        )
        return self._kaydet_sonuc(m, "ardl " + metin)

    def _k_vecrank(self, metin: str):
        from .johansen import johansen

        ana, kosul, aralik, sec = self._parcala(metin)
        lags = _secenek(sec, "lags", 3)
        p = int(lags) if isinstance(lags, str) else 2  # Stata: düzey VAR gecikmesi (varsayılan 2)
        trend = str(_secenek(sec, "trend", 3) or "constant").lower()
        det = {"constant": "sabit", "none": "yok", "trend": "trend", "rconstant": "sabit", "rtrend": "trend"}.get(trend)
        if det is None:
            raise _hata("trend() seçenekleri: constant (varsayılan), none, trend")
        sonuc = johansen(
            self._ts_verisi(kosul, aralik), [self._ad_coz(x) for x in self._degiskenler(ana)], gecikme=p - 1, deterministik=det
        )
        self._yaz(sonuc)
        return sonuc

    # ------------------------------------------------------------------ panel: dinamik ve birim kök
    def _k_xtabond(self, metin: str):
        from .dinamik import xtabond

        ana, kosul, aralik, sec = self._parcala(metin)
        xt = self._veri_gerekli().attrs.get("xtset")
        if xt is None:
            raise _hata("xtabond için önce panel yapısını tanımla: xtset birim zaman")
        formul, d = self._formul(ana, kosul, aralik)
        for c in xt:
            d[c] = self.veri.loc[d.index, c]
        kov, kume = self._vce(sec)
        if kume or kov in ("HC2", "HC3"):
            raise _hata("xtabond için vce(robust) kullan.")
        lags = _secenek(sec, "lags", 3)
        m = xtabond(formul, d, birim=xt[0], zaman=xt[1], gecikme=int(lags) if isinstance(lags, str) else 1, kovaryans=kov)
        return self._kaydet_sonuc(m, "xtabond " + metin)

    def _k_xtunitroot(self, metin: str):
        from .panel_kok import panel_birim_kok

        test, _, kalan = metin.strip().partition(" ")
        test = test.lower()
        if test not in ("llc", "ips", "fisher"):
            raise _hata("Desteklenen: xtunitroot llc | ips | fisher değişken [, lags(# | aic #) trend noconstant demean]")
        ana, kosul, aralik, sec = self._parcala(kalan)
        v = self._veri_gerekli()
        xt = v.attrs.get("xtset")
        if xt is None:
            raise _hata("xtunitroot için önce panel yapısını tanımla: xtset birim zaman")
        if test == "fisher" and not sec.get("dfuller"):
            raise _hata("Fisher testinde birim testi olarak dfuller kullanılır: xtunitroot fisher y, dfuller lags(1)")
        d = v.loc[self._ornek(kosul, aralik)].copy()
        d.attrs = v.attrs
        lags = _secenek(sec, "lags", 3)
        gecikme, en_fazla = 0, None
        if isinstance(lags, str):
            parca = lags.split()
            if parca[0].lower() in ("aic", "bic", "hqic"):
                gecikme = parca[0].lower()
                en_fazla = int(parca[1]) if len(parca) > 1 else None
            else:
                gecikme = int(parca[0])
        det = "trend" if sec.get("trend") else ("yok" if _secenek(sec, "noconstant", 3) else "sabit")
        sonuc = panel_birim_kok(
            d, self._ad_coz(ana.split()[0]), test, xt[0], xt[1], gecikme, en_fazla, det, ortalamadan_arindir=bool(sec.get("demean"))
        )
        self._yaz(sonuc)
        return sonuc

    # ------------------------------------------------------------------ lincom, kırılma, yatay kesit, CIPS, Toda-Yamamoto
    def _stata_terim(self, ifade: str, adlar: list[str]) -> str:
        """Stata terim adlarını model adlarına çevir: 2.bolge → C(bolge)[T.<etiket>], c.x#c.x → I(x ** 2), gen'li değişkenler."""
        etiketler = (self.veri.attrs.get("deger_etiketleri", {}) if self.veri is not None else {}) or {}

        def faktor(m):
            kod, ad = m.group(1), self._ad_coz(m.group(2))
            adaylar = [str(kod), f"{float(kod)}"]
            if ad in etiketler and int(kod) in etiketler[ad]:
                adaylar.insert(0, str(etiketler[ad][int(kod)]))
            for a in adlar:
                if (a.startswith(f"C({ad})[") or a.startswith(f"C({ad},")) and any(a.endswith(f"[T.{c}]") for c in adaylar):
                    return a
            raise _hata(f"{m.group(0)} modelde yok (temel düzey olabilir). Terimler: {adlar}")

        ifade = re.sub(r"_b\[\s*([^\]]+?)\s*\]", r"\1", ifade)  # _b[x] → x (sonra ad eşlemesinden geçer)
        ifade = re.sub(r"(?<![\w.])c\.([A-Za-z_]\w*)#c\.\1(?!\w)", lambda m: f"I({self._ad_coz(m.group(1))} ** 2)", ifade)
        ifade = re.sub(r"(?<![\w.\[])(\d+)b?\.([A-Za-z_]\w*)", faktor, ifade)
        return re.sub(r"(?<![\w.(\[])([A-Za-z_]\w*)(?![\w(\[])", lambda m: self._donusumler.get(m.group(1), m.group(1)), ifade)

    def _k_lincom(self, metin: str):
        from .kombinasyon import lincom

        m = self._model_gerekli()
        ana, _, _, sec = self._parcala(metin)
        if not ana.strip():
            raise _hata("Söz dizimi: lincom x1 + 2*x2  (ör. lincom egitim - tecrube, lincom _cons + 2.bolge)")
        adlar = list(getattr(m, "katsayilar", pd.Series(dtype=float)).index)
        guven = _secenek(sec, "level", 3)
        sonuc = lincom(m, self._stata_terim(ana, adlar), guven=float(guven) / 100 if isinstance(guven, str) else 0.95)
        sonuc.ifade = ana.strip()
        self._yaz(sonuc)
        return sonuc

    def _k_zandrews(self, metin: str):
        from .kirilma import zivot_andrews

        ana, kosul, aralik, sec = self._parcala(metin)
        d = self._ts_verisi(kosul, aralik)
        ad = self._ad_coz(ana.split()[0])
        kirilma = str(_secenek(sec, "break", 3) or "intercept").lower()
        model = {"intercept": "sabit", "trend": "trend", "both": "ikisi"}.get(kirilma)
        if model is None:
            raise _hata("break() seçenekleri: intercept (varsayılan), trend, both")
        yontem = _secenek(sec, "lagmethod", 4)
        lags = _secenek(sec, "lags", 3)
        mx = _secenek(sec, "maxlags", 4)
        if isinstance(lags, str):
            gecikme = int(lags)
        elif isinstance(yontem, str):
            gecikme = {"aic": "aic", "bic": "bic", "ttest": "t", "input": None}.get(yontem.lower())
            if gecikme is None:
                raise _hata("lagmethod(): aic, bic ya da ttest; sabit gecikme için lags(#)")
        else:
            gecikme = "aic"
        trim = _secenek(sec, "trim", 3)
        z = d.attrs["tsset"]
        seri = d.set_index(z)[ad] if isinstance(z, str) and z in d.columns else d[ad]
        sonuc = zivot_andrews(
            seri, model, gecikme, int(mx) if isinstance(mx, str) else None, float(trim) if isinstance(trim, str) else 0.15, ad=ad
        )
        self._yaz(sonuc)
        if sec.get("graph"):
            self._son_sekil = sonuc.grafik()
        return sonuc

    def _k_xtcd(self, metin: str):
        from .kesit import yatay_kesit_bagimliligi

        ana, kosul, aralik, _ = self._parcala(metin)
        v = self._veri_gerekli()
        xt = v.attrs.get("xtset")
        if xt is None:
            raise _hata("xtcd için önce panel yapısını tanımla: xtset birim zaman")
        d = v.loc[self._ornek(kosul, aralik)].copy()
        d.attrs = v.attrs
        sonuclar = [yatay_kesit_bagimliligi(d, self._ad_coz(a), xt[0], xt[1]) for a in self._degiskenler(ana)]
        for s_ in sonuclar:
            self._yaz(s_)
        return sonuclar[0] if len(sonuclar) == 1 else sonuclar

    def _k_xtcsd(self, metin: str):
        from .kesit import yatay_kesit_bagimliligi
        from .panel import PanelSonuc

        m = self._model_gerekli(PanelSonuc)
        xt = self.veri.attrs.get("xtset")
        sonuc = yatay_kesit_bagimliligi(self.veri, m, xt[0], xt[1])
        self._yaz(sonuc)
        return sonuc

    def _k_pescadf(self, metin: str):
        from .kesit import cips

        ana, kosul, aralik, sec = self._parcala(metin)
        v = self._veri_gerekli()
        xt = v.attrs.get("xtset")
        if xt is None:
            raise _hata("pescadf için önce panel yapısını tanımla: xtset birim zaman")
        lags = _secenek(sec, "lags", 3)
        if not isinstance(lags, str):
            raise _hata("Gecikme sayısını ver: pescadf y, lags(1)")
        d = v.loc[self._ornek(kosul, aralik)].copy()
        d.attrs = v.attrs
        det = "trend" if sec.get("trend") else ("yok" if _secenek(sec, "noconstant", 3) else "sabit")
        sonuc = cips(d, self._ad_coz(ana.split()[0]), xt[0], xt[1], int(lags), det, kesik=bool(sec.get("truncate") or sec.get("kesik")))
        self._yaz(sonuc)
        return sonuc

    def _k_todayamamoto(self, metin: str):
        from .nedensellik import toda_yamamoto

        ana, kosul, aralik, sec = self._parcala(metin)
        d = self._ts_verisi(kosul, aralik)
        lags = _secenek(sec, "lags", 3)
        dmax = _secenek(sec, "dmax", 2)
        mx = _secenek(sec, "maxlag", 4)
        olcut = next((o for o in ("aic", "sbic", "hqic", "fpe") if sec.get(o)), "aic")
        sonuc = toda_yamamoto(
            d,
            [self._ad_coz(a) for a in self._degiskenler(ana)],
            gecikme=int(lags) if isinstance(lags, str) else None,
            dmax=int(dmax) if isinstance(dmax, str) else 1,
            bilgi_olcutu=olcut,
            en_fazla_gecikme=int(mx) if isinstance(mx, str) else None,
            trend=bool(sec.get("trend")),
        )
        self._yaz(sonuc)
        return sonuc

    # ------------------------------------------------------------------ grafikler
    def _grafik(self, sekil):
        self._son_sekil = sekil
        self._yaz(f"Grafik hazır: {sekil.baslik or 'şekil'}. Kaydetmek için: graph export dosya.png  (Word'e: rapor komutu)")
        return sekil

    def _k_rvfplot(self, metin: str):
        from .grafik import kalinti_grafigi
        from .regresyon import EKKSonuc

        return self._grafik(kalinti_grafigi(self._model_gerekli(EKKSonuc)))

    def _k_qnorm(self, metin: str):
        from types import SimpleNamespace

        from .grafik import normallik_grafigi
        from .regresyon import EKKSonuc

        ana = self._parcala(metin)[0].strip()
        if ana:
            seri = self._veri_gerekli()[self._ad_coz(ana.split()[0])].dropna()
            return self._grafik(normallik_grafigi(SimpleNamespace(kalintilar=seri), baslik=f"{ana.split()[0]}: normal olasılık grafiği"))
        return self._grafik(normallik_grafigi(self._model_gerekli(EKKSonuc)))

    def _k_coefplot(self, metin: str):
        from .grafik import katsayi_grafigi

        adlar = self._parcala(metin)[0].split()
        eksik = [a for a in adlar if a not in self.saklanan]
        if eksik:
            raise _hata(f"Saklanmamış tahmin(ler): {eksik}. Önce: estimates store ad")
        modeller = [self.saklanan[a] for a in adlar] or [self._model_gerekli()]
        return self._grafik(katsayi_grafigi(*modeller, adlar=adlar or None))

    def _k_marginsplot(self, metin: str):
        from .grafik import marjinal_grafik

        if getattr(self, "_son_marj", None) is None:
            raise _hata("Önce margins çalıştır: margins, at(egitim=(8(2)16))")
        return self._grafik(marjinal_grafik(self._son_marj))

    def _k_tsline(self, metin: str):
        from .grafik import zaman_grafigi

        ana, kosul, aralik, _ = self._parcala(metin)
        d = self._ts_verisi(kosul, aralik)
        z = d.attrs["tsset"] if isinstance(d.attrs["tsset"], str) else None
        return self._grafik(zaman_grafigi(d, [self._ad_coz(x) for x in self._degiskenler(ana)], zaman=z))

    def _k_histogram(self, metin: str):
        from .grafik import histogram

        ana, kosul, aralik, sec = self._parcala(metin)
        d = self._veri_gerekli().loc[self._ornek(kosul, aralik)]
        bins = _secenek(sec, "bin", 3)
        return self._grafik(
            histogram(
                d, self._ad_coz(ana.split()[0]), normal=bool(_secenek(sec, "normal", 4)), kutu=int(bins) if isinstance(bins, str) else None
            )
        )

    def _k_scatter(self, metin: str, dogru: bool = False):
        from .grafik import serpme

        ana, kosul, aralik, _ = self._parcala(metin)
        adlar = [self._ad_coz(x) for x in ana.split()]
        if len(adlar) != 2:
            raise _hata("Söz dizimi: scatter y x  ya da  twoway (scatter y x) (lfit y x)")
        d = self._veri_gerekli().loc[self._ornek(kosul, aralik)]
        return self._grafik(serpme(d, adlar[0], adlar[1], dogru=dogru))

    def _k_twoway(self, metin: str):
        parcalar = re.findall(r"\(([^()]*)\)", metin)
        if not parcalar:
            parcalar = [metin]
        turler = [p.split()[0].lower() for p in parcalar if p.split()]
        sc = [p for p in parcalar if p.split() and p.split()[0].lower() in ("scatter", "sc")]
        if not sc:
            raise _hata("Desteklenen: twoway (scatter y x) (lfit y x)")
        return self._k_scatter(sc[0].split(None, 1)[1], dogru="lfit" in turler)

    def _k_lroc(self, metin: str):
        from .grafik import roc_grafigi
        from .ikili import IkiliSonuc

        m = self._model_gerekli(IkiliSonuc)
        self._yaz(f"ROC eğrisi altındaki alan = {sayi(m.roc().attrs['auc'], 4)}")
        return self._grafik(roc_grafigi(m))

    def _k_graph(self, metin: str):
        alt, _, kalan = metin.strip().partition(" ")
        if alt.lower() not in ("export", "save"):
            raise _hata("Desteklenen: graph export dosya.png [, replace width(#)]  (.png, .pdf, .svg, .jpg)")
        if getattr(self, "_son_sekil", None) is None:
            raise _hata("Kaydedilecek grafik yok; önce bir grafik çiz (rvfplot, coefplot, marginsplot, tsline ...).")
        ana, _, _, sec = self._parcala(kalan)
        dosya = Path(ana.strip().strip('"'))
        if dosya.exists() and not sec.get("replace"):
            raise _hata(f"{dosya} zaten var; üzerine yazmak için: graph export {dosya}, replace")
        self._son_sekil.kaydet(dosya)
        self._yaz(f"Grafik kaydedildi: {dosya}")
        return dosya

    _k_gr = _k_graph
    _k_sc = _k_scatter
    _k_tw = _k_twoway
    _k_hist = _k_histogram

    # ------------------------------------------------------------------ Türkçe ekler
    def _k_yorum(self, metin: str):
        m = self._model_gerekli()
        if not hasattr(m, "yorumla"):
            if hasattr(m, "_karar_cumle"):
                self._yaz(m._karar_cumle())
                return m._karar_cumle()
            raise _hata(f"{type(m).__name__} için Türkçe yorum yok; sonuç tablosundaki karar satırına bak.")
        y = m.yorumla()
        self._yaz(y)
        return y

    def _k_varsayimlar(self, metin: str):
        from .regresyon import EKKSonuc

        r = self._model_gerekli(EKKSonuc).varsayimlar()
        self._yaz(r)
        return r

    def _k_rapor(self, metin: str):
        ana, _, _, sec = self._parcala(metin)
        dosya = ana.strip().strip('"') or None
        m = self._model_gerekli()
        if not hasattr(m, "rapor"):
            raise _hata(
                f"{type(m).__name__} için tez raporu henüz yok. Sonucu tablo olarak kaydetmek için esttab ya da grafik için graph export kullan."
            )
        r = m.rapor(dosya) if dosya else m.rapor()
        self._yaz(f"rapor kaydedildi: {dosya}" if dosya else r)
        return r


def stata(komutlar: str, veri: pd.DataFrame | None = None, yazdir: bool = True):
    """Tek seferlik kullanım: ``ek.stata("regress y x, vce(robust)", veri)``. Son komutun sonucunu döndürür."""
    return Stata(veri, yazdir=yazdir)(komutlar)


__all__ = ["Stata", "stata", "KOMUTLAR"]
