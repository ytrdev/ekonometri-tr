"""Tez tablolarının temel taşları: modelleri yan yana karşılaştırma, korelasyon matrisi, tanımlayıcı istatistikler.

>>> m1 = ek.ekk("np.log(ucret) ~ egitim", veri)
>>> m2 = ek.ekk("np.log(ucret) ~ egitim + tecrube + I(tecrube**2)", veri)
>>> ek.karsilastir(m1, m2)                       # Model 1 – Model 2 yan yana
>>> ek.karsilastir(m1, m2).rapor("modeller.docx")
>>> ek.korelasyon(veri, ["ucret", "egitim", "tecrube"])
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ._yardimci import DIL, l, sayi, tablo_sayi, yildiz
from .rapor import Rapor, Tablo, _Adlar

YILDIZ_NOTU = "*** p < 0,01; ** p < 0,05; * p < 0,10."
_YILDIZ_EN = "*** p < 0.01; ** p < 0.05; * p < 0.10."
_UYUM_EN = {
    "Gözlem sayısı": "Observations",
    "R²": "R²",
    "Düzeltilmiş R²": "Adjusted R²",
    "F": "F",
    "AIC": "AIC",
    "BIC": "BIC",
    "Birim sayısı": "Number of units",
    "R² (içsel)": "R² (within)",
    "R² (birimler arası)": "R² (between)",
    "R² (genel)": "R² (overall)",
    "Birim etkileri": "Unit effects",
    "Zaman etkileri": "Time effects",
    "McFadden sözde R²": "McFadden pseudo R²",
    "LR χ²": "LR χ²",
    "Log-olabilirlik": "Log-likelihood",
}
_MODEL_EN = {
    "EKK": "OLS",
    "Havuzlanmış EKK": "Pooled OLS",
    "Sabit etkiler": "Fixed effects",
    "Rassal etkiler": "Random effects",
    "Logit": "Logit",
    "Probit": "Probit",
    "2SLS": "2SLS",
}


def _satir_etiketi(anahtar: str) -> str:
    if DIL.get() != "en":
        return anahtar
    if anahtar in _UYUM_EN:
        return _UYUM_EN[anahtar]
    if anahtar.startswith("1. aşama F ("):
        return "First-stage F (" + anahtar[len("1. aşama F (") :]
    return anahtar


def _model_etiketi(tur: str) -> str:
    return _MODEL_EN.get(tur, tur) if DIL.get() == "en" else tur


def _kov_en(acik: str) -> str:
    if acik == "klasik":
        return "classical"
    if acik == "klasik (homoskedastik)":
        return "classical (homoskedastic)"
    if acik == "klasik (en çok olabilirlik, bilgi matrisi)":
        return "classical (maximum likelihood, information matrix)"
    if acik == "dayanıklı (sandviç / Huber-White)":
        return "robust (sandwich / Huber-White)"
    if acik == "değişen varyansa dayanıklı (robust)":
        return "heteroskedasticity-robust"
    if acik.startswith("değişen varyansa dayanıklı ("):
        return "heteroskedasticity-robust (" + acik[len("değişen varyansa dayanıklı (") :]
    onek = "değişen varyans ve otokorelasyona dayanıklı (HAC / Newey-West, "
    if acik.startswith(onek) and acik.endswith(" gecikme)"):
        return f"heteroskedasticity- and autocorrelation-robust (HAC / Newey-West, {acik[len(onek) : -len(' gecikme)')]} lags)"
    if acik.startswith("Newey-West (") and acik.endswith(" gecikme)"):
        return f"Newey-West ({acik[len('Newey-West (') : -len(' gecikme)')]} lags)"
    if acik.startswith("kümelenmiş, "):
        return "clustered, " + acik[len("kümelenmiş, ") :].replace(" küme", " clusters", 1)
    onek = "birimlere göre kümelenmiş ("
    if acik.startswith(onek) and acik.endswith(")"):
        ic = acik[len(onek) : -1].replace(
            " küme; değişen varyans ve birim içi otokorelasyona dayanıklı",
            " clusters; robust to heteroskedasticity and within-unit autocorrelation",
        )
        return f"clustered by unit ({ic})"
    return acik


# --------------------------------------------------------------------------- model karşılaştırma
@dataclass
class Karsilastirma:
    """Birden çok modelin yan yana tablosu. ``print()`` ile okunur, ``.rapor("dosya.docx")`` ile Word'e aktarılır."""

    modeller: list
    adlar: list[str]

    def _model_etiketleri(self) -> dict:
        return {k: v for m in self.modeller for k, v in getattr(m, "_etiketler", {}).items()}

    def _tablo(self, etiketler: dict | None = None, no: int = 1) -> Tablo:
        ad = _Adlar({**self._model_etiketleri(), **(etiketler or {})})
        sira: dict[str, str] = {}  # sütun adı -> görünen ad (ilk görülme sırası, sabit en sonda)
        for m in self.modeller:
            for t in m._terimler:
                if t.tur != "sabit":
                    sira.setdefault(t.sutun, ad.satir(t))
        if any(t.tur == "sabit" for m in self.modeller for t in m._terimler):
            sira["Intercept"] = l("Sabit", "Constant")
        satirlar = []
        for sutun, gorunen in sira.items():
            ust, alt = [gorunen], [""]
            for m in self.modeller:
                if sutun in m.katsayilar.index:
                    ust.append(tablo_sayi(m.katsayilar[sutun]) + yildiz(m.p_degerleri[sutun]))
                    alt.append(f"({tablo_sayi(m.standart_hatalar[sutun])})")
                else:
                    ust.append("")
                    alt.append("")
            satirlar += [ust, alt]
        uyumlar = [m._uyum() for m in self.modeller]
        for anahtar in dict.fromkeys(k for u in uyumlar for k in u):
            satirlar.append([_satir_etiketi(anahtar)] + [u.get(anahtar, "") for u in uyumlar])
        from .regresyon import _log_mu

        def bag(m):
            lg = _log_mu(m.bagimli)
            return f"ln({ad.yalin(lg)})" if lg else ad.yalin(m.bagimli)

        bagimlilar = list(dict.fromkeys(bag(m) for m in self.modeller))
        turler = list(dict.fromkeys(_model_etiketi(m.model_turu) for m in self.modeller))
        kovlar = list(dict.fromkeys(m.kovaryans_aciklama for m in self.modeller))
        if DIL.get() == "en":
            kovlar = list(dict.fromkeys(_kov_en(m.kovaryans_aciklama) for m in self.modeller))
            notu = "Values in parentheses are standard errors. "
            if len(bagimlilar) == 1:
                notu += f"Dependent variable: {bagimlilar[0]}. "
            else:
                notu += "Dependent variables: " + "; ".join(f"{a} = {bag(m)}" for a, m in zip(self.adlar, self.modeller)) + ". "
            if len(turler) == 1:
                notu += f"Estimator: {turler[0]}. "
            elif [_model_etiketi(m.model_turu) for m in self.modeller] != list(self.adlar):
                notu += (
                    "Estimators: " + "; ".join(f"{a} = {_model_etiketi(m.model_turu)}" for a, m in zip(self.adlar, self.modeller)) + ". "
                )
            if len(kovlar) == 1:
                notu += f"Standard errors: {kovlar[0]}. "
            notu += _YILDIZ_EN
            baslik = "Comparison of model estimates"
            ilk = "Variable"
        else:
            notu = "Parantez içindeki değerler standart hatalardır. "
            if len(bagimlilar) == 1:
                notu += f"Bağımlı değişken: {bagimlilar[0]}. "
            else:
                notu += "Bağımlı değişkenler: " + "; ".join(f"{a} = {bag(m)}" for a, m in zip(self.adlar, self.modeller)) + ". "
            if len(turler) == 1:
                notu += f"Tahmin yöntemi: {turler[0]}. "
            elif [m.model_turu for m in self.modeller] != list(self.adlar):
                notu += "Tahmin yöntemleri: " + "; ".join(f"{a} = {m.model_turu}" for a, m in zip(self.adlar, self.modeller)) + ". "
            if len(kovlar) == 1:
                notu += f"Standart hatalar: {kovlar[0]}. "
            notu += YILDIZ_NOTU
            baslik = "Model tahmin sonuçlarının karşılaştırılması"
            ilk = "Değişken"
        return Tablo(no, baslik, [ilk] + self.adlar, satirlar, notu)

    def tablo(self, etiketler: dict | None = None) -> pd.DataFrame:
        t = self._tablo(etiketler)
        return pd.DataFrame(t.satirlar, columns=t.sutunlar)

    def __str__(self) -> str:
        t = self._tablo()
        g0 = max(len(s[0]) for s in t.satirlar + [t.sutunlar]) + 2
        g = max(14, *(len(h) + 2 for s in t.satirlar for h in s[1:]), *(len(h) + 2 for h in t.sutunlar[1:]))
        cizgi = "─" * (g0 + g * len(self.adlar))
        govde = [f"{s[0]:<{g0}}" + "".join(f"{h:>{g}}" for h in s[1:]) for s in [t.sutunlar] + t.satirlar]
        n_kat = len(t.satirlar) - len({k for m in self.modeller for k in m._uyum()})
        return "\n".join(
            [
                "═" * len(cizgi),
                govde[0],
                cizgi,
                *govde[1 : n_kat + 1],
                cizgi,
                *govde[n_kat + 1 :],
                "═" * len(cizgi),
                "(std. hata)   " + YILDIZ_NOTU,
            ]
        )

    __repr__ = __str__

    def _repr_html_(self) -> str:
        return Rapor("Model karşılaştırması", [("tablo", self._tablo())]).html()

    def _metin(self, no: int, ad: _Adlar) -> str:
        if DIL.get() == "en":
            return self._metin_en(no, ad)
        M = self.modeller
        c = [f"Tahmin edilen {len(M)} modelin sonuçları Tablo {no} ile karşılaştırmalı olarak sunulmuştur."]
        if all(m.model_turu == "EKK" for m in M):
            c.append(
                "Düzeltilmiş R² değeri "
                + "; ".join(f"{a} için {sayi(m.duzeltilmis_r2, 3)}" for a, m in zip(self.adlar, M))
                + " olarak hesaplanmıştır."
            )
        elif all(hasattr(m, "pseudo_r2") for m in M):
            c.append(
                "McFadden sözde R² değeri "
                + "; ".join(f"{a} için {sayi(m.pseudo_r2, 3)}" for a, m in zip(self.adlar, M))
                + " olarak hesaplanmıştır."
            )
        ayni_orneklem = len({(m.bagimli, m.n) for m in M}) == 1
        if ayni_orneklem and len(M) > 1 and all(hasattr(m, "aic") for m in M):
            ia, ib = int(np.argmin([m.aic for m in M])), int(np.argmin([m.bic for m in M]))
            if ia == ib:
                c.append(
                    f"Bilgi ölçütlerine göre en düşük AIC ({sayi(M[ia].aic, 1)}) ve BIC ({sayi(M[ib].bic, 1)}) değerleri {self.adlar[ia]} ile elde edilmiştir."
                )
            else:
                c.append(
                    f"Bilgi ölçütleri farklı modellere işaret etmektedir: en düşük AIC ({sayi(M[ia].aic, 1)}) {self.adlar[ia]}, en düşük BIC ({sayi(M[ib].bic, 1)}) ise {self.adlar[ib]} ile elde edilmiştir."
                )
        ortak = [t for t in M[0]._terimler if t.tur != "sabit" and all(t.sutun in m.katsayilar.index for m in M)]
        if len(M) > 1 and ortak:
            kararli, degisken = [], []
            for t in ortak:
                b = [m.katsayilar[t.sutun] for m in M]
                p = [m.p_degerleri[t.sutun] for m in M]
                (kararli if (all(x > 0 for x in b) or all(x < 0 for x in b)) and all(x < 0.05 for x in p) else degisken).append(ad.satir(t))
            if kararli:
                c.append(
                    "Tüm modellerde yer alan terimlerden şu "
                    + ("terimin katsayısı" if len(kararli) == 1 else "terimlerin katsayıları")
                    + " her modelde aynı işaretli ve %5 düzeyinde istatistiksel olarak anlamlıdır: "
                    + "; ".join(kararli)
                    + "."
                )
            if degisken:
                c.append(
                    "Şu "
                    + ("terimin" if len(degisken) == 1 else "terimlerin")
                    + " işareti ya da anlamlılığı ise modele göre değişmektedir: "
                    + "; ".join(degisken)
                    + "."
                )
        return " ".join(c)

    def _metin_en(self, no: int, ad: _Adlar) -> str:
        M = self.modeller
        c = [f"Estimates from the {len(M)} models are compared in Table {no}."]
        if all(m.model_turu == "EKK" for m in M):
            c.append("Adjusted R² is " + "; ".join(f"{sayi(m.duzeltilmis_r2, 3)} for {a}" for a, m in zip(self.adlar, M)) + ".")
        elif all(hasattr(m, "pseudo_r2") for m in M):
            c.append("McFadden pseudo R² is " + "; ".join(f"{sayi(m.pseudo_r2, 3)} for {a}" for a, m in zip(self.adlar, M)) + ".")
        ayni_orneklem = len({(m.bagimli, m.n) for m in M}) == 1
        if ayni_orneklem and len(M) > 1 and all(hasattr(m, "aic") for m in M):
            ia, ib = int(np.argmin([m.aic for m in M])), int(np.argmin([m.bic for m in M]))
            if ia == ib:
                c.append(f"The lowest AIC ({sayi(M[ia].aic, 1)}) and BIC ({sayi(M[ib].bic, 1)}) both belong to {self.adlar[ia]}.")
            else:
                c.append(
                    f"The information criteria point to different models: the lowest AIC ({sayi(M[ia].aic, 1)}) belongs to {self.adlar[ia]}, "
                    f"and the lowest BIC ({sayi(M[ib].bic, 1)}) belongs to {self.adlar[ib]}."
                )
        ortak = [t for t in M[0]._terimler if t.tur != "sabit" and all(t.sutun in m.katsayilar.index for m in M)]
        if len(M) > 1 and ortak:
            kararli, degisken = [], []
            for t in ortak:
                b = [m.katsayilar[t.sutun] for m in M]
                p = [m.p_degerleri[t.sutun] for m in M]
                (kararli if (all(x > 0 for x in b) or all(x < 0 for x in b)) and all(x < 0.05 for x in p) else degisken).append(ad.satir(t))
            if kararli:
                yuklem = (
                    "term has the same sign and is statistically significant at the 5% level in every model"
                    if len(kararli) == 1
                    else "terms have the same sign and are statistically significant at the 5% level in every model"
                )
                c.append("Among the terms included in every model, the following " + yuklem + ": " + "; ".join(kararli) + ".")
            if degisken:
                c.append(
                    "The sign or the significance of the following "
                    + ("term" if len(degisken) == 1 else "terms")
                    + " changes across models: "
                    + "; ".join(degisken)
                    + "."
                )
        return " ".join(c)

    def kaydet(self, dosya, etiketler: dict | None = None, tablo_no: int = 1, dil: str = "tr", bicim: str = "apa"):
        """Yalnızca karşılaştırma tablosunu dosyaya yazar — Stata ``esttab m1 m2 using tablo.docx / .tex / .xlsx / .csv``.

        .tex: yalnızca tablo ortamı (kendi tez belgene ``\\input{tablo.tex}`` ile eklenir); .xlsx: Excel sayfası;
        .csv: sade tablo; .docx / .md / .html: tez biçiminde tek tablo.
        """
        from pathlib import Path

        from ._yardimci import DIL
        from .rapor import Rapor, rapor_ayari, tablo_latex, tablolari_excele_yaz

        yol = Path(dosya)
        uz = yol.suffix.lower()
        token = rapor_ayari(dil, bicim)
        try:
            tablo = self._tablo(etiketler, tablo_no)
        finally:
            DIL.reset(token)
        if uz == ".tex":
            yol.write_text(tablo_latex(tablo, dil) + "\n", encoding="utf-8")
        elif uz == ".xlsx":
            tablolari_excele_yaz([tablo], yol, dil)
        elif uz == ".csv":
            pd.DataFrame(tablo.satirlar, columns=tablo.sutunlar).to_csv(yol, index=False, encoding="utf-8-sig")
        else:
            Rapor(tablo.baslik, [("tablo", tablo)], bicim=bicim, dil=dil).kaydet(yol)
        return yol

    def rapor(
        self,
        dosya: str | None = None,
        baslik: str | None = None,
        etiketler: dict | None = None,
        tablo_no: int = 1,
        dil: str = "tr",
        bicim: str = "apa",
    ) -> Rapor:
        """Karşılaştırma tablosu + kısa akademik metin. ``dosya`` verilirse .docx / .md / .html olarak kaydedilir."""
        from .rapor import rapor_ayari

        if baslik is None:
            baslik = "Model comparison" if dil == "en" else "Model Karşılaştırması"
        token = rapor_ayari(dil, bicim)
        try:
            r = Rapor(
                baslik,
                [
                    ("paragraf", self._metin(tablo_no, _Adlar({**self._model_etiketleri(), **(etiketler or {})}))),
                    ("tablo", self._tablo(etiketler, tablo_no)),
                ],
                bicim=bicim,
                dil=dil,
            )
            if dosya is not None:
                r.kaydet(dosya)
            return r
        finally:
            DIL.reset(token)


def karsilastir(*modeller, adlar: list[str] | None = None) -> Karsilastirma:
    """Modelleri yan yana tek tabloda gösterir (katsayı + yıldız, altında parantez içinde standart hata).

    >>> ek.karsilastir(m1, m2, m3)
    >>> ek.karsilastir(m1, m2, adlar=["Temel", "Genişletilmiş"]).rapor("modeller.docx")
    """
    from .regresyon import EkonometriHatasi

    if len(modeller) == 1 and isinstance(modeller[0], (list, tuple)):
        modeller = tuple(modeller[0])
    if not modeller:
        raise EkonometriHatasi("En az bir model ver: ek.karsilastir(m1, m2, ...)")
    for m in modeller:
        if not hasattr(m, "_uyum"):
            raise EkonometriHatasi(f"Karşılaştırılamayan nesne: {type(m).__name__}. ek.ekk(), ek.logit() gibi model sonuçları ver.")
    adlar = list(adlar) if adlar else [f"Model {i}" for i in range(1, len(modeller) + 1)]
    if len(adlar) != len(modeller):
        raise EkonometriHatasi(f"{len(modeller)} model için {len(adlar)} ad verildi; sayılar eşit olmalı.")
    return Karsilastirma(list(modeller), adlar)


# --------------------------------------------------------------------------- korelasyon
@dataclass
class KorelasyonSonuc:
    """Korelasyon matrisi: ``.r`` katsayılar, ``.p`` p-değerleri, ``.n`` gözlem sayısı."""

    r: pd.DataFrame
    p: pd.DataFrame
    n: int
    yontem: str
    _etiketler: dict = field(default_factory=dict, repr=False)

    def _tablo(self, etiketler: dict | None = None, no: int = 1) -> Tablo:
        ad = _Adlar({**self._etiketler, **(etiketler or {})})
        s = list(self.r.columns)
        satirlar = []
        for i, a in enumerate(s):
            satir = [f"{i + 1}. {ad.yalin(a)}"]
            for j, b in enumerate(s):
                if j > i:
                    satir.append("")
                elif j == i:
                    satir.append("1")
                else:
                    satir.append(sayi(self.r.loc[a, b], 3) + yildiz(self.p.loc[a, b]))
            satirlar.append(satir)
        if DIL.get() == "en":
            ad_ = "Pearson" if self.yontem == "pearson" else "Spearman rank"
            return Tablo(
                no,
                f"{ad_} correlation coefficients",
                ["Variable"] + [str(i + 1) for i in range(len(s))],
                satirlar,
                f"N = {sayi(self.n, 0)}. {_YILDIZ_EN}",
            )
        ad_ = "Pearson" if self.yontem == "pearson" else "Spearman sıra"
        return Tablo(
            no,
            f"Değişkenler arasındaki {ad_} korelasyon katsayıları",
            ["Değişken"] + [str(i + 1) for i in range(len(s))],
            satirlar,
            f"N = {sayi(self.n, 0)}. {YILDIZ_NOTU}",
        )

    def tablo(self, etiketler: dict | None = None) -> pd.DataFrame:
        t = self._tablo(etiketler)
        return pd.DataFrame(t.satirlar, columns=t.sutunlar)

    def __str__(self) -> str:
        t = self._tablo()
        g0 = max(len(x[0]) for x in t.satirlar) + 2
        bas = f"{t.baslik} (N = {sayi(self.n, 0)})"
        govde = [f"{x[0]:<{g0}}" + "".join(f"{h:>12}" for h in x[1:]) for x in [t.sutunlar] + t.satirlar]
        return "\n".join([bas, "─" * len(govde[0]), *govde, "─" * len(govde[0]), YILDIZ_NOTU])

    __repr__ = __str__

    def _repr_html_(self) -> str:
        return Rapor("Korelasyon matrisi", [("tablo", self._tablo())]).html()

    def _metin(self, no: int, ad: _Adlar) -> str:
        if DIL.get() == "en":
            return self._metin_en(no, ad)
        s = list(self.r.columns)
        ciftler = [(a, b, float(self.r.loc[a, b]), float(self.p.loc[a, b])) for i, a in enumerate(s) for b in s[:i]]
        c = [f"Değişkenler arasındaki korelasyon katsayıları Tablo {no} ile sunulmuştur."]
        if ciftler:
            a, b, r, p = max(ciftler, key=lambda x: abs(x[2]))
            c.append(
                f"Mutlak değerce en yüksek korelasyon {ad.metin(a)} ile {ad.metin(b)} değişkenleri arasındadır (r = {sayi(r, 3)}; "
                + ("p < 0,001" if p < 0.001 else f"p = {sayi(p, 3)}")
                + ")."
            )
            yuksek = [x for x in ciftler if abs(x[2]) >= 0.8]
            c.append(
                "Mutlak değeri 0,80 ve üzerinde olan korelasyonlar çoklu doğrusal bağlantı açısından dikkate alınmalıdır: "
                + "; ".join(f"{ad.metin(x[0])} – {ad.metin(x[1])} (r = {sayi(x[2], 3)})" for x in yuksek)
                + "."
                if yuksek
                else "Hiçbir değişken çifti arasındaki korelasyonun mutlak değeri 0,80 düzeyine ulaşmamaktadır; bu durum ciddi bir çoklu doğrusal bağlantı sorununa işaret etmemektedir."
            )
        return " ".join(c)

    def _metin_en(self, no: int, ad: _Adlar) -> str:
        s = list(self.r.columns)
        ciftler = [(a, b, float(self.r.loc[a, b]), float(self.p.loc[a, b])) for i, a in enumerate(s) for b in s[:i]]
        c = [f"The correlation coefficients are reported in Table {no}."]
        if ciftler:
            a, b, r, p = max(ciftler, key=lambda x: abs(x[2]))
            p_metin = "p < 0.001" if p < 0.001 else f"p = {sayi(p, 3)}"
            c.append(f"The largest correlation in absolute value is between {ad.metin(a)} and {ad.metin(b)} (r = {sayi(r, 3)}; {p_metin}).")
            yuksek = [x for x in ciftler if abs(x[2]) >= 0.8]
            c.append(
                "Correlations of 0.80 or more in absolute value should be considered for multicollinearity: "
                + "; ".join(f"{ad.metin(x[0])} – {ad.metin(x[1])} (r = {sayi(x[2], 3)})" for x in yuksek)
                + "."
                if yuksek
                else "No pair has a correlation of 0.80 or more in absolute value, so the matrix does not point to a serious multicollinearity problem."
            )
        return " ".join(c)

    def rapor(
        self,
        dosya: str | None = None,
        baslik: str | None = None,
        etiketler: dict | None = None,
        tablo_no: int = 1,
        dil: str = "tr",
        bicim: str = "apa",
    ) -> Rapor:
        from .rapor import rapor_ayari

        if baslik is None:
            baslik = "Correlation analysis" if dil == "en" else "Korelasyon Analizi"
        token = rapor_ayari(dil, bicim)
        try:
            ad = _Adlar({**self._etiketler, **(etiketler or {})})
            r = Rapor(
                baslik,
                [("paragraf", self._metin(tablo_no, ad)), ("tablo", self._tablo(etiketler, tablo_no))],
                bicim=bicim,
                dil=dil,
            )
            if dosya is not None:
                r.kaydet(dosya)
            return r
        finally:
            DIL.reset(token)


def _hata_k(m):
    from .regresyon import EkonometriHatasi

    return EkonometriHatasi(m)


def _cift_korelasyon(veri, sutunlar, yontem, etiketler) -> KorelasyonSonuc:
    if yontem not in ("pearson", "spearman"):
        raise _hata_k(f"Bilinmeyen yöntem: {yontem!r}. Seçenekler: 'pearson', 'spearman'")
    s = _sayisal_sutunlar(veri, sutunlar)
    r = pd.DataFrame(np.eye(len(s)), index=s, columns=s)
    p = pd.DataFrame(np.zeros((len(s), len(s))), index=s, columns=s)
    nm = pd.DataFrame(np.zeros((len(s), len(s)), dtype=int), index=s, columns=s)
    for i, a in enumerate(s):
        nm.loc[a, a] = int(veri[a].notna().sum())
        for b in s[:i]:
            d = veri[[a, b]].dropna()
            if yontem == "spearman":
                d = d.rank()
            n = len(d)
            rr = float(np.clip(d[a].corr(d[b]), -1, 1)) if n > 2 else np.nan
            pp = float(2 * stats.t.sf(abs(rr) * np.sqrt((n - 2) / max(1 - rr**2, 1e-300)), n - 2)) if n > 2 else np.nan
            r.loc[a, b] = r.loc[b, a] = rr
            p.loc[a, b] = p.loc[b, a] = pp
            nm.loc[a, b] = nm.loc[b, a] = n
    sonuc = KorelasyonSonuc(r, p, int(nm.values.min()), yontem, {**veri.attrs.get("etiketler", {}), **(etiketler or {})})
    sonuc.n_matris = nm
    return sonuc


def _sayisal_sutunlar(veri: pd.DataFrame, sutunlar) -> list[str]:
    from .regresyon import EkonometriHatasi

    if sutunlar is None:
        sutunlar = [c for c in veri.columns if pd.api.types.is_numeric_dtype(veri[c]) and not pd.api.types.is_bool_dtype(veri[c])]
    eksik = [c for c in sutunlar if c not in veri.columns]
    if eksik:
        raise EkonometriHatasi(f"Veride olmayan sütun(lar): {eksik}. Mevcut sütunlar: {list(veri.columns)}")
    metin = [c for c in sutunlar if not pd.api.types.is_numeric_dtype(veri[c])]
    if metin:
        raise EkonometriHatasi(f"Sayısal olmayan sütun(lar): {metin}. Bu işlem yalnızca sayısal değişkenlerle yapılır.")
    return list(sutunlar)


def korelasyon(
    veri: pd.DataFrame, sutunlar: list[str] | None = None, yontem: str = "pearson", etiketler: dict | None = None, eksik: str = "liste"
) -> KorelasyonSonuc:
    """Korelasyon matrisi ve anlamlılık yıldızları.

    yontem : "pearson" (doğrusal) ya da "spearman" (sıra korelasyonu; aykırı değerlere ve doğrusal olmayan tekdüze ilişkilere dayanıklı)
    eksik : "liste" — herhangi bir değişkeni eksik olan satır çıkarılır (Stata ``correlate``);
            "cift" — her değişken çifti kendi tam gözlemleriyle hesaplanır (Stata ``pwcorr, sig``)
    """
    if eksik not in ("liste", "cift"):
        raise _hata_k(f"Bilinmeyen eksik veri seçeneği: {eksik!r}. Seçenekler: 'liste' (correlate), 'cift' (pwcorr)")
    if eksik == "cift":
        return _cift_korelasyon(veri, sutunlar, yontem, etiketler)
    from .regresyon import EkonometriHatasi

    if yontem not in ("pearson", "spearman"):
        raise EkonometriHatasi(f"Bilinmeyen yöntem: {yontem!r}. Seçenekler: 'pearson', 'spearman'")
    s = _sayisal_sutunlar(veri, sutunlar)
    v = veri[s].dropna()
    n = len(v)
    if n < 3:
        raise EkonometriHatasi("Korelasyon için en az 3 tam gözlem gerekli.")
    d = v.rank() if yontem == "spearman" else v
    r = d.corr().clip(-1, 1)
    t = r * np.sqrt((n - 2) / (1 - r**2).clip(lower=1e-300))
    pv = np.array(2 * stats.t.sf(np.abs(t.values), n - 2))
    np.fill_diagonal(pv, 0.0)
    p = pd.DataFrame(pv, index=s, columns=s)
    return KorelasyonSonuc(r, p, n, yontem, {**veri.attrs.get("etiketler", {}), **(etiketler or {})})


def _agirlikli_tanimlayici(veri: pd.DataFrame, sutunlar: list[str], w: pd.Series, tur: str) -> pd.DataFrame:
    """Stata summarize [aw=] / [fw=]: aw gözlem sayısına normalleştirilir, varyans paydası N − 1 (fw'de Σw − 1)."""
    satirlar = {}
    for c in sutunlar:
        m = veri[c].notna() & w.notna() & (w > 0)
        x, ag = veri.loc[m, c].astype(float).values, w[m].values
        n = len(x)
        if tur == "fw":
            N = ag.sum()
        else:
            ag = ag * n / ag.sum()
            N = n
        ort = float(np.sum(ag * x) / np.sum(ag)) if n else np.nan
        var = float(np.sum(ag * (x - ort) ** 2) / (N - 1)) if N > 1 else np.nan
        satirlar[c] = {
            "N": int(round(N)),
            "Ortalama": ort,
            "Std. sapma": np.sqrt(var),
            "En küçük": x.min() if n else np.nan,
            "En büyük": x.max() if n else np.nan,
        }
    t = pd.DataFrame(satirlar).T.astype(float)
    t["N"] = t["N"].astype(int)
    return t


def stata_yuzdelik(x: np.ndarray, p: float) -> float:
    """Stata ``summarize, detail`` yüzdelik tanımı: P = n·p/100; P tam sayıysa x₍P₎ ile x₍P+1₎ ortalaması, değilse x₍⌈P⌉₎."""
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    P = n * p / 100
    i = int(np.floor(P + 1e-12))
    if abs(P - i) < 1e-12:
        if i <= 0:
            return float(x[0])
        if i >= n:
            return float(x[-1])
        return float((x[i - 1] + x[i]) / 2)
    return float(x[min(i, n - 1)])


def tanimlayici(
    veri: pd.DataFrame, sutunlar: list[str] | None = None, detay: bool = False, agirlik=None, agirlik_turu: str = "aw"
) -> pd.DataFrame:
    """Tanımlayıcı istatistikler — Stata ``summarize`` ile aynı (std. sapma n − 1 ile).

    detay=True: Stata ``summarize, detail`` — yüzdelikler (1, 5, 10, 25, 50, 75, 90, 95, 99), varyans,
    çarpıklık (m₃/m₂^1,5) ve basıklık (m₄/m₂², normal dağılımda 3).
    """
    s = _sayisal_sutunlar(veri, sutunlar)
    if agirlik is not None:
        return _agirlikli_tanimlayici(veri, s, pd.Series(agirlik, index=veri.index).astype(float), agirlik_turu)
    satirlar = {}
    for c in s:
        x = veri[c].dropna().astype(float).values
        n = len(x)
        d = {
            "N": n,
            "Ortalama": x.mean() if n else np.nan,
            "Std. sapma": x.std(ddof=1) if n > 1 else np.nan,
            "En küçük": x.min() if n else np.nan,
            "En büyük": x.max() if n else np.nan,
        }
        if n:
            m = x - x.mean()
            m2, m3, m4 = np.mean(m**2), np.mean(m**3), np.mean(m**4)
            d["Medyan"] = stata_yuzdelik(x, 50)
            d["Çarpıklık"] = m3 / m2**1.5 if m2 > 0 else np.nan
            d["Basıklık"] = m4 / m2**2 if m2 > 0 else np.nan
            if detay:
                for q in (1, 5, 10, 25, 75, 90, 95, 99):
                    d[f"%{q}"] = stata_yuzdelik(x, q)
                d["Varyans"] = x.var(ddof=1) if n > 1 else np.nan
        satirlar[c] = d
    t = pd.DataFrame(satirlar).T
    sira = ["N", "Ortalama", "Std. sapma", "En küçük", "En büyük", "Medyan", "Çarpıklık", "Basıklık"]
    if detay:
        sira = [
            "N",
            "Ortalama",
            "Std. sapma",
            "Varyans",
            "En küçük",
            "%1",
            "%5",
            "%10",
            "%25",
            "Medyan",
            "%75",
            "%90",
            "%95",
            "%99",
            "En büyük",
            "Çarpıklık",
            "Basıklık",
        ]
    t = t[[c for c in sira if c in t.columns]].astype(float)
    t["N"] = t["N"].astype(int)
    return t


__all__ = ["karsilastir", "Karsilastirma", "korelasyon", "KorelasyonSonuc", "tanimlayici"]
