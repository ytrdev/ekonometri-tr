"""Toda-Yamamoto (1995) nedensellik testi: seriler I(0)/I(1) karışık ya da eşbütünleşik olsa da geçerli Granger testi.

>>> ek.toda_yamamoto(veri, ["ln_ihracat", "ln_kur"], dmax=1)                  # gecikme AIC ile
>>> ek.toda_yamamoto(veri, ["y", "x", "z"], gecikme=2, dmax=1).rapor("ty.docx")

Yöntem:
1. Düzeyde VAR'ın gecikmesi p bilgi ölçütüyle seçilir (Stata ``varsoc``; Lütkepohl tanımları, ortak örneklem).
2. VAR(p + dmax) düzeyde, her denklem EKK ile tahmin edilir (sabit, isteğe bağlı trend).
3. "x, y'nin nedeni değildir" hipotezi yalnızca x'in İLK p gecikmesine Wald χ²(p) testiyle sınanır; fazladan dmax gecikme
   sınanmaz (bu, birim kök altında Wald istatistiğinin standart χ² dağılmasını sağlar).

Kovaryans: denklem başına klasik EKK, s² = e'e / (T − m). Bu, yaygın R uygulamasıyla (vars::VAR + Wald testi, ``vcov(lm)``)
aynıdır. dmax, serilerin en yüksek bütünleşme derecesidir (çoğunlukla 1); ``ek.duraganlik()`` ile belirlenir.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from ._yardimci import DIL, l, p_cumle, p_yaz, sayi
from .regresyon import EkonometriHatasi


def _hata(m: str) -> EkonometriHatasi:
    return EkonometriHatasi(m)


@dataclass
class TodaYamamotoSonucu:
    sutunlar: list[str]
    gecikme: int  # p (sınanan gecikme)
    dmax: int
    secim: str  # "sabit" ya da bilgi ölçütü adı
    trend: bool
    T: int  # tahminde kullanılan gözlem
    tablo: pd.DataFrame  # Neden, Etkilenen, χ², sd, p

    def nedensel(self, neden: str, etkilenen: str, anlamlilik: float = 0.05) -> bool:
        t = self.tablo
        satir = t[(t["Neden"] == neden) & (t["Etkilenen"] == etkilenen)]
        if satir.empty:
            raise _hata(f"{neden} → {etkilenen} yönü tabloda yok.")
        return bool(satir["p"].iloc[0] < anlamlilik)

    def __str__(self) -> str:
        sec = f"{self.secim.upper()} ile seçildi" if self.secim != "sabit" else "verildi"
        s = [
            f"Toda-Yamamoto nedensellik testi — p = {self.gecikme} ({sec}), dmax = {self.dmax}, VAR({self.gecikme + self.dmax}) düzeyde, "
            f"T = {self.T}" + (", trendli" if self.trend else ""),
            "   H0: Neden değişkenin ilk p gecikmesi birlikte sıfırdır (Granger nedeni değildir). Wald χ²(p).",
        ]
        for _, r in self.tablo.iterrows():
            isaret = "✅" if r["p"] < 0.05 else "  "
            s.append(
                f"   {isaret} {r['Neden']:>14} → {r['Etkilenen']:<14} χ²({int(r['sd'])}) = {sayi(r['χ²'], 3):>9}   p = {p_yaz(r['p'])}"
            )
        return "\n".join(s)

    __repr__ = __str__

    def _repr_html_(self) -> str:
        return self.rapor().html()

    def rapor(self, dosya=None, baslik=None, etiketler=None, tablo_no: int = 1, dil: str = "tr", bicim: str = "apa"):
        from .rapor import KAYNAKLAR, Rapor, Tablo, _Adlar, _p_hucre, rapor_ayari

        token = rapor_ayari(dil, bicim)
        try:
            en = dil == "en"
            ad = _Adlar(etiketler)
            baslik = baslik or l("Toda-Yamamoto Nedensellik Testi", "Toda-Yamamoto causality test")
            k = self.gecikme + self.dmax
            sec_tr = f"{self.secim.upper()} bilgi ölçütüyle {self.gecikme}" if self.secim != "sabit" else str(self.gecikme)
            sec_en = f"{self.gecikme}, selected by {self.secim.upper()}" if self.secim != "sabit" else str(self.gecikme)
            if en:
                giris = (
                    "Causality among the variables was examined with the Toda and Yamamoto (1995) procedure, which remains valid when the "
                    "series are integrated or cointegrated. "
                    f"The VAR lag length was set to p = {sec_en}; adding the maximum order of integration (dmax = {self.dmax}), "
                    f"a VAR({k}) was estimated in levels{' with a linear trend' if self.trend else ''}. "
                    f"The null hypothesis of no causality was tested with a Wald χ²({self.gecikme}) test on the first p lags only."
                )
            else:
                giris = (
                    "Değişkenler arasındaki nedensellik, seriler bütünleşik ya da eşbütünleşik olsa da geçerli olan Toda ve Yamamoto (1995) "
                    f"yaklaşımıyla incelenmiştir. VAR modelinin gecikme uzunluğu p = {sec_tr} olarak belirlenmiş; serilerin en yüksek "
                    f"bütünleşme derecesi (dmax = {self.dmax}) eklenerek düzeyde VAR({k}) modeli"
                    f"{' doğrusal trendle' if self.trend else ''} tahmin edilmiştir. Nedensellik olmadığı temel hipotezi yalnızca ilk p "
                    f"gecikmeye uygulanan Wald χ²({self.gecikme}) testiyle sınanmıştır."
                )
            cumle = [giris]
            anlamli = self.tablo[(self.tablo["p"] < 0.05) & (self.tablo["Neden"] != "Tümü")]
            if anlamli.empty:
                cumle.append(
                    "No causal relationship was found at the 5% level in any direction."
                    if en
                    else "Hiçbir yönde %5 düzeyinde anlamlı bir nedensellik ilişkisi bulunamamıştır."
                )
            else:
                for _, r in anlamli.iterrows():
                    if r["Neden"] == "Tümü":
                        continue  # ortak test cümlede ayrıca anlatılmaz; tabloda yer alır
                    n_, e_ = ad.metin(r["Neden"]), ad.metin(r["Etkilenen"])
                    if en:
                        cumle.append(f"{n_} Granger-causes {e_} (χ²({int(r['sd'])}) = {sayi(r['χ²'], 2)}, {p_cumle(r['p'])}).")
                    else:
                        cumle.append(
                            f"{n_} değişkeninden {e_} değişkenine doğru bir nedensellik ilişkisi vardır (χ²({int(r['sd'])}) = "
                            f"{sayi(r['χ²'], 2)}, {p_cumle(r['p'])})."
                        )
            cumle.append(
                "The test is predictive and does not by itself establish structural causality."
                if en
                else "Test öngörüseldir; tek başına yapısal bir neden-sonuç ilişkisi kanıtlamaz."
            )

            def adi(x):
                return l("Tümü", "All") if x == "Tümü" else ad.yalin(x)

            satirlar = [
                [
                    f"{adi(r['Neden'])} → {ad.yalin(r['Etkilenen'])}",
                    sayi(r["χ²"], 3),
                    str(int(r["sd"])),
                    _p_hucre(r["p"]),
                    l("Reddedildi", "Rejected") if r["p"] < 0.05 else l("Reddedilemedi", "Not rejected"),
                ]
                for _, r in self.tablo.iterrows()
            ]
            t = Tablo(
                tablo_no,
                l("Toda-Yamamoto nedensellik testi sonuçları", "Toda-Yamamoto causality test results"),
                [l("Yön", "Direction"), "Wald χ²", l("sd", "df"), "p", l("H₀ kararı", "Decision on H₀")],
                satirlar,
                l(
                    f"H₀: Neden değişken, etkilenen değişkenin Granger nedeni değildir. p = {self.gecikme}, dmax = {self.dmax}, "
                    f"düzeyde VAR({k}), T = {self.T}. 'Tümü' satırı diğer tüm değişkenlerin birlikte sınanmasıdır.",
                    f"H₀: the cause variable does not Granger-cause the affected variable. p = {self.gecikme}, dmax = {self.dmax}, "
                    f"VAR({k}) in levels, T = {self.T}. The 'All' row tests all other variables jointly.",
                ),
            )
            r = Rapor(
                baslik,
                [
                    ("paragraf", " ".join(cumle)),
                    ("tablo", t),
                    ("baslik", l("Kaynakça", "References")),
                    ("liste", sorted([KAYNAKLAR["toda_yamamoto"], KAYNAKLAR["granger"]])),
                ],
                bicim=bicim,
                dil=dil,
            )
            if dosya is not None:
                r.kaydet(dosya)
            return r
        finally:
            DIL.reset(token)


def toda_yamamoto(
    veri: pd.DataFrame,
    sutunlar: list[str],
    gecikme: int | None = None,
    dmax: int = 1,
    bilgi_olcutu: str = "aic",
    en_fazla_gecikme: int | None = None,
    trend: bool = False,
) -> TodaYamamotoSonucu:
    """Toda-Yamamoto nedensellik testi. Veri zaman sırasına göre sıralı olmalı; seriler DÜZEY değerleriyle verilir.

    gecikme          : VAR gecikmesi p (verilmezse bilgi ölçütüyle seçilir; en az 1)
    dmax             : en yüksek bütünleşme derecesi (I(1) seriler için 1)
    bilgi_olcutu     : "aic", "sbic" (= "bic"), "hqic" ya da "fpe"
    en_fazla_gecikme : seçimde üst sınır (varsayılan 8, kısa seride T'ye göre küçültülür)
    trend            : denklemlere doğrusal trend ekle
    """
    from .vektor import _hazirla, var_gecikme_secimi

    if dmax < 0:
        raise _hata("dmax negatif olamaz.")
    Y = _hazirla(veri, sutunlar)
    T_ham, K = Y.shape
    secim = "sabit"
    if gecikme is None:
        olcut = {"bic": "SBIC", "sbic": "SBIC", "sic": "SBIC", "aic": "AIC", "hqic": "HQIC", "hq": "HQIC", "fpe": "FPE"}.get(
            bilgi_olcutu.lower()
        )
        if olcut is None:
            raise _hata("bilgi_olcutu: 'aic', 'sbic', 'hqic' ya da 'fpe' olmalı.")
        pmax = en_fazla_gecikme if en_fazla_gecikme is not None else max(1, min(8, (T_ham - dmax) // (2 * K) - 1))
        tablo = var_gecikme_secimi(pd.DataFrame(Y, columns=sutunlar), list(sutunlar), en_fazla_gecikme=pmax)
        gecikme = max(1, tablo.attrs["secim"][olcut])
        secim = olcut.lower()
    if int(gecikme) < 1:
        raise _hata("Toda-Yamamoto için gecikme en az 1 olmalı.")
    p = int(gecikme)
    k = p + dmax
    T = T_ham - k
    m = 1 + int(trend) + K * k
    if T <= m + 1:
        raise _hata(f"Gözlem sayısı (T = {T}) VAR({k}) için yetersiz; gecikmeyi ya da değişken sayısını azalt.")
    yy = Y[k:]
    parcalar = [np.ones((T, 1))]
    if trend:
        parcalar.append(np.arange(k + 1, T_ham + 1, dtype=float)[:, None])
    for j in range(1, k + 1):  # sütun sırası: L1(tüm değişkenler), L2(...), ...
        parcalar.append(Y[k - j : T_ham - j])
    X = np.hstack(parcalar)
    XtX_inv = np.linalg.inv(X.T @ X)
    B = XtX_inv @ X.T @ yy
    E = yy - X @ B
    bas = 1 + int(trend)

    def sutunlari(degiskenler: list[int]) -> list[int]:
        return [bas + (j - 1) * K + v for j in range(1, p + 1) for v in degiskenler]  # yalnızca ilk p gecikme

    satirlar = []
    for e, etkilenen in enumerate(sutunlar):
        s2 = float(E[:, e] @ E[:, e]) / (T - m)
        V = s2 * XtX_inv
        digerleri = [v for v in range(K) if v != e]
        gruplar = [(sutunlar[v], [v]) for v in digerleri]
        if K > 2:
            gruplar.append(("Tümü", digerleri))  # diğer tüm değişkenler birlikte
        for neden, vs in gruplar:
            idx = sutunlari(vs)
            b = B[idx, e]
            w = float(b @ np.linalg.solve(V[np.ix_(idx, idx)], b))
            satirlar.append({"Neden": neden, "Etkilenen": etkilenen, "χ²": w, "sd": len(idx), "p": float(stats.chi2.sf(w, len(idx)))})
    return TodaYamamotoSonucu(list(sutunlar), p, dmax, secim, trend, T, pd.DataFrame(satirlar))


__all__ = ["toda_yamamoto", "TodaYamamotoSonucu"]
