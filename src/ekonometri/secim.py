"""Sıralı ve çok kategorili seçim, sayım ve sansürlü veri modelleri (Stata: ologit, oprobit, mlogit, poisson, nbreg, tobit).

>>> ek.ologit("memnuniyet ~ gelir + yas + cinsiyet", veri)       # sıralı logit (kesim noktalarıyla)
>>> ek.mlogit("tercih ~ gelir + yas", veri)                       # temel kategori: en sık sonuç (Stata varsayılanı)
>>> ek.poisson("ziyaret ~ yas + sigorta", veri)                    # sayım verisi
>>> ek.nbreg("ziyaret ~ yas + sigorta", veri)                      # aşırı yayılım (α) ile negatif binom
>>> ek.tobit("harcama ~ gelir", veri, alt=0)                       # 0'da soldan sansürlü

Her model ``yorumla()``, ``marjinal_etkiler()`` ve ``rapor()`` sunar. Tahminler en çok olabilirlikle yapılır
(statsmodels / scipy); dayanıklı standart hatalarda Stata çarpanı N/(N−1), kümelenmişte G/(G−1) kullanılır.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import patsy
from scipy import stats

from ._yardimci import benzersiz_indeks, p_cumle, p_yaz, sayi, tablo_sayi, yildiz
from .regresyon import EkonometriHatasi, _terimleri_coz, gorunen_ad

MODEL_ADI = {
    "ologit": "Sıralı logit",
    "oprobit": "Sıralı probit",
    "mlogit": "Çok terimli logit",
    "poisson": "Poisson",
    "nbreg": "Negatif binom",
    "tobit": "Tobit",
}


# =========================================================================== ortak sonuç sınıfı
@dataclass
class MLSonuc:
    """En çok olabilirlik model sonucu (ologit, oprobit, mlogit, poisson, nbreg, tobit)."""

    tur: str
    formul: str
    bagimli: str
    katsayilar: pd.Series  # mlogit için 'sonuç: terim' adlı; ek parametreler (kesim, α, σ) ayrı
    standart_hatalar: pd.Series
    kovaryans_matrisi: pd.DataFrame  # tüm parametreler (katsayılar + ek parametreler)
    ek_parametreler: pd.DataFrame  # kesim noktaları / ln α, α / σ: Değer, Std. hata
    kovaryans_turu: str
    kovaryans_aciklama: str
    n: int
    log_olabilirlik: float
    log_olabilirlik_sifir: float
    k: int
    kategoriler: list = field(default_factory=list)  # sıralı / çok terimli için sonuç kategorileri
    temel: object = None  # mlogit temel kategorisi
    ek_bilgi: dict = field(default_factory=dict)
    X: pd.DataFrame = field(repr=False, default=None)
    y: pd.Series = field(repr=False, default=None)
    _design_info: object = field(repr=False, default=None)
    _terimler: list = field(repr=False, default_factory=list)
    _veri: pd.DataFrame = field(repr=False, default=None)
    _tahmin: object = field(repr=False, default=None)  # (params) -> olasılık / beklenen değer matrisi
    _parametreler: np.ndarray = field(repr=False, default=None)
    _etiketler: dict = field(repr=False, default_factory=dict)
    _birimler: dict = field(repr=False, default_factory=dict)

    ist_adi = "z"

    @property
    def model_turu(self) -> str:
        return MODEL_ADI[self.tur]

    @property
    def z_degerleri(self) -> pd.Series:
        return self.katsayilar / self.standart_hatalar

    t_degerleri = z_degerleri

    @property
    def p_degerleri(self) -> pd.Series:
        return pd.Series(2 * stats.norm.sf(np.abs(self.z_degerleri.values)), index=self.katsayilar.index)

    @property
    def pseudo_r2(self) -> float:
        return float(1 - self.log_olabilirlik / self.log_olabilirlik_sifir)

    @property
    def lr_istatistigi(self) -> float:
        return float(2 * (self.log_olabilirlik - self.log_olabilirlik_sifir))

    @property
    def lr_sd(self) -> int:
        return int(self.ek_bilgi.get("lr_sd", len(self.katsayilar) - ("Intercept" in self.katsayilar.index)))

    @property
    def lr_p_degeri(self) -> float:
        return float(stats.chi2.sf(self.lr_istatistigi, self.lr_sd))

    @property
    def aic(self) -> float:
        return float(-2 * self.log_olabilirlik + 2 * self.k)

    @property
    def bic(self) -> float:
        return float(-2 * self.log_olabilirlik + np.log(self.n) * self.k)

    def guven_araliklari(self, guven: float = 0.95) -> pd.DataFrame:
        z = stats.norm.ppf(0.5 + guven / 2)
        return pd.DataFrame({"alt": self.katsayilar - z * self.standart_hatalar, "üst": self.katsayilar + z * self.standart_hatalar})

    def _uyum(self) -> dict[str, str]:
        d = {"Gözlem sayısı": sayi(self.n, 0), "Log-olabilirlik": sayi(self.log_olabilirlik, 1)}
        if self.tur != "tobit":
            d["McFadden sözde R²"] = sayi(self.pseudo_r2, 3)
        d.update({"LR χ²": sayi(self.lr_istatistigi, 2) + yildiz(self.lr_p_degeri), "AIC": sayi(self.aic, 1), "BIC": sayi(self.bic, 1)})
        return d

    # --- oran yorumları
    def oranlar(self, guven: float = 0.95) -> pd.DataFrame:
        """e^β ve güven aralığı: odds oranı (ologit), göreli risk oranı (mlogit, Stata rrr), olay oranı oranı (poisson/nbreg, irr)."""
        if self.tur in ("oprobit", "tobit"):
            raise EkonometriHatasi(f"{self.model_turu} için üstel oran yorumu yoktur; marjinal_etkiler() kullan.")
        ga = self.guven_araliklari(guven)
        t = pd.DataFrame({"oran": np.exp(self.katsayilar), "alt": np.exp(ga["alt"]), "üst": np.exp(ga["üst"])})
        return t.drop(index=[i for i in t.index if i.endswith("Intercept")], errors="ignore")

    # --- marjinal etkiler
    def tahminler(self, tur: str | None = None) -> pd.DataFrame:
        """Gözlem başına tahminler (Stata ``predict``).

        ologit / oprobit / mlogit: her sonucun olasılığı (sütun başına bir sonuç) · poisson / nbreg: beklenen sayı (``n``) ·
        tobit: doğrusal tahmin (``xb``). ``tur="xb"`` her modelde doğrusal tahmini verir (mlogit'te temel dışı her sonuç için).
        """
        X = self.X.to_numpy(float)
        if tur == "xb":
            if self.tur == "mlogit":
                B = self._parametreler.reshape(len(self.kategoriler) - 1, X.shape[1]).T
                return pd.DataFrame(X @ B, index=self.X.index, columns=[f"xb: {c}" for c in self.kategoriler[1:]])  # ilk kategori temel
            return pd.DataFrame({"xb": X @ self._parametreler[: X.shape[1]]}, index=self.X.index)
        P = self._tahmin(self._parametreler, X)
        if self.tur in ("ologit", "oprobit", "mlogit"):
            t = pd.DataFrame(P, index=self.X.index, columns=[f"Pr({self.bagimli}={c})" for c in self.kategoriler])
            if self.tur == "mlogit":  # Stata: sütunlar sonuç değerlerinin sırasıyla (temel kategori önde değil)

                def anahtar(c):
                    try:
                        return (0, float(c), "")
                    except ValueError:
                        return (1, 0.0, str(c))

                t = t[[f"Pr({self.bagimli}={c})" for c in sorted(self.kategoriler, key=anahtar)]]
            return t
        return pd.DataFrame({"n" if self.tur in ("poisson", "nbreg") else "xb": P[:, 0]}, index=self.X.index)

    def marjinal_etkiler(self) -> pd.DataFrame:
        """Ortalama marjinal etkiler (Stata ``margins, dydx(*)``), delta yöntemiyle; sıralı / çok terimli modellerde her sonuç için ayrı.

        Sürekli değişkende türev, kategorik değişkende referans gruba göre fark. Poisson / nbreg'de beklenen sayı,
        tobit'te gizil değişken (doğrusal tahmin) üzerindeki etkidir.
        """
        b = self._parametreler
        V = self.kovaryans_matrisi.values
        veri, di = self._veri, self._design_info
        y_adi = {c for c in veri.columns if re.search(rf"(?<![\w.]){re.escape(c)}(?!\w)", self.formul.split("~")[0])}
        kategorik = {}
        for faktor, bilgi in di.factor_infos.items():
            if bilgi.type == "categorical":
                kategorik[re.sub(r"^C\((.+?)(?:,.*)?\)$", r"\1", faktor.name()).strip()] = list(bilgi.categories)

        dusur = [di.column_names.index("Intercept")] if self.ek_bilgi.get("sabitsiz") and "Intercept" in di.column_names else []

        def tasarim(d):
            (X,) = patsy.build_design_matrices([di], d)
            X = np.asarray(X, dtype=float)
            return np.delete(X, dusur, axis=1) if dusur else X

        def ortalama(params, d):
            return self._tahmin(params, tasarim(d)).mean(axis=0)

        def jakobiyen(islev, params):
            temel = islev(params)
            J = np.zeros((len(temel), len(params)))
            for j in range(len(params)):
                h = 1e-6 * max(1.0, abs(params[j]))
                a, e = params.copy(), params.copy()
                a[j] += h
                e[j] -= h
                J[:, j] = (islev(a) - islev(e)) / (2 * h)
            return temel, J

        satirlar = []
        cikti = self.kategoriler if self.tur in ("ologit", "oprobit", "mlogit") else [None]
        for c in veri.columns:
            if c in y_adi:
                continue
            if c in kategorik:
                ref, *diger = kategorik[c]
                for lv in diger:
                    d1, d0 = veri.copy(), veri.copy()
                    d1[c] = (
                        pd.Categorical([lv] * len(veri), categories=kategorik[c]) if isinstance(veri[c].dtype, pd.CategoricalDtype) else lv
                    )
                    d0[c] = (
                        pd.Categorical([ref] * len(veri), categories=kategorik[c])
                        if isinstance(veri[c].dtype, pd.CategoricalDtype)
                        else ref
                    )
                    deger, J = jakobiyen(lambda pr, d1=d1, d0=d0: ortalama(pr, d1) - ortalama(pr, d0), b)
                    for i, ck in enumerate(cikti):
                        satirlar.append((c, str(lv), str(ref), "kategorik", ck, deger[i], np.sqrt(J[i] @ V @ J[i])))
            elif pd.api.types.is_numeric_dtype(veri[c]):
                h = 1e-5 * max(float(veri[c].std()), 1e-8)
                art, eks = veri.copy(), veri.copy()
                art[c] = veri[c].astype(float) + h
                eks[c] = veri[c].astype(float) - h
                deger, J = jakobiyen(lambda pr, art=art, eks=eks, h=h: (ortalama(pr, art) - ortalama(pr, eks)) / (2 * h), b)
                for i, ck in enumerate(cikti):
                    satirlar.append((c, "", "", "sürekli", ck, deger[i], np.sqrt(J[i] @ V @ J[i])))
        t = pd.DataFrame(satirlar, columns=["değişken", "grup", "referans", "tür", "sonuç", "AME", "std. hata"])
        t["z"] = t["AME"] / t["std. hata"]
        t["p"] = 2 * stats.norm.sf(np.abs(t["z"]))
        if cikti == [None]:
            t = t.drop(columns="sonuç")
        return t

    # --- çıktı
    def ozet(self) -> str:
        from .regresyon import bagimli_gorunumu

        g = 78
        s = [
            "═" * g,
            f" {self.model_turu} — bağımlı değişken: {bagimli_gorunumu(self.bagimli)}",
            "═" * g,
            f" Gözlem: {self.n:<8} Log-olabilirlik: {sayi(self.log_olabilirlik, 3):<14}"
            + ("" if self.tur == "tobit" else f"Sözde R²: {sayi(self.pseudo_r2, 4)}"),
            f" LR χ²({self.lr_sd}) = {sayi(self.lr_istatistigi, 2)} (p {p_yaz(self.lr_p_degeri)})   AIC: {sayi(self.aic, 1)}   BIC: {sayi(self.bic, 1)}",
            f" Standart hatalar: {self.kovaryans_aciklama}",
        ]
        if self.tur == "mlogit":
            s.append(f" Temel kategori: {self.temel}")
        if self.tur == "tobit":
            s.append(f" Sansür: {self.ek_bilgi['sansur_metni']}")
        s += ["─" * g, f" {'Değişken':<30}{'Katsayı':>12}{'Std. hata':>12}{'z':>9}{'p':>10}", "─" * g]
        z, p = self.z_degerleri, self.p_degerleri
        onceki = None
        for ad in self.katsayilar.index:
            if self.tur == "mlogit":
                cikti, _, terim = ad.partition(": ")
                if cikti != onceki:
                    s.append(f" {cikti}")
                    onceki = cikti
                gor = "   " + gorunen_ad(self, terim)
            else:
                gor = gorunen_ad(self, ad)
            gor = gor if len(gor) <= 29 else gor[:28] + "…"
            s.append(
                f" {gor:<30}{tablo_sayi(self.katsayilar[ad]):>12}{tablo_sayi(self.standart_hatalar[ad]):>12}{sayi(z[ad], 2):>9}{p_yaz(p[ad]):>10} {yildiz(p[ad])}"
            )
        if len(self.ek_parametreler):
            s.append("─" * g)
            for ad, r in self.ek_parametreler.iterrows():
                s.append(f" {ad:<30}{tablo_sayi(r['Değer']):>12}{tablo_sayi(r['Std. hata']):>12}")
        if self.tur == "nbreg":
            t = self.ek_bilgi["alfa_lr"]
            s.append(f" α = 0 LR testi: χ̄²(01) = {sayi(t[0], 2)}, {p_cumle(t[1])}")
            if self.ek_bilgi.get("alfa_sinir"):
                s.append(" α = 0 sınırında: katsayılar Poisson limitidir, α için standart hata yoktur.")
        s.append("═" * g)
        return "\n".join(s)

    def __str__(self) -> str:
        return self.ozet()

    __repr__ = __str__

    def yorumla(self):
        from .yorum import Yorum

        et = self._etiketler or {}

        def ad(x):
            return et.get(x, f"'{x}'")

        y = Yorum(baslik=f"📊 Yorum — {self.model_turu}: {ad(self.bagimli)}")
        y.model_satirlari += [
            (
                f"Model, yalnızca sabit terimli modelden anlamlı biçimde daha iyi (LR χ²({self.lr_sd}) = {sayi(self.lr_istatistigi, 2)}, {p_cumle(self.lr_p_degeri)})."
                if self.lr_p_degeri < 0.05
                else f"Açıklayıcı değişkenler birlikte anlamlı değil (LR χ²({self.lr_sd}) = {sayi(self.lr_istatistigi, 2)}, {p_cumle(self.lr_p_degeri)})."
            ),
            f"Gözlem sayısı: {sayi(self.n, 0)} · standart hatalar: {self.kovaryans_aciklama}.",
        ]
        ame = self.marjinal_etkiler()
        for _, s in ame.iterrows():
            anl = f"{'anlamlı' if s['p'] < 0.05 else 'anlamlı değil'} ({p_cumle(s['p'])})"
            neye = {
                "poisson": f"beklenen {ad(self.bagimli)} sayısı",
                "nbreg": f"beklenen {ad(self.bagimli)} sayısı",
                "tobit": f"gizil {ad(self.bagimli)} değeri",
            }.get(self.tur)
            if neye is None:
                neye = f"{ad(self.bagimli)} = '{s['sonuç']}' olasılığı"
                buyukluk = f"{sayi(abs(100 * s['AME']), 2)} yüzde puan"
            else:
                buyukluk = sayi(abs(s["AME"]), 4)
            yon = "artıyor" if s["AME"] > 0 else "azalıyor"
            if s["tür"] == "sürekli":
                y.katsayi_satirlari.append(f"{ad(s['değişken'])} 1 birim artınca {neye} ortalama {buyukluk} {yon} — {anl}.")
            else:
                y.katsayi_satirlari.append(
                    f"{ad(s['değişken'])} değişkeninde '{s['grup']}' grubunda {neye}, referans gruba ('{s['referans']}') göre ortalama {buyukluk} "
                    f"{'daha yüksek' if s['AME'] > 0 else 'daha düşük'} — {anl}."
                )
        if self.tur in ("poisson", "nbreg"):
            for terim, r in self.oranlar().iterrows():
                y.notlar.append(
                    f"{gorunen_ad(self, terim)}: olay oranı oranı (IRR) = {sayi(r['oran'], 3)} → beklenen sayı %{sayi(100 * (r['oran'] - 1), 1)} değişir."
                )
        if self.tur == "nbreg":
            t = self.ek_bilgi["alfa_lr"]
            y.notlar.append(
                "Aşırı yayılım (α) anlamlı: Poisson yerine negatif binom uygundur."
                if t[1] < 0.05
                else "Aşırı yayılım (α) anlamlı değil: Poisson modeli yeterli olabilir."
            )
        if self.tur == "ologit":
            y.notlar.append("Sıralı logit, katsayıların tüm kesim noktalarında aynı olduğunu varsayar (paralel eğimler / orantılı odds).")
        if self.tur == "mlogit":
            y.notlar.append(f"Katsayılar temel kategoriye ('{self.temel}') göredir; e^β göreli risk oranıdır: model.oranlar().")
        if self.tur == "tobit":
            y.notlar.append(
                "Tobit katsayıları gizil (sansürsüz) değişken üzerindeki etkidir; gözlenen değer üzerindeki etki daha küçüktür."
            )
        y.notlar.append("Yorumlar ortalama marjinal etkilere dayanır (delta yöntemi); nedensel yorum araştırma tasarımına bağlıdır.")
        return y

    def rapor(self, dosya: str | None = None, baslik: str = "Bulgular", etiketler: dict | None = None, tablo_no: int = 1):
        """Yöntem + tanımlayıcı istatistikler + katsayı ve marjinal etki tabloları + yorum + kaynakça."""
        from . import __version__
        from .rapor import KAYNAKLAR, Rapor, Tablo, _Adlar, _p, _p_hucre, _tanimlayici

        ad = _Adlar({**self._etiketler, **(etiketler or {})})
        Y = ad.metin(self.bagimli)
        yontem = {
            "ologit": f"Bağımlı değişken {Y} sıralı kategorilerden oluştuğundan model sıralı logit ile tahmin edilmiştir (McCullagh, 1980).",
            "oprobit": f"Bağımlı değişken {Y} sıralı kategorilerden oluştuğundan model sıralı probit ile tahmin edilmiştir.",
            "mlogit": f"Bağımlı değişken {Y} sırasız birden çok kategoriden oluştuğundan model çok terimli logit ile tahmin edilmiştir (McFadden, 1974); temel kategori '{self.temel}' olarak alınmıştır.",
            "poisson": f"Bağımlı değişken {Y} sayım verisi olduğundan model Poisson regresyonuyla tahmin edilmiştir (Cameron ve Trivedi, 2013).",
            "nbreg": f"Bağımlı değişken {Y} sayım verisi olup aşırı yayılım gösterdiğinden model negatif binom (NB2) regresyonuyla tahmin edilmiştir (Cameron ve Trivedi, 2013).",
            "tobit": f"Bağımlı değişken {Y} {self.ek_bilgi.get('sansur_metni', 'sansürlü')} olduğundan model Tobit ile tahmin edilmiştir (Tobin, 1958).",
        }[self.tur]
        yontem += (
            f" Tahminler en çok olabilirlik yöntemiyle yapılmış, katsayılar ortalama marjinal etkiler (delta yöntemi) üzerinden yorumlanmıştır. "
            f"Standart hatalar: {self.kovaryans_aciklama}. Hesaplar ekonometri-tr {__version__} ile yapılmıştır."
        )
        r = Rapor(baslik, [("baslik", "Yöntem"), ("paragraf", yontem)])
        b, no = _tanimlayici(self, ad, tablo_no)
        r.bloklar += b
        p = self.p_degerleri
        satirlar = []
        for terim in self.katsayilar.index:
            gor = terim if self.tur == "mlogit" else gorunen_ad(self, terim)
            satirlar.append(
                [
                    gor,
                    tablo_sayi(self.katsayilar[terim]) + yildiz(p[terim]),
                    tablo_sayi(self.standart_hatalar[terim]),
                    sayi(self.z_degerleri[terim], 2),
                    _p_hucre(p[terim]),
                ]
            )
        for nm, rr in self.ek_parametreler.iterrows():
            satirlar.append([nm, tablo_sayi(rr["Değer"]), tablo_sayi(rr["Std. hata"]), "", ""])
        lr = f"LR χ²({self.lr_sd}) = {sayi(self.lr_istatistigi, 2)}"
        notu = (
            f"N = {sayi(self.n, 0)}; {lr} ({_p(self.lr_p_degeri)}); log-olabilirlik = {sayi(self.log_olabilirlik, 2)}"
            + ("" if self.tur == "tobit" else f"; McFadden sözde R² = {sayi(self.pseudo_r2, 3)}")
            + f". Standart hatalar: {self.kovaryans_aciklama}. *** p < 0,01; ** p < 0,05; * p < 0,10."
        )
        r.bloklar += [
            ("baslik", f"{self.model_turu} Bulguları"),
            (
                "paragraf",
                f"Tahmin sonuçları Tablo {no} ile sunulmuştur. "
                + (
                    f"Olabilirlik oranı testi modelin bir bütün olarak anlamlı olduğunu göstermektedir ({lr}; {_p(self.lr_p_degeri)})."
                    if self.lr_p_degeri < 0.05
                    else f"Olabilirlik oranı testine göre model bir bütün olarak anlamlı değildir ({lr}; {_p(self.lr_p_degeri)})."
                ),
            ),
            ("tablo", Tablo(no, f"{self.model_turu} tahmin sonuçları", ["Değişken", "Katsayı", "SH", "z", "p"], satirlar, notu)),
        ]
        no += 1
        ame = self.marjinal_etkiler()
        ame_satir = []
        for _, s in ame.iterrows():
            gor = f"{ad.yalin(s['değişken'])}: {s['grup']}" if s["tür"] == "kategorik" else ad.yalin(s["değişken"])
            on = [str(s["sonuç"])] if "sonuç" in ame.columns else []
            ame_satir.append(
                on + [gor, tablo_sayi(s["AME"]) + yildiz(s["p"]), tablo_sayi(s["std. hata"]), sayi(s["z"], 2), _p_hucre(s["p"])]
            )
        bas = (["Sonuç"] if "sonuç" in ame.columns else []) + ["Değişken", "AME", "SH", "z", "p"]
        cumle = self.yorumla().katsayi_satirlari
        r.bloklar += [
            ("paragraf", f"Ortalama marjinal etkiler Tablo {no} ile sunulmuştur. " + " ".join(cumle[:8])),
            (
                "tablo",
                Tablo(
                    no,
                    "Ortalama marjinal etkiler",
                    bas,
                    ame_satir,
                    "Delta yöntemiyle standart hatalar. *** p < 0,01; ** p < 0,05; * p < 0,10.",
                    sol=len(bas) - 4,
                ),
            ),
        ]
        if self.tur == "nbreg":
            t = self.ek_bilgi["alfa_lr"]
            r.bloklar.append(
                (
                    "paragraf",
                    f"Aşırı yayılım parametresinin sıfır olduğunu sınayan olabilirlik oranı testi (χ̄²(01) = {sayi(t[0], 2)}; {_p(t[1])}) "
                    + (
                        "aşırı yayılımın varlığına ve negatif binom modelinin Poisson modeline tercih edilmesi gerektiğine işaret etmektedir."
                        if t[1] < 0.05
                        else "aşırı yayılıma dair kanıt sunmamaktadır."
                    ),
                )
            )
        kay = {
            "ologit": ["mccullagh"],
            "oprobit": ["mccullagh"],
            "mlogit": ["mcfadden"],
            "poisson": ["cameron_trivedi"],
            "nbreg": ["cameron_trivedi"],
            "tobit": ["tobin"],
        }[self.tur]
        r.bloklar += [("baslik", "Kaynakça"), ("liste", sorted(KAYNAKLAR[k] for k in kay))]
        if dosya is not None:
            r.kaydet(dosya)
        return r


# =========================================================================== yardımcılar
def _cagiran_ortami():
    """Formüldeki adları, ekonometri paketinin dışındaki ilk çağıranın ortamında arar (np, log, ln, exp, sqrt da tanımlı)."""
    import inspect

    f = inspect.currentframe()
    while f is not None and str(f.f_globals.get("__name__", "")).startswith("ekonometri"):
        f = f.f_back
    ad_alanlari = [f.f_locals, f.f_globals] if f is not None else []
    return patsy.EvalEnvironment(ad_alanlari).with_outer_namespace({"np": np, "log": np.log, "ln": np.log, "exp": np.exp, "sqrt": np.sqrt})


def _veri_hazirla(formul: str, veri: pd.DataFrame, sabit: bool, derinlik: int = 3):
    if not isinstance(veri, pd.DataFrame):
        raise EkonometriHatasi("veri bir pandas DataFrame olmalı.")
    veri = benzersiz_indeks(veri)
    env = _cagiran_ortami()
    try:
        y_df, X_df = patsy.dmatrices(formul if sabit else formul + " - 1", veri, return_type="dataframe", NA_action="drop", eval_env=env)
    except patsy.PatsyError as h:
        raise EkonometriHatasi(f"Formül okunamadı: {h}\nMevcut sütunlar: {list(veri.columns)}") from None
    if y_df.shape[1] != 1:
        # metin bağımlı değişken: patsy kuklalara açar; ham sütunu kullan
        ad = formul.split("~")[0].strip()
        if ad not in veri.columns:
            raise EkonometriHatasi("Bağımlı değişken tek bir sütun olmalı.")
        y_ham = veri.loc[X_df.index, ad]
        y_adi = ad
    else:
        y_ham = y_df.iloc[:, 0]
        y_adi = y_df.columns[0]
    kullanilan = [c for c in veri.columns if isinstance(c, str) and re.search(rf"(?<![\w.]){re.escape(c)}(?!\w)", formul)]
    return y_ham, y_adi, X_df, veri.loc[X_df.index, kullanilan], veri


def _sandvic(H_inv: np.ndarray, skorlar: np.ndarray, kovaryans: str, kume_kod: np.ndarray | None) -> tuple[np.ndarray, str]:
    n = len(skorlar)
    if kovaryans == "saglam":
        return n / (n - 1) * H_inv @ (skorlar.T @ skorlar) @ H_inv, "dayanıklı (sandviç)"
    G = int(kume_kod.max()) + 1
    toplam = np.column_stack([np.bincount(kume_kod, weights=skorlar[:, j], minlength=G) for j in range(skorlar.shape[1])])
    return G / (G - 1) * H_inv @ (toplam.T @ toplam) @ H_inv, f"kümelenmiş, {G} küme"


def _kovaryans_sec(kovaryans: str, kume: str | None, veri_tum: pd.DataFrame, idx) -> tuple[str, np.ndarray | None]:
    if kume is not None and kovaryans == "klasik":
        kovaryans = "kume"
    if kovaryans not in ("klasik", "saglam", "kume"):
        raise EkonometriHatasi("kovaryans 'klasik', 'saglam' (vce(robust)) ya da 'kume' (vce(cluster)) olmalı.")
    if kovaryans == "kume":
        if kume is None or kume not in veri_tum.columns:
            raise EkonometriHatasi("Kümelenmiş standart hata için veride bulunan bir küme sütunu ver.")
        return kovaryans, pd.factorize(veri_tum.loc[idx, kume])[0]
    return kovaryans, None


def _skor_sayisal(loglikeobs, params: np.ndarray) -> np.ndarray:
    """Gözlem başına skorlar (sayısal türev)."""
    S = np.zeros((len(loglikeobs(params)), len(params)))
    for j in range(len(params)):
        h = 1e-6 * max(1.0, abs(params[j]))
        a, e = params.copy(), params.copy()
        a[j] += h
        e[j] -= h
        S[:, j] = (loglikeobs(a) - loglikeobs(e)) / (2 * h)
    return S


def _sonuc(
    tur, formul, y_adi, X_df, adlar, params, V, ek_df, kov, kov_ac, n, ll, ll0, k, kveri, tahmin, veri_tum, di=None, **kw
) -> MLSonuc:
    nk = len(adlar)
    di = di if di is not None else X_df.design_info
    return MLSonuc(
        tur=tur,
        formul=formul,
        bagimli=y_adi,
        katsayilar=pd.Series(params[:nk], index=adlar, name="katsayı"),
        standart_hatalar=pd.Series(np.sqrt(np.diag(V)[:nk]), index=adlar, name="std. hata"),
        kovaryans_matrisi=pd.DataFrame(V),
        ek_parametreler=ek_df,
        kovaryans_turu=kov,
        kovaryans_aciklama=kov_ac,
        n=n,
        log_olabilirlik=float(ll),
        log_olabilirlik_sifir=float(ll0),
        k=k,
        X=X_df,
        _design_info=di,
        _terimler=_terimleri_coz(di),
        _veri=kveri,
        _tahmin=tahmin,
        _parametreler=params,
        _etiketler=dict(veri_tum.attrs.get("etiketler", {})),
        **kw,
    )


def _etiket(c) -> str:
    """Kategori etiketi: 2.0 → "2" (patsy bağımlı değişkeni float'a çevirir)."""
    if isinstance(c, (float, np.floating)) and float(c).is_integer():
        return str(int(c))
    return str(c)


# =========================================================================== sıralı logit / probit
def _sirali(formul, veri, dagilim, kovaryans, kume):
    from statsmodels.miscmodels.ordinal_model import OrderedModel

    y_ham, y_adi, X_df, kveri, veri_tum = _veri_hazirla(formul, veri, sabit=True, derinlik=4)
    di = X_df.design_info
    X_df = X_df.drop(columns=[c for c in X_df.columns if c == "Intercept"])
    kat = (
        list(y_ham.cat.categories[y_ham.cat.categories.isin(y_ham.unique())])
        if isinstance(y_ham.dtype, pd.CategoricalDtype)
        else sorted(y_ham.dropna().unique())
    )
    if len(kat) < 3:
        raise EkonometriHatasi(
            f"Sıralı model için bağımlı değişkenin en az 3 kategorisi olmalı (bulunan: {kat}); 2 kategori için logit/probit kullan."
        )
    kod = pd.Categorical(y_ham, categories=kat, ordered=True).codes
    X = X_df.to_numpy(float)
    n, k = X.shape
    J = len(kat)
    mod = OrderedModel(kod, X, distr=dagilim)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bas = mod.fit(method="bfgs", disp=False, maxiter=5000, gtol=1e-10)
        sonuc = mod.fit(start_params=bas.params, method="newton", disp=False, maxiter=200)
    th = sonuc.params  # β, a1, log(Δ2), ...
    # Stata parametreleştirmesi: kesim_j = a1 + Σ exp(...)
    kesim = mod.transform_threshold_params(th)[1:-1]
    Jac = np.eye(len(th))
    for j in range(1, J - 1):  # kesim_{j+1} = a1 + Σ_{i≤j} exp(θ_i)
        Jac[k + j] = 0.0
        Jac[k + j, k] = 1.0
        for i in range(1, j + 1):
            Jac[k + j, k + i] = np.exp(th[k + i])
    H = -mod.hessian(th)
    H_inv = np.linalg.inv(H)
    kov, kume_kod = _kovaryans_sec(kovaryans, kume, veri_tum, X_df.index)
    if kov == "klasik":
        V_th, kov_ac = H_inv, "klasik (OIM)"
    else:
        V_th, kov_ac = _sandvic(H_inv, _skor_sayisal(mod.loglikeobs, th), kov, kume_kod)
    params = np.r_[th[:k], kesim]
    V = Jac @ V_th @ Jac.T
    ek_df = pd.DataFrame({"Değer": kesim, "Std. hata": np.sqrt(np.diag(V)[k:])}, index=[f"/kesim{j}" for j in range(1, J)])
    oranlar = np.bincount(kod, minlength=J) / n
    ll0 = float(np.sum(np.bincount(kod, minlength=J) * np.log(oranlar)))
    F = stats.logistic.cdf if dagilim == "logit" else stats.norm.cdf

    def tahmin(pr, Xm):
        b, c = pr[:k], pr[k:]
        xb = Xm @ b
        F_ = np.column_stack([np.zeros(len(xb))] + [F(cj - xb) for cj in c] + [np.ones(len(xb))])
        return np.diff(F_, axis=1)

    tur = "ologit" if dagilim == "logit" else "oprobit"
    s = _sonuc(
        tur,
        formul,
        y_adi,
        X_df,
        list(X_df.columns),
        params,
        V,
        ek_df,
        kov,
        kov_ac,
        n,
        sonuc.llf,
        ll0,
        len(params),
        kveri,
        tahmin,
        veri_tum,
        di=di,
        kategoriler=[_etiket(c) for c in kat],
        y=pd.Series(kod, index=X_df.index),
        ek_bilgi={"lr_sd": k, "sabitsiz": True},
    )
    return s


def ologit(formul: str, veri: pd.DataFrame, kovaryans: str = "klasik", kume: str | None = None) -> MLSonuc:
    """Sıralı logit (Stata ``ologit``): P(y ≤ j) = Λ(kesim_j − xβ). Kategoriler küçükten büyüğe sıralı olmalı (sayı ya da sıralı kategorik)."""
    return _sirali(formul, veri, "logit", kovaryans, kume)


def oprobit(formul: str, veri: pd.DataFrame, kovaryans: str = "klasik", kume: str | None = None) -> MLSonuc:
    """Sıralı probit (Stata ``oprobit``): P(y ≤ j) = Φ(kesim_j − xβ)."""
    return _sirali(formul, veri, "probit", kovaryans, kume)


# =========================================================================== çok terimli logit
def mlogit(formul: str, veri: pd.DataFrame, temel=None, kovaryans: str = "klasik", kume: str | None = None) -> MLSonuc:
    """Çok terimli logit (Stata ``mlogit``). ``temel`` verilmezse Stata gibi en sık görülen kategori temel alınır."""
    from statsmodels.discrete.discrete_model import MNLogit

    y_ham, y_adi, X_df, kveri, veri_tum = _veri_hazirla(formul, veri, sabit=True)
    sayim = y_ham.value_counts(sort=False)
    kat = list(sayim.index)
    try:
        kat = sorted(kat)
    except TypeError:
        pass
    if len(kat) < 3:
        raise EkonometriHatasi("Çok terimli logit için en az 3 kategori gerekli; 2 kategori için logit kullan.")
    if temel is None:
        temel = max(kat, key=lambda c: (sayim[c], -kat.index(c)))  # en sık; eşitlikte ilk kategori
    if temel not in kat:
        raise EkonometriHatasi(f"Temel kategori {temel!r} bağımlı değişkende yok. Kategoriler: {kat}")
    sira = [temel] + [c for c in kat if c != temel]
    kod = pd.Series(pd.Categorical(y_ham, categories=sira).codes, index=X_df.index)
    X = X_df.to_numpy(float)
    n, k = X.shape
    mod = MNLogit(kod.values, X)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = mod.fit(method="newton", disp=False, maxiter=200)
    P = r.params  # k × (J−1)
    params = P.ravel(order="F")  # sonuç blokları sırasıyla (statsmodels Hessian / skor sırasıyla aynı)
    H_inv = np.linalg.inv(-mod.hessian(params))
    kov, kume_kod = _kovaryans_sec(kovaryans, kume, veri_tum, X_df.index)
    if kov == "klasik":
        V, kov_ac = H_inv, "klasik (OIM)"
    else:
        V, kov_ac = _sandvic(H_inv, mod.score_obs(params), kov, kume_kod)
    adlar = [f"{c}: {t}" for c in sira[1:] for t in X_df.columns]
    J = len(sira)
    ll0 = float(np.sum(kod.value_counts().values * np.log(kod.value_counts().values / n)))

    def tahmin(pr, Xm):
        B = pr.reshape(J - 1, k).T
        e = np.column_stack([np.zeros(len(Xm)), Xm @ B])
        e -= e.max(axis=1, keepdims=True)
        p = np.exp(e)
        return p / p.sum(axis=1, keepdims=True)

    s = _sonuc(
        "mlogit",
        formul,
        y_adi,
        X_df,
        adlar,
        params,
        V,
        pd.DataFrame(columns=["Değer", "Std. hata"]),
        kov,
        kov_ac,
        n,
        r.llf,
        ll0,
        len(params),
        kveri,
        tahmin,
        veri_tum,
        kategoriler=[_etiket(c) for c in sira],
        temel=temel,
        y=kod,
        ek_bilgi={"lr_sd": (k - 1) * (J - 1)},
    )
    return s


# =========================================================================== Poisson ve negatif binom
def _maruziyet(veri_tum, X_df, maruziyet):
    if maruziyet is None:
        return None
    if maruziyet not in veri_tum.columns:
        raise EkonometriHatasi(f"Maruziyet değişkeni {maruziyet!r} veride yok.")
    m = veri_tum.loc[X_df.index, maruziyet].astype(float).to_numpy()
    if (m <= 0).any():
        raise EkonometriHatasi("Maruziyet değerleri pozitif olmalı.")
    return np.log(m)


def _ofset(ofset, n: int):
    """Gözlem başına log maruziyet (Stata margins / predict n gibi); farklı satır sayısında (yeni veri) ortalama."""
    if ofset is None:
        return 0.0
    return ofset if len(ofset) == n else float(ofset.mean())


def poisson(formul: str, veri: pd.DataFrame, kovaryans: str = "klasik", kume: str | None = None, maruziyet: str | None = None) -> MLSonuc:
    """Poisson regresyonu (Stata ``poisson``); ``maruziyet`` Stata ``exposure()`` karşılığıdır (log maruziyet ofset olarak girer)."""
    import statsmodels.api as sm

    y_ham, y_adi, X_df, kveri, veri_tum = _veri_hazirla(formul, veri, sabit=True)
    y = y_ham.astype(float).to_numpy()
    if (y < 0).any() or not np.allclose(y, np.round(y)):
        raise EkonometriHatasi("Poisson için bağımlı değişken negatif olmayan tam sayı (sayım) olmalı.")
    X = X_df.to_numpy(float)
    n, k = X.shape
    ofset = _maruziyet(veri_tum, X_df, maruziyet)
    mod = sm.GLM(y, X, family=sm.families.Poisson(), offset=ofset)
    r = mod.fit(tol=1e-12, maxiter=200)
    b = r.params
    mu = r.fittedvalues
    H_inv = np.linalg.inv((X * mu[:, None]).T @ X)
    kov, kume_kod = _kovaryans_sec(kovaryans, kume, veri_tum, X_df.index)
    if kov == "klasik":
        V, kov_ac = H_inv, "klasik (OIM)"
    else:
        V, kov_ac = _sandvic(H_inv, X * (y - mu)[:, None], kov, kume_kod)
    ll = float(np.sum(stats.poisson.logpmf(y, mu)))
    mu0 = np.full(n, y.mean()) if ofset is None else np.exp(np.log(y.sum() / np.exp(ofset).sum()) + ofset)
    ll0 = float(np.sum(stats.poisson.logpmf(y, mu0)))

    def tahmin(pr, Xm):
        return np.exp(Xm @ pr + _ofset(ofset, len(Xm)))[:, None]

    return _sonuc(
        "poisson",
        formul,
        y_adi,
        X_df,
        list(X_df.columns),
        b,
        V,
        pd.DataFrame(columns=["Değer", "Std. hata"]),
        kov,
        kov_ac,
        n,
        ll,
        ll0,
        k,
        kveri,
        tahmin,
        veri_tum,
        y=pd.Series(y, index=X_df.index),
    )


def nbreg(formul: str, veri: pd.DataFrame, kovaryans: str = "klasik", kume: str | None = None, maruziyet: str | None = None) -> MLSonuc:
    """Negatif binom (NB2) regresyonu (Stata ``nbreg``): Var(y) = μ + α·μ². α = 0 için LR testi (χ̄²(01)) raporlanır."""
    from statsmodels.discrete.discrete_model import NegativeBinomial

    y_ham, y_adi, X_df, kveri, veri_tum = _veri_hazirla(formul, veri, sabit=True)
    y = y_ham.astype(float).to_numpy()
    if (y < 0).any() or not np.allclose(y, np.round(y)):
        raise EkonometriHatasi("Negatif binom için bağımlı değişken negatif olmayan tam sayı (sayım) olmalı.")
    X = X_df.to_numpy(float)
    n, k = X.shape
    ofset = _maruziyet(veri_tum, X_df, maruziyet)
    mod = NegativeBinomial(y, X, loglike_method="nb2", offset=ofset)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r0 = mod.fit(method="bfgs", disp=False, maxiter=5000, gtol=1e-10)
        r = mod.fit(start_params=r0.params, method="newton", disp=False, maxiter=200)
    params = r.params.copy()  # β, α
    llf = float(r.llf)
    pois = poisson(formul, veri, kovaryans=kovaryans, kume=kume, maruziyet=maruziyet)
    # NB, Poisson'u kapsar: geçerli bir en çok olabilirlik çözümü Poisson'un log-olabilirliğinden küçük olamaz. Optimizasyon
    # α sınırına (0) gidip NaN verirse ya da (sürüm / başlangıç değerine bağlı olarak) yanlış bir noktada durursa
    # ln α ile sınırlı, Poisson'dan başlayan yedek optimizasyon yapılır.
    gecerli = np.all(np.isfinite(params)) and np.isfinite(llf) and params[-1] > 0 and llf >= pois.log_olabilirlik - 1e-8
    if not gecerli:
        from scipy.optimize import minimize

        b0 = pois._parametreler

        def eksi_ll(th):
            return -mod.loglike(np.r_[th[:-1], np.exp(th[-1])])

        en_iyi = None
        for ln_a0 in (-2.0, -8.0, -15.0):
            sonuc = minimize(
                eksi_ll,
                np.r_[b0, ln_a0],
                method="L-BFGS-B",
                bounds=[(None, None)] * k + [(-25.0, 5.0)],
                options={"ftol": 1e-15, "gtol": 1e-10, "maxiter": 20000},
            )
            if en_iyi is None or sonuc.fun < en_iyi.fun:
                en_iyi = sonuc
        params = np.r_[en_iyi.x[:-1], np.exp(en_iyi.x[-1])]
        llf = float(mod.loglike(params))
        if params[-1] >= 1e-6:  # iç çözüm: Newton ile son yakınsama (standart hatalar için tam Hessian)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                r2 = mod.fit(start_params=params, method="newton", disp=False, maxiter=200)
            if np.all(np.isfinite(r2.params)) and r2.params[-1] > 0 and np.isfinite(r2.llf) and r2.llf >= llf:
                params, llf = r2.params.copy(), float(r2.llf)
    sinirda = params[-1] < 1e-6 or llf < pois.log_olabilirlik
    if sinirda:
        # α ≈ 0: aşırı yayılım yok, model Poisson'a indirgenir. Çok küçük α'da NB olabilirliği sayısal olarak güvenilmez
        # olduğundan β, log-olabilirlik ve β'nın kovaryansı Poisson limitinden alınır (LR = 0, p = 1). α'nın standart
        # hatası bu sınırda tanımlı değildir; Hessian'dan üretilen sayı (şişmiş /ln α hatası) yazılmaz.
        params = np.r_[pois._parametreler, 1e-8]  # α ≈ 0 (ortamdan bağımsız sabit sınır değeri)
        llf = pois.log_olabilirlik
        Vp = np.asarray(pois.kovaryans_matrisi, dtype=float)
        V = np.full((k + 1, k + 1), np.nan)
        V[:k, :k] = Vp[:k, :k]
        kov, kov_ac = pois.kovaryans_turu, pois.kovaryans_aciklama
        se_alfa = se_lnalfa = float("nan")
    else:
        alfa_ici = params[-1]
        H_inv = np.linalg.inv(-mod.hessian(params))
        kov, kume_kod = _kovaryans_sec(kovaryans, kume, veri_tum, X_df.index)
        if kov == "klasik":
            V, kov_ac = H_inv, "klasik (OIM)"
        else:
            V, kov_ac = _sandvic(H_inv, mod.score_obs(params), kov, kume_kod)
        se_alfa = float(np.sqrt(V[-1, -1]))
        se_lnalfa = se_alfa / alfa_ici
    alfa = params[-1]
    ek_df = pd.DataFrame({"Değer": [np.log(alfa), alfa], "Std. hata": [se_lnalfa, se_alfa]}, index=["/ln α", "α"])
    lr = 0.0 if sinirda else max(0.0, 2 * (llf - pois.log_olabilirlik))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ll0 = float(NegativeBinomial(y, np.ones((n, 1)), loglike_method="nb2", offset=ofset).fit(disp=False, maxiter=500).llf)
    # NB, Poisson'u kapsar: sabitli NB'nin en büyük olabilirliği sabitli Poisson'unkinden küçük olamaz (α → 0 sınırında
    # optimizasyon yakınsamazsa bu alt sınır kullanılır).
    ll0 = max(ll0, pois.log_olabilirlik_sifir) if np.isfinite(ll0) else pois.log_olabilirlik_sifir

    def tahmin(pr, Xm):
        return np.exp(Xm @ pr[:k] + _ofset(ofset, len(Xm)))[:, None]

    return _sonuc(
        "nbreg",
        formul,
        y_adi,
        X_df,
        list(X_df.columns),
        params,
        V,
        ek_df,
        kov,
        kov_ac,
        n,
        llf,
        ll0,
        k + 1,
        kveri,
        tahmin,
        veri_tum,
        y=pd.Series(y, index=X_df.index),
        ek_bilgi={
            "alfa_lr": (lr, 0.5 * stats.chi2.sf(lr, 1) if lr > 0 else 1.0),
            "lr_sd": k - 1,
            "alfa_sinir": sinirda,
        },
    )


# =========================================================================== Tobit
def tobit(
    formul: str, veri: pd.DataFrame, alt: float | None = 0.0, ust: float | None = None, kovaryans: str = "klasik", kume: str | None = None
) -> MLSonuc:
    """Tobit (Stata ``tobit y x, ll(0)``): alt ve/veya üst sınırdan sansürlü doğrusal model. σ ve var(e) raporlanır."""
    from scipy.optimize import minimize

    y_ham, y_adi, X_df, kveri, veri_tum = _veri_hazirla(formul, veri, sabit=True)
    y = y_ham.astype(float).to_numpy()
    X = X_df.to_numpy(float)
    n, k = X.shape
    sol = (y <= alt) if alt is not None else np.zeros(n, bool)
    sag = (y >= ust) if ust is not None else np.zeros(n, bool)
    orta = ~(sol | sag)
    if sol.sum() + sag.sum() == 0:
        raise EkonometriHatasi("Sansürlü gözlem yok; doğrusal regresyon (ek.ekk) kullan.")

    def llobs(th):
        b, s = th[:k], np.exp(th[k])
        xb = X @ b
        lo = np.empty(n)
        lo[orta] = stats.norm.logpdf((y[orta] - xb[orta]) / s) - np.log(s)
        if alt is not None:
            lo[sol] = stats.norm.logcdf((alt - xb[sol]) / s)
        if ust is not None:
            lo[sag] = stats.norm.logsf((ust - xb[sag]) / s)
        return lo

    bas = np.linalg.lstsq(X[orta], y[orta], rcond=None)[0]
    th0 = np.r_[bas, np.log(np.std(y[orta] - X[orta] @ bas) + 1e-6)]
    r = minimize(lambda t: -llobs(t).sum(), th0, method="BFGS", options={"gtol": 1e-10, "maxiter": 10000})
    th = r.x
    from statsmodels.tools.numdiff import approx_hess3

    for _ in range(20):  # Newton düzeltmesi
        g = _skor_sayisal(llobs, th).sum(axis=0)
        Hm = approx_hess3(th, lambda t: llobs(t).sum())
        adim = np.linalg.solve(-Hm, g)
        th = th + adim
        if np.max(np.abs(adim)) < 1e-10:
            break
    Hm = approx_hess3(th, lambda t: llobs(t).sum())
    H_inv = np.linalg.inv(-Hm)
    kov, kume_kod = _kovaryans_sec(kovaryans, kume, veri_tum, X_df.index)
    if kov == "klasik":
        V_th, kov_ac = H_inv, "klasik (OIM)"
    else:
        V_th, kov_ac = _sandvic(H_inv, _skor_sayisal(llobs, th), kov, kume_kod)
    s = np.exp(th[k])
    Jac = np.eye(k + 1)
    Jac[k, k] = s  # σ = exp(lnσ)
    params = np.r_[th[:k], s]
    V = Jac @ V_th @ Jac.T
    se_s = np.sqrt(V[k, k])
    ek_df = pd.DataFrame({"Değer": [s, s**2], "Std. hata": [se_s, 2 * s * se_s]}, index=["/σ", "var(e)"])
    # sabit terimli tobit (LR karşılaştırması)
    X0 = np.ones((n, 1))

    def ll0obs(t):
        xb = X0 @ t[:1]
        sg = np.exp(t[1])
        lo = np.empty(n)
        lo[orta] = stats.norm.logpdf((y[orta] - xb[orta]) / sg) - np.log(sg)
        if alt is not None:
            lo[sol] = stats.norm.logcdf((alt - xb[sol]) / sg)
        if ust is not None:
            lo[sag] = stats.norm.logsf((ust - xb[sag]) / sg)
        return lo

    r0 = minimize(lambda t: -ll0obs(t).sum(), np.r_[y.mean(), np.log(y.std() + 1e-6)], method="BFGS", options={"gtol": 1e-10})
    metin = " ve ".join(
        x
        for x in [
            f"{sayi(alt, 4)} değerinde soldan sansürlü ({int(sol.sum())} gözlem)" if alt is not None else "",
            f"{sayi(ust, 4)} değerinde sağdan sansürlü ({int(sag.sum())} gözlem)" if ust is not None else "",
        ]
        if x
    )

    def tahmin(pr, Xm):
        return (Xm @ pr[:k])[:, None]

    return _sonuc(
        "tobit",
        formul,
        y_adi,
        X_df,
        list(X_df.columns),
        params,
        V,
        ek_df,
        kov,
        kov_ac,
        n,
        float(llobs(th).sum()),
        float(-r0.fun),
        k + 1,
        kveri,
        tahmin,
        veri_tum,
        y=pd.Series(y, index=X_df.index),
        ek_bilgi={"sansur_metni": metin, "lr_sd": k - 1, "sol": int(sol.sum()), "sag": int(sag.sum())},
    )


__all__ = ["ologit", "oprobit", "mlogit", "poisson", "nbreg", "tobit", "MLSonuc"]
