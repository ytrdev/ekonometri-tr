"""Araç değişken (IV) yöntemi: iki aşamalı en küçük kareler (2SLS) ve tanı testleri.

Formülde içsel değişken ve araçları köşeli parantezle yazılır:

>>> veri = ek.veri.yukle("getiri")
>>> model = ek.iv2sls("np.log(ucret) ~ tecrube + I(tecrube**2) + cinsiyet + [egitim ~ uzaklik + kardes]", veri)
>>> model.tanilar()              # zayıf araç (birinci aşama F), Wu-Hausman içsellik, Sargan aşırı tanımlama
>>> model.rapor("iv.docx")       # tez bölümü: EKK ile karşılaştırma, birinci aşama, tanılar, yorum
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import patsy
from scipy import stats

from ._yardimci import DIL, benzersiz_indeks, p_cumle, p_yaz, sayi, tablo_sayi, yildiz
from .regresyon import EkonometriHatasi, _terimleri_coz, bagimli_gorunumu, gorunen_ad

_KOSELI = re.compile(r"\[([^\[\]~]+)~([^\[\]]+)\]")
ZAYIF_ESIK = 10.0  # Staiger ve Stock (1997) pratik kuralı


@dataclass
class IVTesti:
    ad: str
    h0: str
    istatistik: float
    serbestlik: tuple
    p_degeri: float | None
    sorun_var: bool
    karar: str
    sonuc: str = ""

    def __str__(self) -> str:
        sd = ", ".join(str(s) for s in self.serbestlik)
        p = f", {p_cumle(self.p_degeri)}" if self.p_degeri is not None else ""
        return f"{'⚠️' if self.sorun_var else '✅'} {self.ad}: istatistik = {sayi(self.istatistik, 3)} [{sd}]{p}\n   H0: {self.h0}\n   {self.karar}"

    __repr__ = __str__


@dataclass
class IVSonuc:
    """2SLS tahmin sonuçları. ``ek.iv2sls(...)`` tarafından döndürülür."""

    formul: str
    bagimli: str
    icsel: list[str]
    araclar: list[str]
    katsayilar: pd.Series
    standart_hatalar: pd.Series
    t_degerleri: pd.Series
    p_degerleri: pd.Series
    kovaryans_matrisi: pd.DataFrame
    kovaryans_turu: str
    kovaryans_aciklama: str
    n: int
    k: int
    sse: float
    r2: float
    duzeltilmis_r2: float
    f_istatistigi: float
    f_p_degeri: float
    f_serbestlik: tuple[int, int]
    kalintilar: pd.Series
    ekk: object  # aynı yapısal modelin EKK tahmini (karşılaştırma için)
    birinci_asama: dict  # içsel değişken -> {"F", "p", "sd", "kismi_r2", "model"}
    X: pd.DataFrame = field(repr=False, default=None)
    Z: pd.DataFrame = field(repr=False, default=None)
    y: pd.Series = field(repr=False, default=None)
    _terimler: list = field(repr=False, default_factory=list)
    _birimler: dict = field(repr=False, default_factory=dict)
    _veri: pd.DataFrame = field(repr=False, default=None)
    _testler: list | None = field(repr=False, default=None)
    kucuk_orneklem: bool = False
    _etiketler: dict = field(repr=False, default_factory=dict)

    model_turu = "2SLS"

    @property
    def ist_adi(self) -> str:
        """Stata ivregress gibi: varsayılan z, ``small`` (kucuk_orneklem=True) ile t."""
        return "t" if self.kucuk_orneklem else "z"

    @property
    def genel_test(self) -> str:
        if self.kucuk_orneklem:
            return f"F({self.f_serbestlik[0]}, {self.f_serbestlik[1]}) = {sayi(self.f_istatistigi, 2)}"
        return f"Wald χ²({self.f_serbestlik[0]}) = {sayi(self.f_istatistigi, 2)}"

    @property
    def sd(self) -> int:
        return self.n - self.k

    def _uyum(self) -> dict[str, str]:
        d = {"Gözlem sayısı": sayi(self.n, 0), "R²": sayi(self.r2, 3)}
        for ic, b in self.birinci_asama.items():
            d[f"1. aşama F ({ic})"] = sayi(b["F"], 2)
        return d

    def guven_araliklari(self, guven: float = 0.95) -> pd.DataFrame:
        t = stats.t.ppf(0.5 + guven / 2, self.sd) if self.kucuk_orneklem else stats.norm.ppf(0.5 + guven / 2)
        return pd.DataFrame({"alt": self.katsayilar - t * self.standart_hatalar, "üst": self.katsayilar + t * self.standart_hatalar})

    # ---- tanı testleri
    def tanilar(self) -> list[IVTesti]:
        """Zayıf araç (birinci aşama F), Wu-Hausman içsellik ve (aşırı tanımlıysa) Sargan testleri."""
        if self._testler is None:
            self._testler = _tanilar(self)
        return self._testler

    # ---- çıktılar
    def ozet(self) -> str:
        g = 78
        s = [
            "═" * g,
            f" 2SLS (Araç Değişken) Sonuçları — bağımlı değişken: {bagimli_gorunumu(self.bagimli)}",
            "═" * g,
            f" Gözlem sayısı: {self.n:<10} R²: {sayi(self.r2, 4):<10} {self.genel_test} (p {p_yaz(self.f_p_degeri)})",
            f" İçsel: {', '.join(self.icsel)}   Araçlar: {', '.join(self.araclar)}",
            f" Standart hatalar: {self.kovaryans_aciklama}",
            "─" * g,
            f" {'Değişken':<28}{'Katsayı':>12}{'Std. hata':>12}{self.ist_adi:>9}{'p':>10}",
            "─" * g,
        ]
        for ad in self.katsayilar.index:
            gor = gorunen_ad(self, ad)
            gor = (gor if len(gor) <= 25 else gor[:24] + "…") + (" ◂" if ad in self.icsel else "")
            p = self.p_degerleri[ad]
            s.append(
                f" {gor:<28}{tablo_sayi(self.katsayilar[ad]):>12}{tablo_sayi(self.standart_hatalar[ad]):>12}"
                f"{sayi(self.t_degerleri[ad], 2):>9}{p_yaz(p):>10} {yildiz(p)}"
            )
        s += ["─" * g, " ◂ içsel değişken · Tanı testleri: model.tanilar() · Yorum: model.yorumla()", "═" * g]
        return "\n".join(s)

    def __str__(self) -> str:
        return self.ozet()

    __repr__ = __str__

    def yorumla(self):
        from .yorum import yorumla

        y = yorumla(self)
        y.baslik = y.baslik.replace("Yorum", "Yorum (2SLS)")
        y.model_satirlari = [
            f"İçsel değişken: {', '.join(repr(i) for i in self.icsel)}; araçlar: {', '.join(repr(a) for a in self.araclar)}.",
            f"Gözlem sayısı: {sayi(self.n, 0)} · standart hatalar: {self.kovaryans_aciklama}.",
        ]
        for t in self.tanilar():
            y.model_satirlari.append(f"{t.ad}: {t.karar}")
        for ic in self.icsel:
            if ic in self.ekk.katsayilar.index:
                y.notlar.insert(
                    0,
                    f"EKK tahmini '{ic}' için {sayi(self.ekk.katsayilar[ic], 4)}, 2SLS tahmini {sayi(self.katsayilar[ic], 4)}: "
                    + (
                        "EKK, içsellik nedeniyle etkiyi olduğundan büyük gösteriyor olabilir."
                        if self.ekk.katsayilar[ic] > self.katsayilar[ic]
                        else "EKK, içsellik nedeniyle etkiyi olduğundan küçük gösteriyor olabilir."
                    ),
                )
        y.notlar = [n for n in y.notlar if "model.varsayimlar()" not in n and "nedensellik için" not in n]
        y.notlar.append(
            "2SLS tahmininin nedensel yorumu, araçların geçerli olmasına (yalnızca içsel değişken aracılığıyla etki etmesine) bağlıdır; bu varsayım test edilemez, kuramla savunulmalıdır."
        )
        return y

    def rapor(
        self,
        dosya: str | None = None,
        baslik: str = "Bulgular",
        etiketler: dict | None = None,
        tablo_no: int = 1,
        dil: str = "tr",
        bicim: str = "apa",
    ):
        """Yöntem + birinci aşama + EKK/2SLS karşılaştırması + tanı testleri + yorum (tez bölümü)."""
        r = _rapor(self, baslik, etiketler, tablo_no, dil, bicim)
        if dosya is not None:
            r.kaydet(dosya)
        return r


# --------------------------------------------------------------------------- tahmin
def _ekk_kat(X, y):
    XtX_inv = np.linalg.inv(X.T @ X)
    b = XtX_inv @ X.T @ y
    return b, y - X @ b, XtX_inv


def _wald_f(b, V, idx, sd):
    q = len(idx)
    bi = b[idx]
    F = float(bi @ np.linalg.inv(V[np.ix_(idx, idx)]) @ bi) / q
    return F, float(stats.f.sf(F, q, sd))


def iv2sls(
    formul: str,
    veri: pd.DataFrame,
    kovaryans: str = "klasik",
    birimler: dict[str, str] | None = None,
    kucuk_orneklem: bool = False,
    kume: str | None = None,
) -> IVSonuc:
    """İki aşamalı en küçük kareler (2SLS) — Stata ``ivregress 2sls`` ile aynı.

    Formül: ``"y ~ dışsal1 + dışsal2 + [içsel ~ araç1 + araç2]"``. Araç sayısı içsel değişken sayısından az olamaz.

    kovaryans : "klasik", "saglam" (Stata ``vce(robust)``; "HC1" de kabul edilir) ya da "kume" (``vce(cluster kume)``)
    kucuk_orneklem : Stata ``small`` seçeneği. False (Stata varsayılanı): σ² = e'e / N, z ve Wald χ²;
        True: σ² = e'e / (N − k), t ve F, dayanıklı/kümelenmiş çarpanlarda küçük örneklem düzeltmesi.
    """
    if kovaryans == "HC1":
        kovaryans = "saglam"
    if kume is not None and kovaryans == "klasik":
        kovaryans = "kume"
    if kovaryans not in ("klasik", "saglam", "kume"):
        raise EkonometriHatasi(f"Bilinmeyen kovaryans türü: {kovaryans!r}. Seçenekler: 'klasik', 'saglam', 'kume'")
    if kovaryans == "kume" and (kume is None or kume not in veri.columns):
        raise EkonometriHatasi("Kümelenmiş standart hata için veride bulunan bir küme sütunu ver: kume='okul'.")
    if not isinstance(veri, pd.DataFrame):
        raise EkonometriHatasi("veri bir pandas DataFrame olmalı.")
    veri = benzersiz_indeks(veri)
    eslesme = _KOSELI.findall(formul)
    if len(eslesme) != 1:
        raise EkonometriHatasi(
            "Formülde içsel değişkenler ve araçları tek bir köşeli parantez içinde yazılmalı: 'y ~ x1 + [içsel ~ araç1 + araç2]'."
        )
    icsel_ifade, arac_ifade = (e.strip() for e in eslesme[0])
    yapisal = _KOSELI.sub(icsel_ifade, formul)
    env = patsy.EvalEnvironment.capture(1).with_outer_namespace({"np": np, "log": np.log, "ln": np.log, "exp": np.exp, "sqrt": np.sqrt})
    try:
        ortak = patsy.dmatrices(f"{yapisal} + {arac_ifade}", veri, return_type="dataframe", NA_action="drop", eval_env=env)[0].index
        v = veri.loc[ortak]
        y_df, X_df = patsy.dmatrices(yapisal, v, return_type="dataframe", eval_env=env)
        X2_df = patsy.dmatrix(f"0 + {icsel_ifade}", v, return_type="dataframe", eval_env=env)
        Z2_df = patsy.dmatrix(f"0 + {arac_ifade}", v, return_type="dataframe", eval_env=env)
    except patsy.PatsyError as h:
        raise EkonometriHatasi(f"Formül okunamadı: {h}\nMevcut sütunlar: {list(veri.columns)}") from None
    icsel = list(X2_df.columns)
    if any(c not in X_df.columns for c in icsel):
        raise EkonometriHatasi("İçsel değişkenler sayısal olmalı (kategorik içsel değişken desteklenmiyor).")
    araclar = list(Z2_df.columns)
    if len(araclar) < len(icsel):
        raise EkonometriHatasi(
            f"Model eksik tanımlı: {len(icsel)} içsel değişken için en az {len(icsel)} araç gerekli ({len(araclar)} verildi)."
        )
    if set(araclar) & set(X_df.columns):
        raise EkonometriHatasi("Araç değişkenler modelin açıklayıcı değişkenleri arasında yer almamalı (dışlanmış araç olmalı).")
    disal = [c for c in X_df.columns if c not in icsel]
    y = y_df.iloc[:, 0].values.astype(float)
    X = X_df.values.astype(float)
    X1 = X_df[disal].values.astype(float)
    Z = np.column_stack([X1, Z2_df.values.astype(float)])
    n, k = X.shape
    if np.linalg.matrix_rank(Z) < Z.shape[1]:
        raise EkonometriHatasi("Araçlar ile dışsal değişkenler arasında tam doğrusal bağlantı var.")

    # 2SLS
    PzX = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    A_inv = np.linalg.inv(PzX.T @ X)
    b = A_inv @ PzX.T @ y
    e = y - X @ b
    sse = float(e @ e)
    sd = n - k
    Bh = np.linalg.inv(PzX.T @ PzX)
    if kovaryans == "klasik":
        V = sse / (sd if kucuk_orneklem else n) * Bh
        kov_ac = "klasik"
    elif kovaryans == "saglam":
        orta = (PzX * (e**2)[:, None]).T @ PzX
        V = (n / sd if kucuk_orneklem else 1.0) * Bh @ orta @ Bh
        kov_ac = "değişen varyansa dayanıklı (robust)"
    else:
        kod = pd.factorize(v.loc[y_df.index, kume])[0]
        G = int(kod.max()) + 1
        Pe = PzX * e[:, None]
        toplam = np.column_stack([np.bincount(kod, weights=Pe[:, j], minlength=G) for j in range(k)])
        c = G / (G - 1) * ((n - 1) / sd if kucuk_orneklem else 1.0)
        V = c * Bh @ (toplam.T @ toplam) @ Bh
        kov_ac = f"kümelenmiş, {G} küme ({kume})"
    se = np.sqrt(np.diag(V))
    tv = b / se
    pv = 2 * stats.t.sf(np.abs(tv), sd) if kucuk_orneklem else 2 * stats.norm.sf(np.abs(tv))
    sabit = "Intercept" in X_df.columns
    sst = float(np.sum((y - y.mean()) ** 2)) if sabit else float(y @ y)
    r2 = 1 - sse / sst
    idx = [i for i, c in enumerate(X_df.columns) if c != "Intercept"]
    if not idx:
        F, Fp = float("nan"), float("nan")
    elif kucuk_orneklem:
        F, Fp = _wald_f(b, V, idx, sd)
    else:  # Stata ivregress varsayılanı: Wald χ²
        W = float(b[idx] @ np.linalg.inv(V[np.ix_(idx, idx)]) @ b[idx])
        F, Fp = W, float(stats.chi2.sf(W, len(idx)))

    # birinci aşama: her içsel değişken için dışlanmış araçların ortak anlamlılığı
    birinci = {}
    j_arac = list(range(X1.shape[1], Z.shape[1]))
    for ic in icsel:
        x2 = X_df[ic].values.astype(float)
        g, u, ZtZ_inv = _ekk_kat(Z, x2)
        sd1 = n - Z.shape[1]
        if kovaryans == "klasik":
            V1 = float(u @ u) / sd1 * ZtZ_inv
        elif kovaryans == "saglam":
            V1 = n / sd1 * ZtZ_inv @ ((Z * (u**2)[:, None]).T @ Z) @ ZtZ_inv
        else:
            kod1 = pd.factorize(v.loc[y_df.index, kume])[0]
            G1 = int(kod1.max()) + 1
            Zu = Z * u[:, None]
            t1 = np.column_stack([np.bincount(kod1, weights=Zu[:, j], minlength=G1) for j in range(Z.shape[1])])
            V1 = G1 / (G1 - 1) * (n - 1) / sd1 * ZtZ_inv @ (t1.T @ t1) @ ZtZ_inv
        F1, p1 = _wald_f(g, V1, j_arac, sd1)
        # kısmi R²: dışsal değişkenlerin etkisi ayıklandıktan sonra araçların açıkladığı pay
        _, x_art, _ = _ekk_kat(X1, x2) if X1.shape[1] else (None, x2 - 0, None)
        Z2_art = Z2_df.values - X1 @ np.linalg.lstsq(X1, Z2_df.values, rcond=None)[0] if X1.shape[1] else Z2_df.values
        _, kalan, _ = _ekk_kat(Z2_art, x_art)
        kismi = 1 - float(kalan @ kalan) / float(x_art @ x_art)
        birinci[ic] = {
            "F": F1,
            "p": p1,
            "sd": (len(araclar), sd1),
            "kismi_r2": kismi,
            "katsayilar": pd.Series(g, index=disal + araclar),
            "standart_hatalar": pd.Series(np.sqrt(np.diag(V1)), index=disal + araclar),
        }

    from .regresyon import ekk

    ekk_model = ekk(yapisal, v, kovaryans={"klasik": "klasik", "saglam": "HC1", "kume": "kume"}[kovaryans], birimler=birimler, kume=kume)
    isimler = list(X_df.columns)
    seri = lambda val, ad: pd.Series(val, index=isimler, name=ad)  # noqa: E731
    kullanilan = [c for c in veri.columns if isinstance(c, str) and re.search(rf"(?<![\w.]){re.escape(c)}(?!\w)", yapisal)]
    return IVSonuc(
        formul=formul,
        bagimli=y_df.columns[0],
        icsel=icsel,
        araclar=araclar,
        katsayilar=seri(b, "katsayı"),
        standart_hatalar=seri(se, "std. hata"),
        t_degerleri=seri(tv, "t"),
        p_degerleri=seri(pv, "p"),
        kovaryans_matrisi=pd.DataFrame(V, index=isimler, columns=isimler),
        kovaryans_turu=kovaryans,
        kovaryans_aciklama=kov_ac,
        n=n,
        k=k,
        sse=sse,
        r2=float(r2),
        duzeltilmis_r2=float(1 - (1 - r2) * (n - 1) / sd),
        f_istatistigi=F,
        f_p_degeri=Fp,
        f_serbestlik=(len(idx), sd) if kucuk_orneklem else (len(idx),),
        kalintilar=pd.Series(e, index=y_df.index, name="kalıntı"),
        ekk=ekk_model,
        birinci_asama=birinci,
        X=X_df,
        Z=pd.DataFrame(Z, index=X_df.index, columns=disal + araclar),
        y=y_df.iloc[:, 0],
        _terimler=_terimleri_coz(X_df.design_info),
        _birimler=dict(birimler or {}),
        _veri=v[kullanilan],
        kucuk_orneklem=kucuk_orneklem,
        _etiketler=dict(veri.attrs.get("etiketler", {})),
    )


def _tanilar(m: IVSonuc) -> list[IVTesti]:
    testler = []
    for ic, b in m.birinci_asama.items():
        zayif = b["F"] < ZAYIF_ESIK
        testler.append(
            IVTesti(
                f"Birinci aşama F ({ic})",
                "Dışlanmış araçların içsel değişken üzerindeki etkisi sıfırdır.",
                b["F"],
                b["sd"],
                b["p"],
                zayif,
                (
                    f"F = {sayi(b['F'], 2)} < 10: araçlar zayıf; 2SLS tahminleri yanlı ve standart hatalar güvenilmez olabilir."
                    if zayif
                    else f"F = {sayi(b['F'], 2)} ≥ 10: araçlar güçlü (kısmi R² = {sayi(b['kismi_r2'], 3)})."
                ),
                "Zayıf" if zayif else "Güçlü",
            )
        )
    # Wu-Hausman (Durbin-Wu-Hausman, regresyon biçimi): yapısal denkleme birinci aşama kalıntıları eklenir
    X = m.X.values
    Z = m.Z.values
    y = m.y.values
    V_art = np.column_stack([m.X[ic].values - Z @ np.linalg.lstsq(Z, m.X[ic].values, rcond=None)[0] for ic in m.icsel])
    W = np.column_stack([X, V_art])
    q = V_art.shape[1]
    _, e_u, _ = _ekk_kat(W, y)
    _, e_r, _ = _ekk_kat(X, y)
    sd = len(y) - W.shape[1]
    F = ((float(e_r @ e_r) - float(e_u @ e_u)) / q) / (float(e_u @ e_u) / sd)
    p = float(stats.f.sf(F, q, sd))
    testler.append(
        IVTesti(
            "Wu-Hausman içsellik testi",
            "İçsel varsayılan değişken(ler) aslında dışsaldır (EKK tutarlıdır).",
            float(F),
            (q, sd),
            p,
            p < 0.05,
            "p < 0,05: H0 reddedildi — değişken içsel; EKK tutarsız, 2SLS kullanılmalı."
            if p < 0.05
            else "p ≥ 0,05: H0 reddedilemedi — içselliğe dair güçlü kanıt yok; EKK daha etkin olabilir.",
            "İçsel" if p < 0.05 else "Dışsal",
        )
    )
    # Durbin (1954) skor testi — Stata estat endogenous'un ilk satırı
    Xd = X
    e0 = y - Xd @ np.linalg.lstsq(Xd, y, rcond=None)[0]  # EKK kalıntıları (içsel değişkenler dışsal sayılarak)
    Z1 = np.column_stack([Z, m.X[m.icsel].values])

    def izdus(A, B):
        return B @ np.linalg.lstsq(B, A, rcond=None)[0]

    e1 = izdus(e0, Z1)
    e2 = izdus(m.kalintilar.values, Z)
    D = float((e1 @ e1 - e2 @ e2) / ((e0 @ e0) / len(y)))
    pD = float(stats.chi2.sf(D, q))
    testler.insert(
        len(testler) - 1,
        IVTesti(
            "Durbin içsellik testi",
            "İçsel varsayılan değişken(ler) aslında dışsaldır (EKK tutarlıdır).",
            D,
            (q,),
            pD,
            pD < 0.05,
            "p < 0,05: H0 reddedildi — değişken içsel; EKK tutarsız, 2SLS kullanılmalı."
            if pD < 0.05
            else "p ≥ 0,05: H0 reddedilemedi — içselliğe dair güçlü kanıt yok; EKK daha etkin olabilir.",
            "İçsel" if pD < 0.05 else "Dışsal",
        ),
    )
    # Sargan (aşırı tanımlama)
    asiri = len(m.araclar) - len(m.icsel)
    if asiri > 0:
        e = m.kalintilar.values
        _, u, _ = _ekk_kat(Z, e)
        r2 = 1 - float(u @ u) / float(np.sum((e - e.mean()) ** 2))
        S = len(e) * r2
        p = float(stats.chi2.sf(S, asiri))
        testler.append(
            IVTesti(
                "Sargan aşırı tanımlama testi",
                "Tüm araçlar geçerlidir (hata terimiyle ilişkisizdir).",
                float(S),
                (asiri,),
                p,
                p < 0.05,
                "p < 0,05: H0 reddedildi — araçlardan en az biri geçersiz olabilir."
                if p < 0.05
                else "p ≥ 0,05: H0 reddedilemedi — aşırı tanımlama kısıtları geçerli; araçların geçersizliğine dair kanıt yok.",
                "Geçersiz olabilir" if p < 0.05 else "Geçerli",
            )
        )
        L = Z.shape[1]
        B = S * (len(e) - L) / (len(e) - S)  # Basmann (1960) — Stata estat overid'in ikinci satırı
        pB = float(stats.chi2.sf(B, asiri))
        testler.append(
            IVTesti(
                "Basmann aşırı tanımlama testi",
                "Tüm araçlar geçerlidir (hata terimiyle ilişkisizdir).",
                float(B),
                (asiri,),
                pB,
                pB < 0.05,
                "p < 0,05: H0 reddedildi — araçlardan en az biri geçersiz olabilir."
                if pB < 0.05
                else "p ≥ 0,05: H0 reddedilemedi — araçların geçersizliğine dair kanıt yok.",
                "Geçersiz olabilir" if pB < 0.05 else "Geçerli",
            )
        )
    return testler


def _iv_ad_en(ad: str) -> str:
    if ad.startswith("Birinci aşama F ("):
        return "First-stage F (" + ad[len("Birinci aşama F (") :]
    return {
        "Wu-Hausman içsellik testi": "Wu-Hausman endogeneity test",
        "Sargan aşırı tanımlama testi": "Sargan overidentification test",
        "Basmann aşırı tanımlama testi": "Basmann overidentification test",
    }.get(ad, ad)


def _iv_sonuc_en(sonuc: str) -> str:
    return {"Zayıf": "Weak", "Güçlü": "Strong", "Geçersiz olabilir": "May be invalid", "Geçerli": "Valid"}.get(sonuc, sonuc)


# --------------------------------------------------------------------------- rapor
def _rapor_en(m: IVSonuc, baslik, etiketler, tablo_no, bicim):
    from . import __version__
    from .karsilastirma import _kov_en, karsilastir
    from .rapor import KAYNAKLAR, Rapor, Tablo, _Adlar, _p, _p_hucre, _tanimlayici, katsayi_paragraflari

    ad = _Adlar({**m._etiketler, **(etiketler or {})})
    ic_ad = ", ".join(ad.metin(i) for i in m.icsel)
    ar_ad = ", ".join(ad.metin(a) for a in m.araclar)
    tek = len(m.icsel) == 1
    r = Rapor(baslik, bicim=bicim, dil="en")
    sargan_cumle = ", and instrument validity with the Sargan (1958) overidentification test" if len(m.araclar) > len(m.icsel) else ""
    r.bloklar += [
        ("baslik", "Method"),
        (
            "paragraf",
            f"Because {ic_ad} may be correlated with the error term, the model was estimated by two-stage least squares (2SLS) as well as by OLS (Wooldridge, 2020). "
            f"{ar_ad} {'was' if len(m.araclar) == 1 else 'were'} used as {'the instrument' if len(m.araclar) == 1 else 'instruments'}. "
            "Instrument strength was assessed with the first-stage F statistic on the excluded instruments (Staiger and Stock, 1997), "
            f"and endogeneity with the Wu-Hausman test (Hausman, 1978; Wu, 1973){sargan_cumle}. "
            f"Standard errors: {_kov_en(m.kovaryans_aciklama)}. The calculations were done with ekonometri-tr {__version__}.",
        ),
    ]
    b, no = _tanimlayici(m, ad, tablo_no)
    r.bloklar += b
    sat, ilk = [], []
    for ic, ba in m.birinci_asama.items():
        for a in m.araclar:
            pa = 2 * stats.t.sf(abs(ba["katsayilar"][a] / ba["standart_hatalar"][a]), ba["sd"][1])
            sat.append(
                [
                    ad.yalin(ic),
                    ad.yalin(a),
                    tablo_sayi(ba["katsayilar"][a]) + yildiz(pa),
                    f"({tablo_sayi(ba['standart_hatalar'][a])})",
                    "",
                    "",
                ]
            )
        sat.append(
            [ad.yalin(ic), "Excluded instruments (joint)", "", "", f"{sayi(ba['F'], 2)} ({_p_hucre(ba['p'])})", sayi(ba["kismi_r2"], 3)]
        )
        guc = (
            "An F statistic above the rule-of-thumb threshold of 10 indicates that the instruments are strong."
            if ba["F"] >= ZAYIF_ESIK
            else "An F statistic below the rule-of-thumb threshold of 10 indicates weak instruments; the 2SLS estimates should be read with caution."
        )
        ilk.append(
            f"In the first stage for {ad.metin(ic)}, the joint F statistic on the excluded instruments is {sayi(ba['F'], 2)} ({_p(ba['p'])}) and the partial R² is {sayi(ba['kismi_r2'], 3)}. {guc}"
        )
    r.bloklar += [
        ("baslik", "First stage"),
        ("paragraf", f"First-stage estimates are reported in Table {no}. " + " ".join(ilk)),
        (
            "tablo",
            Tablo(
                no,
                "First-stage regression",
                ["Endogenous variable", "Instrument", "Coefficient", "SE", "F (p)", "Partial R²"],
                sat,
                "Each first stage includes every exogenous regressor in the model; only the excluded instruments are shown. *** p < 0.01; ** p < 0.05; * p < 0.10.",
                sol=2,
            ),
        ),
    ]
    no += 1
    testler = m.tanilar()
    wh = next(t for t in testler if t.ad.startswith("Wu-Hausman"))
    sg = next((t for t in testler if t.ad.startswith("Sargan")), None)
    sahip = "is" if tek else "are"
    c = [
        f"The diagnostic tests are reported in Table {no}.",
        f"The Wu-Hausman test (F({wh.serbestlik[0]}, {wh.serbestlik[1]}) = {sayi(wh.istatistik, 2)}; {_p(wh.p_degeri)}) "
        + (
            f"indicates that {ic_ad} {sahip} endogenous, so OLS is inconsistent and the 2SLS estimates should be preferred."
            if wh.sorun_var
            else f"does not reject the hypothesis that {ic_ad} {sahip} exogenous, so OLS is consistent and more efficient."
        ),
    ]
    if sg:
        c.append(
            f"The Sargan test (χ²({sg.serbestlik[0]}) = {sayi(sg.istatistik, 2)}; {_p(sg.p_degeri)}) "
            + (
                "suggests that at least one instrument may be invalid."
                if sg.sorun_var
                else "does not reject the overidentifying restrictions, so there is no evidence that the instruments are invalid."
            )
        )
    r.bloklar += [
        ("baslik", "Diagnostic tests"),
        ("paragraf", " ".join(c)),
        (
            "tablo",
            Tablo(
                no,
                "Instrumental-variable diagnostics",
                ["Test", "Statistic", "Degrees of freedom", "p", "Result"],
                [
                    [_iv_ad_en(t.ad), sayi(t.istatistik, 3), ", ".join(map(str, t.serbestlik)), _p_hucre(t.p_degeri), _iv_sonuc_en(t.sonuc)]
                    for t in testler
                ],
                "The practical threshold for the first-stage F is 10 (Staiger and Stock, 1997). Wu-Hausman H₀: the variable is exogenous. Sargan H₀: the instruments are valid.",
            ),
        ),
    ]
    no += 1
    k = karsilastir(m.ekk, m, adlar=["OLS", "2SLS"])
    tab = k._tablo(etiketler, no)
    tab.baslik = "OLS and 2SLS estimates"
    fark = []
    for ic in m.icsel:
        be, bi = m.ekk.katsayilar[ic], m.katsayilar[ic]
        fark.append(
            f"The coefficient on {ad.metin(ic)} is {tablo_sayi(be)} by OLS and {tablo_sayi(bi)} by 2SLS. "
            + (
                "The larger OLS estimate points to upward bias from unobserved factors."
                if be > bi
                else "The smaller OLS estimate points to downward bias, for example from measurement error."
            )
        )
    r.bloklar += [
        ("baslik", "2SLS results"),
        ("paragraf", f"OLS and 2SLS estimates are compared in Table {no}. " + " ".join(fark) + " The 2SLS findings are interpreted below."),
        ("tablo", tab),
    ]
    r.bloklar += katsayi_paragraflari(m, ad)
    r.bloklar.append(
        (
            "paragraf",
            "A causal reading of the 2SLS estimates requires that the instruments affect the outcome only through the endogenous regressor. That exclusion restriction cannot be tested in full and has to be justified by the theory.",
        )
    )
    kay = ["wooldridge", "staiger", "hausman", "wu"] + (["sargan"] if sg else [])
    r.bloklar += [("baslik", "References"), ("liste", sorted(KAYNAKLAR[x] for x in kay))]
    return r


def _rapor(m: IVSonuc, baslik, etiketler, tablo_no, dil="tr", bicim="apa"):
    from .rapor import rapor_ayari

    if dil == "en" and baslik == "Bulgular":
        baslik = "Findings"
    token = rapor_ayari(dil, bicim)
    try:
        if dil == "en":
            return _rapor_en(m, baslik, etiketler, tablo_no, bicim)
        return _rapor_tr(m, baslik, etiketler, tablo_no, bicim)
    finally:
        DIL.reset(token)


def _rapor_tr(m: IVSonuc, baslik, etiketler, tablo_no, bicim="apa"):
    from . import __version__
    from .karsilastirma import karsilastir
    from .rapor import KAYNAKLAR, Rapor, Tablo, _Adlar, _p, _p_hucre, _tanimlayici, katsayi_paragraflari

    ad = _Adlar({**m._etiketler, **(etiketler or {})})
    ic_ad = ", ".join(ad.metin(i) for i in m.icsel)
    ar_ad = ", ".join(ad.metin(a) for a in m.araclar)
    r = Rapor(baslik, bicim=bicim, dil="tr")
    r.bloklar += [
        ("baslik", "Yöntem"),
        (
            "paragraf",
            f"Modelde {ic_ad} {'değişkeninin' if len(m.icsel) == 1 else 'değişkenlerinin'} hata terimiyle ilişkili (içsel) olabileceği dikkate alınarak model, EKK'nin yanı sıra "
            f"iki aşamalı en küçük kareler (2SLS) yöntemiyle de tahmin edilmiştir (Wooldridge, 2020). Araç değişken olarak {ar_ad} kullanılmıştır. "
            "Araçların gücü birinci aşama regresyonunda dışlanmış araçların ortak anlamlılığını sınayan F istatistiğiyle (Staiger ve Stock, 1997), "
            "değişkenin içselliği Wu-Hausman testiyle (Hausman, 1978; Wu, 1973)"
            + (", araçların geçerliliği ise Sargan (1958) aşırı tanımlama testiyle" if len(m.araclar) > len(m.icsel) else "")
            + f" değerlendirilmiştir. Standart hatalar: {m.kovaryans_aciklama}. Hesaplar ekonometri-tr {__version__} ile yapılmıştır.",
        ),
    ]
    b, no = _tanimlayici(m, ad, tablo_no)
    r.bloklar += b

    # birinci aşama
    sat = []
    for ic, ba in m.birinci_asama.items():
        for a in m.araclar:
            pa = 2 * stats.t.sf(abs(ba["katsayilar"][a] / ba["standart_hatalar"][a]), ba["sd"][1])
            sat.append(
                [
                    ad.yalin(ic),
                    ad.yalin(a),
                    tablo_sayi(ba["katsayilar"][a]) + yildiz(pa),
                    f"({tablo_sayi(ba['standart_hatalar'][a])})",
                    "",
                    "",
                ]
            )
        sat.append(
            [ad.yalin(ic), "Dışlanmış araçlar (ortak)", "", "", f"{sayi(ba['F'], 2)} ({_p_hucre(ba['p'])})", sayi(ba["kismi_r2"], 3)]
        )
    ilk = []
    for ic, ba in m.birinci_asama.items():
        ilk.append(
            f"{ad.metin(ic)} için birinci aşama regresyonunda dışlanmış araçların ortak anlamlılığına ilişkin F istatistiği {sayi(ba['F'], 2)} ({_p(ba['p'])}), kısmi R² ise {sayi(ba['kismi_r2'], 3)} olarak hesaplanmıştır; "
            + (
                "F değerinin 10 eşiğinin üzerinde olması araçların güçlü olduğunu göstermektedir."
                if ba["F"] >= ZAYIF_ESIK
                else "F değerinin 10 eşiğinin altında olması zayıf araç sorununa işaret etmektedir; 2SLS tahminleri temkinli yorumlanmalıdır."
            )
        )
    r.bloklar += [
        ("baslik", "Birinci Aşama"),
        ("paragraf", f"Birinci aşama tahmin sonuçları Tablo {no} ile sunulmuştur. " + " ".join(ilk)),
        (
            "tablo",
            Tablo(
                no,
                "Birinci aşama regresyon sonuçları",
                ["İçsel değişken", "Araç", "Katsayı", "SH", "F (p)", "Kısmi R²"],
                sat,
                "Birinci aşama regresyonları modeldeki tüm dışsal değişkenleri içermektedir; tabloda yalnızca dışlanmış araçların katsayıları gösterilmiştir. *** p < 0,01; ** p < 0,05; * p < 0,10.",
                sol=2,
            ),
        ),
    ]
    no += 1

    # tanılar
    testler = m.tanilar()
    wh = next(t for t in testler if t.ad.startswith("Wu-Hausman"))
    sg = next((t for t in testler if t.ad.startswith("Sargan")), None)
    c = [
        f"Tanı testlerinin sonuçları Tablo {no} ile sunulmuştur.",
        f"Wu-Hausman testi (F({wh.serbestlik[0]}, {wh.serbestlik[1]}) = {sayi(wh.istatistik, 2)}; {_p(wh.p_degeri)}) "
        + (
            f"{ic_ad} {'değişkeninin' if len(m.icsel) == 1 else 'değişkenlerinin'} içsel olduğunu göstermektedir; bu durumda EKK tahmincisi tutarsızdır ve 2SLS tahminleri esas alınmalıdır."
            if wh.sorun_var
            else f"{ic_ad} {'değişkeninin' if len(m.icsel) == 1 else 'değişkenlerinin'} dışsal olduğu hipotezini reddetmemektedir; bu durumda EKK tahmincisi tutarlı ve daha etkindir."
        ),
    ]
    if sg:
        c.append(
            f"Sargan testi (χ²({sg.serbestlik[0]}) = {sayi(sg.istatistik, 2)}; {_p(sg.p_degeri)}) "
            + (
                "araçlardan en az birinin geçersiz olabileceğine işaret etmektedir."
                if sg.sorun_var
                else "aşırı tanımlama kısıtlarının geçerli olduğu hipotezini reddetmemektedir; araçların geçersizliğine dair kanıt bulunmamaktadır."
            )
        )
    r.bloklar += [
        ("baslik", "Tanı Testleri"),
        ("paragraf", " ".join(c)),
        (
            "tablo",
            Tablo(
                no,
                "Araç değişken tanı testleri",
                ["Test", "İstatistik", "Serbestlik", "p", "Sonuç"],
                [
                    [
                        t.ad,
                        sayi(t.istatistik, 3),
                        ", ".join(map(str, t.serbestlik)),
                        _p_hucre(t.p_degeri),
                        t.sonuc,
                    ]
                    for t in testler
                ],
                "Birinci aşama F için pratik eşik 10'dur (Staiger ve Stock, 1997). Wu-Hausman H₀: değişken dışsaldır. Sargan H₀: araçlar geçerlidir.",
            ),
        ),
    ]
    no += 1

    # EKK ve 2SLS karşılaştırması + yorum
    k = karsilastir(m.ekk, m, adlar=["EKK", "2SLS"])
    tab = k._tablo(etiketler, no)
    tab.baslik = "EKK ve 2SLS tahmin sonuçları"
    fark = []
    for ic in m.icsel:
        be, bi = m.ekk.katsayilar[ic], m.katsayilar[ic]
        fark.append(
            f"{ad.metin(ic)} değişkeninin katsayısı EKK ile {tablo_sayi(be)}, 2SLS ile {tablo_sayi(bi)} olarak tahmin edilmiştir; "
            + (
                "EKK tahmininin daha büyük olması, gözlenmeyen faktörlerin neden olduğu yukarı yönlü bir yanlılığa işaret etmektedir."
                if be > bi
                else "EKK tahmininin daha küçük olması, aşağı yönlü bir yanlılığa (ör. ölçüm hatası) işaret etmektedir."
            )
        )
    r.bloklar += [
        ("baslik", "2SLS Bulguları"),
        (
            "paragraf",
            f"EKK ve 2SLS tahmin sonuçları Tablo {no} ile karşılaştırmalı olarak sunulmuştur. "
            + " ".join(fark)
            + " Aşağıda 2SLS bulguları yorumlanmaktadır.",
        ),
        ("tablo", tab),
    ]
    r.bloklar += katsayi_paragraflari(m, ad)
    r.bloklar.append(
        (
            "paragraf",
            "2SLS tahminlerinin nedensel yorumu, araçların içsel değişken dışında bağımlı değişkeni etkilememesi (dışlama kısıtı) varsayımına dayanmaktadır; bu varsayım istatistiksel olarak tam olarak sınanamaz ve kuramsal olarak gerekçelendirilmelidir.",
        )
    )
    kay = ["wooldridge", "staiger", "hausman", "wu"] + (["sargan"] if sg else [])
    r.bloklar += [("baslik", "Kaynakça"), ("liste", sorted(KAYNAKLAR[x] for x in kay))]
    return r


__all__ = ["iv2sls", "IVSonuc", "IVTesti"]
