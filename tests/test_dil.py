"""İngilizce rapor: panel, logit/probit, 2SLS, durağanlık, hata düzeltme, Granger, karşılaştırma, korelasyon."""

import numpy as np
import pandas as pd

import ekonometri as ek


def test_duraganlik_ve_granger_ingilizce():
    rng = np.random.default_rng(2)
    n = 80
    veri = pd.DataFrame({"a": np.cumsum(rng.normal(size=n)), "b": np.cumsum(rng.normal(size=n))})
    tr = str(ek.duraganlik(veri).rapor())
    en = str(ek.duraganlik(veri).rapor(dil="en"))
    assert "Birim kök testi sonuçları" in tr and "durağan" in tr
    assert "Unit root test results" in en and "stationary" in en and "Birim kök" not in en
    gtr = str(ek.granger(veri["a"], veri["b"], gecikme=1).rapor())
    gen = str(ek.granger(veri["a"], veri["b"], gecikme=1).rapor(dil="en"))
    assert "Granger nedeni" in gtr
    assert "Granger-cause" in gen and "Null hypothesis" in gen and "Granger nedeni" not in gen


def test_hata_duzeltme_ingilizce():
    rng = np.random.default_rng(5)
    n = 80
    x = np.cumsum(rng.normal(size=n))
    d = pd.DataFrame({"x": x, "z": np.cumsum(rng.normal(size=n))})
    h = ek.hata_duzeltme("z ~ x", d, gecikme=0)
    assert "yorumlanamaz" in str(h.rapor())
    en = str(h.rapor(dil="en"))
    assert "cannot be interpreted" in en and "Error-correction" in en and "yorumlanamaz" not in en


def test_karsilastirma_ve_korelasyon_ingilizce():
    veri = ek.veri.yukle("ucret")
    m1 = ek.ekk("np.log(ucret) ~ egitim", veri)
    m2 = ek.ekk("np.log(ucret) ~ egitim + tecrube", veri)
    k = ek.karsilastir(m1, m2, adlar=["Temel", "Geniş"])
    assert "Tablo 1 ile" in str(k.rapor()) and "Sabit" in k.tablo()["Değişken"].tolist()
    en = str(k.rapor(dil="en"))
    assert "Table 1" in en and "Adjusted R²" in en and "Constant" in en and "Tablo" not in en
    kor = ek.korelasyon(veri, ["ucret", "egitim", "tecrube"])
    assert "korelasyon katsayıları" in str(kor.rapor())
    ken = str(kor.rapor(dil="en"))
    assert "correlation" in ken and "korelasyon" not in ken


def test_logit_probit_ingilizce():
    veri = ek.veri.yukle("sinav")
    logit = ek.logit("gecti ~ calisma_saati", veri)
    probit = ek.probit("gecti ~ calisma_saati", veri)
    assert "sözde R²" in str(logit.rapor()) and "Odds oranı" not in str(probit.rapor())
    len_ = str(logit.rapor(dil="en"))
    pen = str(probit.rapor(dil="en"))
    assert "pseudo R²" in len_ and "Odds ratio" in len_ and "percentage points" in len_
    assert "pseudo R²" in pen and "Odds ratio" not in pen and "Average marginal effects" in pen


def test_panel_ve_iv_ingilizce():
    firma = ek.veri.yukle("firma")
    a = ek.panel_analizi("np.log(satis) ~ np.log(calisan) + arge", firma, birim="firma", zaman="yil")
    tr = str(a.rapor())
    en = str(a.rapor(dil="en"))
    assert "## Model Seçimi" in tr and "Hausman testi" in tr
    assert "Model selection" in en and "Fixed effects" in en and "Model Seçimi" not in en and "Hausman testi" not in en
    getiri = ek.veri.yukle("getiri")
    iv = ek.iv2sls("np.log(ucret) ~ tecrube + [egitim ~ uzaklik]", getiri)
    assert "Birinci Aşama" in str(iv.rapor())
    iven = str(iv.rapor(dil="en"))
    assert "First stage" in iven and "two-stage least squares" in iven and "Wu-Hausman" in iven and "Birinci Aşama" not in iven
