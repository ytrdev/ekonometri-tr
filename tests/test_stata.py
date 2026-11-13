"""Stata komut arayüzü: komutlar doğrudan API ile aynı sonucu vermeli; Stata ifade kuralları (eksik değer, etiketler) korunmalı."""

import numpy as np
import pandas as pd
import pytest

import ekonometri as ek


@pytest.fixture
def s():
    return ek.Stata(yazdir=False)


def test_regress_robust_ve_donusumler(s):
    s("sysuse ucret")
    s("gen lnw = ln(ucret)")
    s("gen tec2 = tecrube^2")
    m = s("regress lnw egitim tecrube tec2 i.cinsiyet, vce(robust)")
    d = ek.ekk("np.log(ucret) ~ egitim + tecrube + I(tecrube**2) + C(cinsiyet)", ek.veri.yukle("ucret"), kovaryans="HC1")
    assert np.allclose(m.katsayilar.values, d.katsayilar.values) and np.allclose(m.standart_hatalar.values, d.standart_hatalar.values)
    # ln() ve ^2 ile üretilen değişkenler yorumda log ve kare terim olarak tanınır
    assert "dönüm noktası" in str(s("rapor")) or "ters U" in str(m.yorumla())
    assert "tecrube²" in str(m)


def test_faktor_degisken_dili(s):
    s("sysuse ucret")
    m = s("reg ucret egitim c.tecrube##c.tecrube ib(last).bolge i.cinsiyet#c.egitim")
    sutunlar = list(m.katsayilar.index)
    assert "I(tecrube**2)" in sutunlar or "I(tecrube ** 2)" in sutunlar
    assert not any("Doğu Anadolu" in c for c in sutunlar)  # ib(last): son düzey temel
    assert any(c.startswith("C(cinsiyet)") and ":egitim" in c for c in sutunlar)


def test_if_in_ve_noconstant(s):
    s("sysuse ucret")
    m = s('regress ucret egitim if sektor == "Kamu" & tecrube > 5')
    v = ek.veri.yukle("ucret")
    alt = v[(v.sektor == "Kamu") & (v.tecrube > 5)]
    assert m.n == len(alt) and np.allclose(m.katsayilar.values, ek.ekk("ucret ~ egitim", alt).katsayilar.values)
    assert s("regress ucret egitim in 1/100").n == 100
    assert "Intercept" not in s("regress ucret egitim, noconstant").katsayilar.index


def test_kumelenmis_ve_test(s):
    s("sysuse firma")
    m = s("regress satis calisan arge, vce(cluster firma)")
    d = ek.ekk("satis ~ calisan + arge", ek.veri.yukle("firma"), kume="firma")
    assert np.allclose(m.standart_hatalar.values, d.standart_hatalar.values) and m.cikarim_sd == 149
    w = s("test calisan arge")
    assert w.serbestlik == (2, 149)


def test_stata_eksik_deger_kurallari(s):
    s.veri = pd.DataFrame({"x": [1.0, np.nan, 5.0, 200.0], "y": [1, 2, 3, 4]})
    # Stata'da eksik değer +∞'dur: 'x > 100' eksik gözlemi de seçer
    assert s("count if x > 100") == 2
    assert s("count if x > 100 & !missing(x)") == 1
    assert s("count if x == .") == 1 and s("count if x < .") == 3
    s("gen z = x / (y - 2)")  # sıfıra bölme → eksik
    assert np.isnan(s.veri["z"].iloc[1]) and s.veri["z"].iloc[0] == -1
    s("gen buyuk = x > 3")
    assert s.veri["buyuk"].tolist() == [0.0, 1.0, 1.0, 1.0]
    s("replace x = 0 if missing(x)")
    assert s.veri["x"].tolist() == [1.0, 0.0, 5.0, 200.0]
    s("gen k = cond(y > 2, 1, 0)")
    assert s.veri["k"].tolist() == [0, 0, 1, 1]
    assert s("count if inlist(y, 1, 4)") == 2 and s("count if inrange(y, 2, 3)") == 2


def test_deger_etiketleri_ve_dta(tmp_path, s):
    d = pd.DataFrame({"ucret": [10.0, 20, 30, 40, 50, 60], "cins": [1, 2, 1, 2, 1, 2]})
    pd.io.stata.StataWriterUTF8(
        tmp_path / "a.dta", d, write_index=False, value_labels={"cins": {1: "Erkek", 2: "Kadın"}}, version=118
    ).write_file()
    s(f'use "{tmp_path / "a.dta"}", clear')
    assert s("count if cins == 2") == 3  # kodla karşılaştırma etikete çevrilir
    assert s('count if cins == "Kadın"') == 3
    m = s("regress ucret ib2.cins")
    assert "C(cins)[T.Erkek]" in m.katsayilar.index  # temel düzey: kod 2 (Kadın)


def test_egen_ve_bysort(s):
    s.veri = pd.DataFrame({"g": ["a", "a", "b", "b", "b"], "x": [1.0, 3, 2, 4, 6]})
    s("bysort g: egen ort = mean(x)")
    s("egen toplam = total(x)")
    assert s.veri["ort"].tolist() == [2, 2, 4, 4, 4] and s.veri["toplam"].iloc[0] == 16


def test_logit_margins(s):
    s("sysuse sinav")
    m = s("logit gecti calisma_saati devam i.burs, vce(robust)")
    d = ek.logit("gecti ~ calisma_saati + devam + C(burs)", ek.veri.yukle("sinav"), kovaryans="saglam")
    assert np.allclose(m.standart_hatalar.values, d.standart_hatalar.values)
    ame = s("margins, dydx(*)")
    assert {"calisma_saati", "devam", "burs"} <= set(ame["değişken"])
    assert len(s("margins, dydx(devam)")) == 1


def test_xtreg_hausman(s):
    s("sysuse firma")
    s("gen lns = ln(satis)")
    s("gen lnk = ln(sermaye)")
    with pytest.raises(ek.EkonometriHatasi, match="xtset"):
        s("xtreg lns lnk arge, fe")
    s("xtset firma yil")
    fe = s("xtreg lns lnk arge, fe")
    s("estimates store fe")
    re_ = s("xtreg lns lnk arge")  # Stata varsayılanı: re
    s("estimates store re")
    assert fe.yontem == "sabit" and re_.yontem == "rassal" and re_.ist_adi == "z"
    v = ek.veri.yukle("firma")
    d = ek.panel("np.log(satis) ~ np.log(sermaye) + arge", v, birim="firma", zaman="yil", yontem="sabit")
    assert np.allclose(fe.katsayilar.values, d.katsayilar.values)
    h = s("hausman fe re")
    assert np.isclose(h.istatistik, ek.hausman(fe, re_, sigmamore=False).istatistik)
    c = s("xtreg lns lnk arge, fe vce(robust)")
    assert c.kovaryans_turu == "kume" and c.cikarim_sd == 149
    k = s("esttab fe re")
    assert k.adlar == ["fe", "re"]


def test_ivregress(s):
    s("sysuse getiri")
    s("gen lnw = ln(ucret)")
    m = s("ivregress 2sls lnw tecrube i.cinsiyet (egitim = uzaklik kardes)")
    d = ek.iv2sls("np.log(ucret) ~ tecrube + C(cinsiyet) + [egitim ~ uzaklik + kardes]", ek.veri.yukle("getiri"))
    assert np.allclose(m.katsayilar.values, d.katsayilar.values) and np.allclose(m.standart_hatalar.values, d.standart_hatalar.values)
    assert m.ist_adi == "z" and s("ivregress 2sls lnw tecrube (egitim = uzaklik), small").ist_adi == "t"
    with pytest.raises(ek.EkonometriHatasi, match="tam tanımlı"):
        s("estat overid")  # son model tam tanımlı (Stata da hata verir)
    s("ivregress 2sls lnw tecrube (egitim = uzaklik kardes)")
    assert [t.ad.split()[0] for t in s("estat overid")] == ["Sargan", "Basmann"]
    assert [t.ad.split()[0] for t in s("estat endogenous")] == ["Durbin", "Wu-Hausman"]


def test_zaman_serisi_dfuller_ve_operatorler(s):
    s.veri = pd.DataFrame({"t": range(1, 121), "y": np.cumsum(np.random.default_rng(1).normal(size=120))})
    with pytest.raises(ek.EkonometriHatasi, match="tsset"):
        s("dfuller y")
    s("tsset t")
    r = s("dfuller y, lags(2)")
    assert r.gecikme == 2 and np.isclose(r.istatistik, ek.adf(s.veri["y"], gecikme=2).istatistik)
    m = s("regress D.y L.D.y L2.y")
    assert m.n == 118  # D.y ve L2.y 3. gözlemden başlar
    assert "LD.y" in str(m) and "L2.y" in str(m)


def test_do_dosyasi_yorumlar_ve_devam(tmp_path, s):
    (tmp_path / "analiz.do").write_text(
        """* yorum satırı
sysuse ucret   // satır sonu yorumu
/* çok satırlı
   yorum */
regress ucret egitim ///
    tecrube, robust
""",
        encoding="utf-8",
    )
    m = s.do(tmp_path / "analiz.do")
    assert list(m.katsayilar.index) == ["Intercept", "egitim", "tecrube"] and m.kovaryans_turu == "HC1"


def test_kisaltmalar_ve_hatalar(s):
    s("sysuse ucret")
    assert s("su egitim").loc["egitim", "N"] == 1500
    assert s("reg ucret egi").katsayilar.index[1] == "egitim"  # değişken adı kısaltması
    with pytest.raises(ek.EkonometriHatasi, match="desteklenmiyor"):
        s("xtgls y x")
    with pytest.raises(ek.EkonometriHatasi, match="bulunamadı"):
        s("reg ucret yokdegisken")
    with pytest.raises(ek.EkonometriHatasi, match="already defined"):
        s("gen egitim = 1")
    s("capture gen egitim = 1")  # capture hatayı yutar


def test_tek_satir_kullanim():
    m = ek.stata("regress ucret egitim, vce(hc3)", ek.veri.yukle("ucret"), yazdir=False)
    assert m.kovaryans_turu == "HC3"
