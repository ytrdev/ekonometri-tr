"""Logit / probit: statsmodels ile birebir karşılaştırma, marjinal etkiler, yorum ve rapor."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

import ekonometri as ek

FORMUL = "gecti ~ calisma_saati + I(calisma_saati**2) + devam + burs + anne_egitim"


@pytest.fixture(scope="module")
def veri():
    return ek.veri.yukle("sinav")


@pytest.mark.parametrize("tur", ["logit", "probit"])
def test_statsmodels_ile_ayni(veri, tur):
    m = getattr(ek, tur)(FORMUL, veri)
    r = getattr(smf, tur)(FORMUL, veri).fit(disp=0)
    assert np.allclose(m.katsayilar.values, r.params.values, rtol=1e-8)
    assert np.allclose(m.standart_hatalar.values, r.bse.values, rtol=1e-7)
    assert np.isclose(m.log_olabilirlik, r.llf) and np.isclose(m.pseudo_r2, r.prsquared)
    assert np.isclose(m.lr_istatistigi, r.llr) and np.isclose(m.aic, r.aic) and np.isclose(m.bic, r.bic)
    rs = getattr(smf, tur)(FORMUL, veri).fit(disp=0, cov_type="HC0")
    n = len(veri)  # Stata vce(robust) çarpanı N/(N−1)
    assert np.allclose(
        getattr(ek, tur)(FORMUL, veri, kovaryans="saglam").standart_hatalar.values, rs.bse.values * np.sqrt(n / (n - 1)), rtol=1e-7
    )


@pytest.mark.parametrize("tur", ["logit", "probit"])
def test_marjinal_etkiler_statsmodels_ile_ayni(veri, tur):
    f = "gecti ~ calisma_saati + devam + burs + anne_egitim"
    ame = getattr(ek, tur)(f, veri).marjinal_etkiler()
    sm = getattr(smf, tur)(f, veri).fit(disp=0).get_margeff(dummy=True).summary_frame()  # burs metin → kategorik
    for ek_ad, sm_ad in [
        ("calisma_saati", "calisma_saati"),
        ("devam", "devam"),
        ("anne_egitim", "anne_egitim"),
        ("burs: Evet", "burs[T.Evet]"),
    ]:
        assert np.isclose(ame.loc[ek_ad, "AME"], sm.loc[sm_ad, "dy/dx"], rtol=1e-6)
        assert np.isclose(ame.loc[ek_ad, "std. hata"], sm.loc[sm_ad, "Std. Err."], rtol=1e-4)


def test_kare_terimli_marjinal_etki_ham_degiskenden(veri):
    """Kare terim varken AME, β1 + 2·β2·x türevinin ortalaması olmalı (terim başına değil, değişken başına)."""
    m = ek.logit(FORMUL, veri)
    b = m.katsayilar
    xb = m.X.values @ b.values
    p = 1 / (1 + np.exp(-xb))
    x = veri["calisma_saati"].values
    beklenen = np.mean(p * (1 - p) * (b["calisma_saati"] + 2 * b["I(calisma_saati ** 2)"] * x))
    ame = m.marjinal_etkiler()
    assert np.isclose(ame.loc["calisma_saati", "AME"], beklenen, rtol=1e-6)
    assert "I(calisma_saati ** 2)" not in ame.index


def test_yorum_odds_siniflandirma(veri):
    m = ek.logit("gecti ~ calisma_saati + burs", veri, birimler={"calisma_saati": "saat"})
    y = str(m.yorumla())
    assert "'calisma_saati' 1 saat artınca 'gecti' = 1 olma olasılığı ortalama" in y and "yüzde puan" in y
    assert "'Evet' grubunun" in y and "('Hayır')" in y
    oo = m.odds_oranlari()
    assert np.isclose(oo.loc["calisma_saati", "odds oranı"], np.exp(m.katsayilar["calisma_saati"]))
    with pytest.raises(ek.EkonometriHatasi):
        ek.probit("gecti ~ calisma_saati", veri).odds_oranlari()
    s = m.siniflandirma()
    assert int(s.values.sum()) == m.n
    assert np.isclose(s.attrs["dogruluk"], np.mean((m.olasiliklar >= 0.5) == (m.y == 1)))
    t = m.tahmin(pd.DataFrame({"calisma_saati": [0, 30], "burs": ["Hayır", "Evet"]}))
    assert t.iloc[0] < t.iloc[1] and 0 < t.iloc[0] < 1


def test_rapor(veri, tmp_path):
    m = ek.logit("gecti ~ calisma_saati + devam + burs", veri)
    r = m.rapor(tmp_path / "logit.docx", etiketler={"calisma_saati": "Çalışma süresi"})
    metin = str(r)
    assert [t.no for t in r.tablolar] == [1, 2, 3, 4, 5]
    assert "Ortalama marjinal etkiler" in metin and "Odds oranı" in metin and "McFadden, D. (1974)" in metin
    assert "Çalışma süresi değişkeni 1 birim arttığında 'gecti' = 1 olma olasılığı ortalama" in metin
    assert "Sınıflandırma tablosu" in metin
    assert (tmp_path / "logit.docx").stat().st_size > 5000
    assert "Odds oranı" not in str(ek.probit("gecti ~ calisma_saati", veri).rapor())


def test_karsilastirma_logit_probit(veri):
    k = ek.karsilastir(ek.logit("gecti ~ calisma_saati + burs", veri), ek.probit("gecti ~ calisma_saati + burs", veri))
    t = k.tablo()
    assert "McFadden sözde R²" in list(t["Değişken"])
    assert "Logit" in k._tablo().notu and "Probit" in k._tablo().notu
    assert "sözde R²" in str(k.rapor())


def test_hatalar(veri):
    with pytest.raises(ek.EkonometriHatasi, match="0 ve 1"):
        ek.logit("puan ~ calisma_saati", veri)
    with pytest.raises(ek.EkonometriHatasi, match="ayrışma"):
        d = pd.DataFrame({"y": [0, 0, 0, 0, 1, 1, 1, 1], "x": [1, 2, 3, 4, 5, 6, 7, 8]})
        ek.logit("y ~ x", d)
    with pytest.raises(ek.EkonometriHatasi):
        ek.logit("gecti ~ calisma_saati", veri, kovaryans="HC1")
