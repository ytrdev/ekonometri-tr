"""Varsayım testleri statsmodels'in uygulamalarıyla birebir karşılaştırılır."""

import numpy as np
import pytest
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.stats.diagnostic as smd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson, jarque_bera

import ekonometri as ek

UCRET = "np.log(ucret) ~ egitim + tecrube + I(tecrube**2) + cinsiyet + sektor"
IHRACAT = "np.log(ihracat) ~ np.log(reel_kur) + dunya_buyumesi + C(ceyrek)"


@pytest.fixture(scope="module")
def ucret():
    v = ek.veri.yukle("ucret")
    return ek.ekk(UCRET, v), smf.ols(UCRET, v).fit()


@pytest.fixture(scope="module")
def ihracat():
    v = ek.veri.yukle("ihracat")
    return ek.ekk(IHRACAT, v), smf.ols(IHRACAT, v).fit()


def test_breusch_pagan(ucret):
    m, r = ucret
    lm, p, *_ = smd.het_breuschpagan(r.resid, r.model.exog)
    t = ek.breusch_pagan(m)
    assert t.istatistik == pytest.approx(lm, rel=1e-9)
    assert t.p_degeri == pytest.approx(p, rel=1e-6)
    assert t.sorun_var  # veri bilerek değişen varyanslı üretildi


def test_white(ucret):
    m, r = ucret
    lm, p, *_ = smd.het_white(r.resid, r.model.exog)
    t = ek.white(m)
    assert t.istatistik == pytest.approx(lm, rel=1e-8)
    assert t.p_degeri == pytest.approx(p, rel=1e-6)


def test_jarque_bera(ucret):
    m, r = ucret
    jb, p, s, k = jarque_bera(r.resid)
    t = ek.jarque_bera(m)
    assert t.istatistik == pytest.approx(jb, rel=1e-9)
    assert t.p_degeri == pytest.approx(p, rel=1e-8)
    assert t.detay["carpiklik"] == pytest.approx(s, rel=1e-9)
    assert t.detay["basiklik"] == pytest.approx(k, rel=1e-9)


def test_reset(ucret):
    m, r = ucret
    rr = smd.linear_reset(r, power=4, test_type="fitted", use_f=True)  # Stata estat ovtest: ŷ², ŷ³, ŷ⁴
    t = ek.reset(m)
    assert t.istatistik == pytest.approx(float(rr.fvalue), rel=1e-6)  # statsmodels ŷ⁴'ü ölçeklemediği için kendi sonucu ~1e-8 oynar
    assert t.p_degeri == pytest.approx(float(rr.pvalue), rel=1e-6)


def test_vif(ucret):
    m, r = ucret
    X = r.model.exog
    beklenen = [variance_inflation_factor(X, i) for i in range(1, X.shape[1])]
    np.testing.assert_allclose(ek.vif(m)["VIF"].values, beklenen, rtol=1e-9)


def test_durbin_watson(ihracat):
    m, r = ihracat
    t = ek.durbin_watson(m)
    assert t.istatistik == pytest.approx(durbin_watson(r.resid), rel=1e-12)
    assert t.sorun_var  # ihracat verisinde bilerek AR(1) hata var


@pytest.mark.parametrize("gecikme", [1, 2, 4])
def test_breusch_godfrey(ihracat, gecikme):
    m, r = ihracat
    lm, p, *_ = smd.acorr_breusch_godfrey(r, nlags=gecikme)
    t = ek.breusch_godfrey(m, gecikme)
    assert t.istatistik == pytest.approx(lm, rel=1e-8)
    assert t.p_degeri == pytest.approx(p, rel=1e-6)
    assert t.sorun_var


def test_rapor_zaman_serisini_tanir(ihracat):
    m, _ = ihracat
    rapor = m.varsayimlar()
    adlar = [t.ad for t in rapor.testler]
    assert "Durbin-Watson" in adlar and any(a.startswith("Breusch-Godfrey") for a in adlar)
    assert "otokorelasyon" in str(rapor).lower()


def test_rapor_yatay_kesitte_otokorelasyonu_atlar(ucret):
    m, _ = ucret
    rapor = m.varsayimlar()
    assert all("Durbin" not in t.ad for t in rapor.testler)
    assert len(rapor.sorunlar) >= 1
    assert "<table>" in rapor._repr_html_()
    assert set(rapor.tablo().columns) >= {"Test", "p", "Sonuç"}


def test_saglam_hata_kullanilinca_oneri_degisir():
    m = ek.ekk(UCRET, ek.veri.yukle("ucret"), kovaryans="HC1")
    bp = [t for t in m.varsayimlar().testler if t.ad == "Breusch-Pagan"][0]
    assert "Zaten sağlam" in bp.oneri


def test_homoskedastik_veride_bp_sorun_gormez():
    rng = np.random.default_rng(1)
    import pandas as pd

    x = rng.normal(size=2000)
    v = pd.DataFrame({"x": x, "y": 1 + 2 * x + rng.normal(size=2000)})
    t = ek.breusch_pagan(ek.ekk("y ~ x", v))
    assert not t.sorun_var
    _ = sm  # import kullanılıyor


def test_breusch_pagan_stata_varsayilani(ucret):
    """Stata 'estat hettest' (tahmin edilen değerler, normal hata): statsmodels robust=False ile aynı."""
    m, r = ucret
    Z = np.column_stack([np.ones(m.n), r.fittedvalues])
    lm, p, *_ = smd.het_breuschpagan(r.resid, Z, robust=False)
    t = ek.breusch_pagan(m, "tahmin", koenker=False)
    assert np.isclose(t.istatistik, lm) and np.isclose(t.p_degeri, p) and t.detay["serbestlik"] == 1
    lm2, *_ = smd.het_breuschpagan(r.resid, r.model.exog, robust=False)
    assert np.isclose(ek.breusch_pagan(m, koenker=False).istatistik, lm2)
    lm3, *_ = smd.het_breuschpagan(r.resid, np.column_stack([np.ones(m.n), m.X["egitim"]]))
    assert np.isclose(ek.breusch_pagan(m, ["egitim"]).istatistik, lm3)
