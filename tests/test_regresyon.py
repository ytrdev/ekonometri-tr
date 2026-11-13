"""EKK sonuçları statsmodels ile birebir karşılaştırılır (referans uygulama)."""

import numpy as np
import pandas as pd
import pytest
import statsmodels.formula.api as smf

import ekonometri as ek

FORMULLER = [
    ("ucret", "np.log(ucret) ~ egitim + tecrube + I(tecrube**2) + cinsiyet + sektor + bolge"),
    ("konut", "np.log(fiyat) ~ np.log(metrekare) + oda + bina_yasi + kat + asansor + site + il"),
    ("sinav", "puan ~ calisma_saati + I(calisma_saati**2) + devam + uyku + anne_egitim + burs"),
    ("ihracat", "np.log(ihracat) ~ np.log(reel_kur) + dunya_buyumesi + C(ceyrek)"),
]


@pytest.fixture(params=FORMULLER, ids=[f[0] for f in FORMULLER])
def ikili(request):
    ad, formul = request.param
    veri = ek.veri.yukle(ad)
    return ad, formul, veri


def test_katsayilar_ve_uyum(ikili):
    _, formul, veri = ikili
    m = ek.ekk(formul, veri)
    r = smf.ols(formul, veri).fit()
    np.testing.assert_allclose(m.katsayilar.values, r.params.values, rtol=1e-9)
    np.testing.assert_allclose(m.standart_hatalar.values, r.bse.values, rtol=1e-9)
    np.testing.assert_allclose(m.t_degerleri.values, r.tvalues.values, rtol=1e-9)
    np.testing.assert_allclose(m.p_degerleri.values, r.pvalues.values, rtol=1e-7, atol=1e-300)
    assert m.r2 == pytest.approx(r.rsquared, rel=1e-10)
    assert m.duzeltilmis_r2 == pytest.approx(r.rsquared_adj, rel=1e-10)
    assert m.f_istatistigi == pytest.approx(r.fvalue, rel=1e-9)
    assert m.f_p_degeri == pytest.approx(r.f_pvalue, rel=1e-6, abs=1e-300)
    assert m.log_olabilirlik == pytest.approx(r.llf, rel=1e-10)
    assert m.aic == pytest.approx(r.aic, rel=1e-10)
    assert m.bic == pytest.approx(r.bic, rel=1e-10)
    assert m.n == int(r.nobs) and m.sd == int(r.df_resid)
    np.testing.assert_allclose(m.kalintilar.values, r.resid.values, atol=1e-8)
    ga = m.guven_araliklari()
    np.testing.assert_allclose(ga.values, r.conf_int().values, rtol=1e-9)


@pytest.mark.parametrize("tur", ["HC0", "HC1", "HC2", "HC3"])
def test_saglam_standart_hatalar(ikili, tur):
    _, formul, veri = ikili
    m = ek.ekk(formul, veri, kovaryans=tur)
    r = smf.ols(formul, veri).fit(cov_type=tur, use_t=True)
    np.testing.assert_allclose(m.standart_hatalar.values, r.bse.values, rtol=1e-9)
    np.testing.assert_allclose(m.p_degerleri.values, r.pvalues.values, rtol=1e-7, atol=1e-300)
    assert m.f_istatistigi == pytest.approx(r.fvalue, rel=1e-8)


@pytest.mark.parametrize("gecikme", [1, 4, 8])
def test_hac_newey_west(gecikme):
    veri = ek.veri.yukle("ihracat")
    formul = "np.log(ihracat) ~ np.log(reel_kur) + dunya_buyumesi"
    m = ek.ekk(formul, veri, kovaryans="HAC", gecikme=gecikme)
    r = smf.ols(formul, veri).fit(cov_type="HAC", cov_kwds={"maxlags": gecikme, "use_correction": False}, use_t=True)
    np.testing.assert_allclose(m.standart_hatalar.values, r.bse.values, rtol=1e-9)


def test_wald_testi_statsmodels_ile_ayni():
    veri = ek.veri.yukle("ucret")
    formul = "np.log(ucret) ~ egitim + tecrube + I(tecrube**2) + cinsiyet + sektor"
    m = ek.ekk(formul, veri)
    r = smf.ols(formul, veri).fit()
    w = m.wald_testi("egitim = 0.085")
    rr = r.f_test("egitim = 0.085")
    assert w.f_istatistigi == pytest.approx(float(np.squeeze(rr.fvalue)), rel=1e-9)
    assert w.p_degeri == pytest.approx(float(rr.pvalue), rel=1e-7)
    w2 = m.wald_testi(["tecrube", "I(tecrube ** 2)"])
    assert w2.serbestlik[0] == 2


def test_tahmin_yeni_veri():
    veri = ek.veri.yukle("sinav")
    formul = "puan ~ calisma_saati + I(calisma_saati**2) + devam + burs"
    m = ek.ekk(formul, veri)
    r = smf.ols(formul, veri).fit()
    yeni = veri.head(5)
    np.testing.assert_allclose(m.tahmin(yeni).values, r.predict(yeni).values, rtol=1e-10)


def test_dogru_parametreleri_yaklasik_bulur():
    """Simüle veri, bilinen gerçek modelden üretildi: tahminler gerçek değerlere yakın olmalı."""
    m = ek.ekk("np.log(ucret) ~ egitim + tecrube + I(tecrube**2) + cinsiyet + sektor + bolge", ek.veri.yukle("ucret"))
    assert m.katsayilar["egitim"] == pytest.approx(0.085, abs=0.01)
    assert m.katsayilar["cinsiyet[T.Kadın]"] == pytest.approx(-0.13, abs=0.05)


def test_sabitsiz_model():
    veri = ek.veri.yukle("sinav")
    m = ek.ekk("puan ~ calisma_saati + devam - 1", veri)
    r = smf.ols("puan ~ calisma_saati + devam - 1", veri).fit()
    np.testing.assert_allclose(m.katsayilar.values, r.params.values, rtol=1e-9)
    assert m.r2 == pytest.approx(r.rsquared, rel=1e-10)
    assert m.f_istatistigi == pytest.approx(r.fvalue, rel=1e-8)


def test_eksik_veri_atilir():
    veri = ek.veri.yukle("sinav").copy()
    veri.loc[[0, 5, 9], "devam"] = np.nan
    m = ek.ekk("puan ~ devam + calisma_saati", veri)
    assert m.n == len(veri) - 3


def test_kukla_tuzagi_turkce_hata():
    veri = pd.DataFrame({"y": [1.0, 2, 3, 4, 5, 6], "a": [1, 0, 1, 0, 1, 0], "b": [0, 1, 0, 1, 0, 1]})
    with pytest.raises(ek.EkonometriHatasi, match="çoklu doğrusal bağlantı"):
        ek.ekk("y ~ a + b", veri)


def test_hatali_formul_turkce_hata():
    with pytest.raises(ek.EkonometriHatasi, match="Formül okunamadı"):
        ek.ekk("puan ~ olmayan_degisken", ek.veri.yukle("sinav"))


def test_gecersiz_kovaryans():
    with pytest.raises(ek.EkonometriHatasi):
        ek.ekk("puan ~ devam", ek.veri.yukle("sinav"), kovaryans="HC9")


def test_kisa_yollar_log_calisir():
    veri = ek.veri.yukle("konut")
    a = ek.ekk("log(fiyat) ~ log(metrekare)", veri)
    b = ek.ekk("np.log(fiyat) ~ np.log(metrekare)", veri)
    np.testing.assert_allclose(a.katsayilar.values, b.katsayilar.values)


def test_ozet_ve_html():
    m = ek.ekk("puan ~ devam + burs", ek.veri.yukle("sinav"))
    metin = m.ozet()
    assert "EKK Regresyon Sonuçları" in metin and "Sabit" in metin and "***" in metin
    assert "<table>" in m._repr_html_()
