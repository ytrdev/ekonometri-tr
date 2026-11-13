"""Durağanlık ve eşbütünleşme testleri: statsmodels ile birebir karşılaştırma, kararlar ve raporlar."""

import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint, grangercausalitytests
from statsmodels.tsa.stattools import kpss as sm_kpss

import ekonometri as ek

DET = [("sabit", "c"), ("trend", "ct"), ("yok", "n")]


@pytest.fixture(scope="module")
def seriler():
    v = ek.veri.yukle("ihracat")
    return {
        "ln_ihracat": np.log(v["ihracat"]),
        "reel_kur": v["reel_kur"],
        "dunya": v["dunya_buyumesi"],
        "fark": np.log(v["ihracat"]).diff().dropna(),
    }


@pytest.mark.parametrize("det,r", DET)
@pytest.mark.parametrize("olcut", ["aic", "bic"])
def test_adf_statsmodels_ile_ayni(seriler, det, r, olcut):
    for s in seriler.values():
        a = ek.adf(s, det, bilgi_olcutu=olcut)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = adfuller(s.values, regression=r, autolag=olcut.upper())
        assert a.gecikme == b[2] and a.n == b[3]
        assert np.isclose(a.istatistik, b[0], rtol=1e-9) and np.isclose(a.p_degeri, b[1], rtol=1e-8, atol=1e-15)
        for k, deger in a.kritik.items():
            assert np.isclose(deger, b[4][k[1:] + "%"])


@pytest.mark.parametrize("det,r", DET)
def test_adf_sabit_gecikme(seriler, det, r):
    s = seriler["reel_kur"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = adfuller(s.values, regression=r, maxlag=4, autolag=None)
    a = ek.adf(s, det, gecikme=4)
    assert a.gecikme == 4 and np.isclose(a.istatistik, b[0]) and np.isclose(a.p_degeri, b[1])


@pytest.mark.parametrize("det,r", DET[:2])
def test_kpss_statsmodels_ile_ayni(seriler, det, r):
    for s in seriler.values():
        a = ek.kpss(s, det)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = sm_kpss(s.values, regression=r, nlags="auto")
        assert a.gecikme == b[2] and np.isclose(a.istatistik, b[0], rtol=1e-10) and np.isclose(a.p_degeri, b[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = sm_kpss(seriler["reel_kur"].values, regression=r, nlags=5)
    assert np.isclose(ek.kpss(seriler["reel_kur"], det, gecikme=5).istatistik, b[0])


def test_engle_granger_statsmodels_ile_ayni():
    v = ek.veri.yukle("ihracat")
    y, x, z = np.log(v["ihracat"]).values, np.log(v["reel_kur"]).values, v["dunya_buyumesi"].values
    for formul, X in (
        ("np.log(ihracat) ~ np.log(reel_kur)", x),
        ("np.log(ihracat) ~ np.log(reel_kur) + dunya_buyumesi", np.column_stack([x, z])),
    ):
        e = ek.engle_granger(formul, v)
        c = coint(y, X)
        assert np.isclose(e.adf.istatistik, c[0]) and np.isclose(e.p_degeri, c[1])
        assert np.allclose(list(e.kritik.values()), c[2])


def test_simule_kararlar():
    rng = np.random.default_rng(42)
    n = 300
    rw = pd.Series(np.cumsum(rng.normal(size=n)), name="rw")
    gurultu = pd.Series(rng.normal(size=n), name="g")
    assert not ek.adf(rw).duragan and ek.adf(gurultu).duragan
    assert not ek.kpss(rw).duragan and ek.kpss(gurultu).duragan
    x = np.cumsum(rng.normal(size=n))
    d = pd.DataFrame({"x": x, "y": 2 + 0.8 * x + rng.normal(size=n), "z": np.cumsum(rng.normal(size=n))})
    assert ek.engle_granger("y ~ x", d).esbutunlesik
    assert not ek.engle_granger("z ~ x", d).esbutunlesik
    dr = ek.duraganlik(d, ["x", "y"])
    assert dr.derece("x") == "I(1)" and dr.derece("y") == "I(1)"
    assert "eşbütünleşme testleriyle sınanmasına" in str(dr.rapor())


def test_duraganlik_tablosu_ve_rapor(tmp_path):
    v = ek.veri.yukle("ihracat")
    d = ek.duraganlik(v, ["ihracat", "reel_kur", "dunya_buyumesi"], log=["ihracat", "reel_kur"], etiketler={"reel_kur": "Reel kur"})
    t = d.tablo()
    assert list(t["Değişken"]) == ["ln(ihracat)", "ln(Reel kur)", "dunya_buyumesi"]
    assert list(t["Karar"]) == ["I(1)", "I(1)", "I(0)"]
    assert t.iloc[0, 1].endswith(f"[{d.sonuclar['ln(ihracat)']['duzey']['adf'].gecikme}]")
    r = d.rapor(tmp_path / "d.docx", tablo_no=3)
    metin = str(r)
    assert "Tablo 3 ile" in metin and "ARDL sınır testi" in metin and "Kwiatkowski, D." in metin
    assert (tmp_path / "d.docx").exists()


def test_engle_granger_raporu():
    rng = np.random.default_rng(3)
    x = np.cumsum(rng.normal(size=200))
    d = pd.DataFrame({"x": x, "y": 1 + 0.5 * x + rng.normal(size=200)})
    metin = str(ek.engle_granger("y ~ x", d).rapor())
    assert "reddedilmektedir" in metin and "x = 0,5" in metin and "Engle, R. F." in metin


def _sm_granger(etkilenen, neden, p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sonuc = grangercausalitytests(np.column_stack([np.asarray(etkilenen, float), np.asarray(neden, float)]), maxlag=p)
    return sonuc[p][0]


def test_granger_statsmodels_ile_ayni():
    v = ek.veri.yukle("ihracat")
    y, x = np.log(v["ihracat"]), np.log(v["reel_kur"])
    for p in (1, 2, 4):
        g = ek.granger(y, x, gecikme=p)
        bir = _sm_granger(y, x, p)
        iki = _sm_granger(x, y, p)
        assert g.secim == "sabit" and g.p_degeri == g.yonler[0].p_degeri
        for yon, sm_sonuc in ((g.yonler[0], bir), (g.yonler[1], iki)):
            F, pv, df_payda, df_pay = sm_sonuc["ssr_ftest"]
            chi2, chi2_p, _ = sm_sonuc["ssr_chi2test"]
            assert yon.serbestlik == (df_pay, df_payda)
            assert np.isclose(yon.F, F, rtol=1e-10) and np.isclose(yon.p_degeri, pv, rtol=1e-10, atol=1e-15)
            assert np.isclose(yon.chi2, chi2, rtol=1e-10) and np.isclose(yon.chi2_p, chi2_p, rtol=1e-10, atol=1e-15)


def test_granger_otomatik_gecikme_ve_tek_yon():
    v = ek.veri.yukle("ihracat")
    y, x = np.log(v["ihracat"]), np.log(v["reel_kur"])
    g = ek.granger(y, x, cift=False, bilgi_olcutu="bic")
    assert g.secim == "BIC" and len(g.yonler) == 1
    sm_sonuc = _sm_granger(y, x, g.gecikme)
    assert np.isclose(g.yonler[0].F, sm_sonuc["ssr_ftest"][0])


def test_granger_simulasyon_yonu():
    rng = np.random.default_rng(1)
    n = 500
    x = rng.normal(size=n)
    y = np.zeros(n)
    for t in range(1, n):
        y[t] = 0.3 * y[t - 1] + 0.9 * x[t - 1] + rng.normal()
    g = ek.granger(pd.Series(y, name="y"), pd.Series(x, name="x"), gecikme=1)
    assert g.yonler[0].nedensel and g.yonler[0].p_degeri < 1e-6
    assert not g.yonler[1].nedensel and g.yonler[1].p_degeri > 0.05


def test_granger_raporu(tmp_path):
    v = ek.veri.yukle("ihracat")
    r = ek.granger(v["ihracat"], v["reel_kur"], gecikme=2)
    metin = str(r.rapor(tmp_path / "g.docx", etiketler={"reel_kur": "Reel kur"}))
    assert "Granger, C. W. J. (1969)" in metin and "Reel kur" in metin and "yapısal" in metin
    assert (tmp_path / "g.docx").exists()


def _ecm_dgp(n=400, seed=7):
    rng = np.random.default_rng(seed)
    x = np.cumsum(rng.normal(size=n))
    y = np.empty(n)
    y[0] = 2 + 0.5 * x[0]
    for t in range(1, n):
        y[t] = y[t - 1] - 0.3 * (y[t - 1] - 2 - 0.5 * x[t - 1]) + 0.4 * (x[t] - x[t - 1]) + rng.normal()
    return pd.DataFrame({"y": y, "x": x})


def _kisa_donem_ols(y, x, p, es_zamanli):
    """Uzun dönem tüm örneklemde; kısa dönem t = p+1 .. n-1. Sütun sırası kütüphane ile aynı."""
    n = len(y)
    b_uzun, *_ = np.linalg.lstsq(np.column_stack([np.ones(n), x]), y, rcond=None)
    u = y - np.column_stack([np.ones(n), x]) @ b_uzun
    t = np.arange(p + 1, n)
    cols = [np.ones(len(t)), u[t - 1]]
    for i in range(1, p + 1):
        cols.append(y[t - i] - y[t - i - 1])
    for j in range(x.shape[1]):
        for lag in range(0 if es_zamanli else 1, p + 1):
            cols.append(x[t - lag, j] - x[t - lag - 1, j])
    W = np.column_stack(cols)
    dy = y[t] - y[t - 1]
    return sm.OLS(dy, W).fit()


def test_hata_duzeltme_ols_ile_ayni():
    d = _ecm_dgp()
    for p, es_zamanli in ((0, True), (1, True), (1, False)):
        ecm = ek.hata_duzeltme("y ~ x", d, gecikme=p, es_zamanli=es_zamanli)
        sm_model = _kisa_donem_ols(d["y"].to_numpy(), d[["x"]].to_numpy(), p, es_zamanli)
        assert np.allclose(ecm.kisa_donem.katsayilar.to_numpy(), sm_model.params, rtol=1e-8, atol=1e-8)
        assert np.allclose(ecm.kisa_donem.standart_hatalar.to_numpy(), sm_model.bse, rtol=1e-8, atol=1e-8)
        assert np.isclose(ecm.lamda, sm_model.params[1])
    ecm = ek.hata_duzeltme("y ~ x", d, gecikme=0)
    assert ecm.esbutunlesme.esbutunlesik and ecm.lamda < -0.15
    assert abs(ecm.uzun_donem.katsayilar["x"] - 0.5) < 0.1
    assert ecm.yari_omur is not None and "yarısı yaklaşık" in ecm._ect_cumle()


def test_hata_duzeltme_iki_aciklayici_ve_rapor(tmp_path):
    rng = np.random.default_rng(4)
    n = 250
    x = np.cumsum(rng.normal(size=n))
    z = np.cumsum(rng.normal(size=n))
    y = np.empty(n)
    y[0] = x[0] + z[0]
    for t in range(1, n):
        denge = y[t - 1] - x[t - 1] - 0.5 * z[t - 1]
        y[t] = y[t - 1] - 0.25 * denge + 0.2 * (x[t] - x[t - 1]) + rng.normal()
    d = pd.DataFrame({"y": y, "x": x, "z": z})
    ecm = ek.engle_granger("y ~ x + z", d).hata_duzeltme(gecikme=1, es_zamanli=True)
    sm_model = _kisa_donem_ols(d["y"].to_numpy(), d[["x", "z"]].to_numpy(), 1, True)
    assert list(ecm.kisa_donem.katsayilar.index) == ["Intercept", "ect_l1", "d_y_l1", "d_x0_l0", "d_x0_l1", "d_x1_l0", "d_x1_l1"]
    assert np.allclose(ecm.kisa_donem.katsayilar.to_numpy(), sm_model.params, rtol=1e-8, atol=1e-8)
    metin = str(ecm.rapor(tmp_path / "e.docx", etiketler={"x": "Kur"}))
    assert "Kur = " in metin and "Engle, R. F." in metin and "ECT(-1)" in metin
    assert (tmp_path / "e.docx").exists()


def test_hata_duzeltme_esbutunlesme_yoksa_uyari():
    rng = np.random.default_rng(42)
    n = 300
    np.cumsum(rng.normal(size=n))
    rng.normal(size=n)
    x = np.cumsum(rng.normal(size=n))
    d = pd.DataFrame({"x": x, "y": 2 + 0.8 * x + rng.normal(size=n), "z": np.cumsum(rng.normal(size=n))})
    h = ek.hata_duzeltme("z ~ x", d, gecikme=0)
    assert not h.esbutunlesme.esbutunlesik
    assert "yorumlanamaz" in str(h.rapor())


def test_zaman_boslugu_ve_eksik_gozlem():
    rng = np.random.default_rng(0)
    y = pd.Series(rng.normal(size=40), name="a")
    x = pd.Series(rng.normal(size=40), name="b")
    y.iloc[0] = np.nan
    ek.granger(y, x, gecikme=1)
    y.iloc[10] = np.nan
    with pytest.raises(ek.EkonometriHatasi):
        ek.granger(y, x, gecikme=1)
    d = pd.DataFrame({"y": rng.normal(size=30), "x": rng.normal(size=30)}, index=[*range(15), *range(16, 31)])
    with pytest.raises(ek.EkonometriHatasi):
        ek.hata_duzeltme("y ~ x", d, gecikme=0)


def test_hatalar():
    v = ek.veri.yukle("ihracat")
    with pytest.raises(ek.EkonometriHatasi):
        ek.adf(v["reel_kur"], "kuadratik")
    with pytest.raises(ek.EkonometriHatasi):
        ek.adf(pd.Series([1.0] * 50))
    with pytest.raises(ek.EkonometriHatasi):
        ek.adf(pd.Series(range(5), dtype=float))
    with pytest.raises(ek.EkonometriHatasi):
        ek.kpss(v["reel_kur"], "yok")
    with pytest.raises(ek.EkonometriHatasi):
        ek.engle_granger("np.log(ihracat) ~ np.log(reel_kur) + C(ceyrek)", v)
    with pytest.raises(ek.EkonometriHatasi):
        ek.duraganlik(v, ["dunya_buyumesi"], log=True)


def test_var_granger_ekk_f_testi():
    rng = np.random.default_rng(5)
    n, p = 120, 2
    e = rng.normal(size=(n, 3))
    Y = np.zeros((n, 3))
    for t in range(1, n):
        Y[t, 0] = 0.2 * Y[t - 1, 0] + 0.7 * Y[t - 1, 1] + e[t, 0]
        Y[t, 1] = 0.3 * Y[t - 1, 1] + e[t, 1]
        Y[t, 2] = 0.25 * Y[t - 1, 2] + 0.1 * Y[t - 1, 0] + e[t, 2]
    veri = pd.DataFrame(Y, columns=["a", "b", "c"])
    sonuc = ek.var_granger(veri, ["a", "b", "c"], gecikme=p)
    yon = next(y for y in sonuc.yonler if y.neden == "b" and y.etkilenen == "a")
    t = np.arange(p, n)
    Z = np.column_stack(
        [
            np.ones(len(t)),
            Y[t - 1, 0],
            Y[t - 2, 0],
            Y[t - 1, 1],
            Y[t - 2, 1],
            Y[t - 1, 2],
            Y[t - 2, 2],
        ]
    )
    R = np.zeros((p, Z.shape[1]))
    R[0, 3] = 1
    R[1, 4] = 1
    f = sm.OLS(Y[t, 0], Z).fit().f_test(R)
    assert np.isclose(yon.F, float(np.squeeze(f.fvalue)), rtol=1e-8)
    assert np.isclose(yon.p_degeri, float(np.squeeze(f.pvalue)), rtol=1e-8)
    assert yon.nedensel
    ters = next(y for y in sonuc.yonler if y.neden == "a" and y.etkilenen == "b")
    assert not ters.nedensel
    assert "does not Granger-cause" in sonuc.rapor(dil="en").markdown()
    with pytest.raises(ek.EkonometriHatasi):
        ek.var_granger(veri, ["a", "b"], gecikme=1)
