"""Panel veri: linearmodels ile birebir karşılaştırma, model seçim testleri ve rapor."""

import numpy as np
import pandas as pd
import pytest
from linearmodels import PanelOLS, PooledOLS, RandomEffects

import ekonometri as ek
from ekonometri.panel import bp_lm_testi, f_testi

F = "np.log(satis) ~ np.log(calisan) + np.log(sermaye) + arge + ihracatci"
LM_F = "ls ~ 1 + ll + lk + arge + ihracatci"
ESLE = {
    "Intercept": "Intercept",
    "np.log(calisan)": "ll",
    "np.log(sermaye)": "lk",
    "arge": "arge",
    "ihracatci[T.Evet]": "ihracatci[T.Evet]",
}
KW = {"birim": "firma", "zaman": "yil"}


def _lm_veri(v):
    d = v.set_index(["firma", "yil"]).copy()
    d["ls"], d["ll"], d["lk"] = np.log(d.satis), np.log(d.calisan), np.log(d.sermaye)
    return d


def _ayni(seri, lm, rtol=1e-8):
    s = seri.rename(index=ESLE)
    assert np.allclose(s.values, lm[s.index].values, rtol=rtol)


@pytest.fixture(scope="module", params=["dengeli", "dengesiz"])
def veri(request):
    v = ek.veri.yukle("firma")
    if request.param == "dengesiz":
        v = v.sample(frac=0.85, random_state=1).sort_values(["firma", "yil"]).reset_index(drop=True)
    return v


def test_havuz(veri):
    m = ek.panel(F, veri, yontem="havuz", **KW)
    r = PooledOLS.from_formula(LM_F, _lm_veri(veri)).fit()
    _ayni(m.katsayilar, r.params)
    _ayni(m.standart_hatalar, r.std_errors)


def test_sabit_etkiler(veri):
    m = ek.panel(F, veri, yontem="sabit", **KW)
    r = PanelOLS.from_formula("ls ~ 1 + ll + lk + arge + EntityEffects", _lm_veri(veri)).fit()
    assert m.cikarilanlar == ["ihracatci[T.Evet]"]
    _ayni(m.katsayilar, r.params)
    _ayni(m.standart_hatalar, r.std_errors)
    assert m.sd == r.df_resid
    assert np.isclose(m.r2_ic, r.rsquared_within)
    # Stata xtreg: birimler arası ve genel R² korelasyon kareleridir
    d = _lm_veri(veri)
    tahmin = pd.Series(m.X.values @ m.katsayilar.values, index=d.index)
    assert np.isclose(m.r2_genel, np.corrcoef(d["ls"], tahmin)[0, 1] ** 2)
    ort = pd.DataFrame({"y": d["ls"], "xb": tahmin}).groupby(level=0).mean()
    assert np.isclose(m.r2_arasi, np.corrcoef(ort["y"], ort["xb"])[0, 1] ** 2)
    assert np.isclose(m.f_istatistigi, r.f_statistic.stat)
    h = ek.panel(F, veri, yontem="havuz", **KW)
    assert np.isclose(f_testi(h, m).istatistik, r.f_pooled.stat)


def test_rassal_etkiler(veri):
    m = ek.panel(F, veri, yontem="rassal", **KW)
    r = RandomEffects.from_formula(LM_F, _lm_veri(veri)).fit()
    _ayni(m.katsayilar, r.params)
    _ayni(m.standart_hatalar, r.std_errors)
    assert np.isclose(m.sigma2_u, r.variance_decomposition["Effects"]) and np.isclose(m.sigma2_e, r.variance_decomposition["Residual"])


def test_kumelenmis_standart_hata(veri):
    m = ek.panel(F, veri, yontem="sabit", kovaryans="kume", **KW)
    r = PanelOLS.from_formula("ls ~ 1 + ll + lk + arge + EntityEffects", _lm_veri(veri)).fit(
        cov_type="clustered", cluster_entity=True, debiased=False
    )
    G = m.birim_sayisi
    c = G / (G - 1) * (m.n - 1) / (m.n - m.k)  # Stata küçük örneklem düzeltmesi
    _ayni(m.standart_hatalar / np.sqrt(c), r.std_errors)


def test_iki_yonlu_sabit_etkiler():
    v = ek.veri.yukle("firma")
    m = ek.panel(F, v, yontem="sabit", zaman_etkisi=True, **KW)
    r = PanelOLS.from_formula("ls ~ 1 + ll + lk + arge + EntityEffects + TimeEffects", _lm_veri(v)).fit()
    ana = ["np.log(calisan)", "np.log(sermaye)", "arge"]
    _ayni(m.katsayilar[ana], r.params)
    _ayni(m.standart_hatalar[ana], r.std_errors)
    # iki yönlü model gerçek parametreleri yakalar (0,62 / 0,31 / 0,045)
    assert abs(m.katsayilar["np.log(calisan)"] - 0.62) < 0.05 and abs(m.katsayilar["np.log(sermaye)"] - 0.31) < 0.05
    assert "yil[2016]" not in [t.sutun for t in m._terimler]
    assert "Zaman etkileri" in m._uyum() and m._uyum()["Zaman etkileri"] == "Var"


def test_bp_lm_dengeli_formul():
    v = ek.veri.yukle("firma")
    h = ek.panel(F, v, yontem="havuz", **KW)
    e = h.kalintilar.values
    N, T = 150, 10
    toplam = e.reshape(N, T).sum(axis=1)
    beklenen = N * T / (2 * (T - 1)) * (np.sum(toplam**2) / np.sum(e**2) - 1) ** 2
    assert np.isclose(bp_lm_testi(h).istatistik, beklenen)


def test_hausman_elle_hesapla():
    v = ek.veri.yukle("firma")
    s = ek.panel(F, v, yontem="sabit", **KW)
    r = ek.panel(F, v, yontem="rassal", **KW)
    h = ek.hausman(s, r)
    o = ["np.log(calisan)", "np.log(sermaye)", "arge"]
    d = (s.katsayilar[o] - r.katsayilar[o]).values
    Vs, Vr = s.kovaryans_matrisi.loc[o, o].values, r.kovaryans_matrisi.loc[o, o].values
    if h.detay["sigmamore"]:
        Vs = Vs * (r.sse / r.sd) / (s.sse / s.sd)
    assert np.isclose(h.istatistik, d @ np.linalg.inv(Vs - Vr) @ d)
    assert h.serbestlik == (3,) and h.secilen == "sabit"


def test_panel_analizi_ve_rapor(tmp_path):
    v = ek.veri.yukle("firma")
    a = ek.panel_analizi(F, v, **KW)
    assert a.onerilen == "sabit" and a.nihai.kovaryans_turu == "kume"
    assert [t.secilen for t in a.testler] == ["sabit", "rassal", "sabit"]
    assert list(a.test_tablosu()["Tercih"]) == ["Sabit etkiler", "Rassal etkiler", "Sabit etkiler"]
    r = a.rapor(tmp_path / "panel.docx", etiketler={"calisan": "Çalışan sayısı"})
    metin = str(r)
    for parca in (
        "## Model Seçimi",
        "Hausman testi (χ²(3)",
        "'ihracatci' değişkeni birim etkileri",
        "Çalışan sayısı değişkeni %1 arttığında",
        "Baltagi, B. H.",
    ):
        assert parca in metin
    assert "Bağımlı değişken: ln(satis)." in metin
    assert (tmp_path / "panel.docx").stat().st_size > 5000
    assert "Önerilen model: Sabit etkiler" in str(a)


def test_rassal_etkiler_secilir_iliskisiz_etkide():
    rng = np.random.default_rng(8)
    N, T = 120, 6
    a = np.repeat(rng.normal(0, 1, N), T)
    x = rng.normal(size=N * T)
    d = pd.DataFrame({"i": np.repeat(np.arange(N), T), "t": np.tile(np.arange(T), N), "x": x})
    d["y"] = 1 + 0.5 * d["x"] + a + rng.normal(0, 1, N * T)
    an = ek.panel_analizi("y ~ x", d, birim="i", zaman="t")
    assert an.testler[2].secilen == "rassal" and an.onerilen == "rassal"


def test_hatalar():
    v = ek.veri.yukle("firma")
    with pytest.raises(ek.EkonometriHatasi, match="birden fazla"):
        ek.panel(F, pd.concat([v, v.head(2)]), **KW)
    with pytest.raises(ek.EkonometriHatasi):
        ek.panel(F, v, birim="sirket", zaman="yil")
    with pytest.raises(ek.EkonometriHatasi):
        ek.panel(F, v, yontem="gmm", **KW)
    with pytest.raises(ek.EkonometriHatasi):
        ek.panel(F, v, kovaryans="HC1", **KW)


def test_tek_model_raporu(tmp_path):
    v = ek.veri.yukle("firma")
    for yontem in ("havuz", "sabit", "rassal"):
        m = ek.panel(F, v, yontem=yontem, kovaryans="kume", zaman_etkisi=True, **KW)
        r = m.rapor(tmp_path / f"{yontem}.docx", etiketler={"calisan": "Çalışan sayısı"})
        metin = str(r)
        assert "## Yöntem" in metin and "Çalışan sayısı değişkeni %1 arttığında" in metin and "Baltagi" in metin
        assert "Zaman etkileri modele dahildir" in metin and "yil[" not in metin
        assert (tmp_path / f"{yontem}.docx").exists()
    assert "| z |" in str(ek.panel(F, v, yontem="rassal", **KW).rapor())
