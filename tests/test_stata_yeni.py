"""Stata arayüzünün yeni komutları: seçim / sayım / tobit, margins at(), estat, grafikler, esttab using, zaman serisi, panel."""

import numpy as np
import pandas as pd
import pytest

import ekonometri as ek


@pytest.fixture
def s():
    return ek.Stata(yazdir=False)


@pytest.fixture(scope="module")
def c8():
    rng = np.random.default_rng(21)
    n = 600
    d = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    d["o"] = np.digitize(0.8 * d.x - 0.5 * d.z + rng.logistic(size=n), [-1, 0.5, 2]) + 1
    d["m"] = rng.choice([1, 2, 3], n, p=[0.5, 0.3, 0.2])
    d["c"] = rng.poisson(np.exp(0.3 + 0.5 * d.x) * rng.gamma(2, 0.5, n))
    d["t"] = np.maximum(1 + 2 * d.x - d.z + rng.normal(scale=1.5, size=n), 0)
    d["b"] = (d.x + rng.logistic(size=n) > 0).astype(int)
    d["g"] = rng.integers(1, 21, n)
    return d


def test_secim_modelleri(s, c8):
    s.veri = c8.copy()
    m = s("ologit o x z")
    assert np.allclose(m.katsayilar.values, ek.ologit("o ~ x + z", c8).katsayilar.values)
    m = s("mlogit m x z, baseoutcome(3)")
    assert str(m.temel) in ("3", "3.0")
    assert s("mlogit m x z").temel in (1, 1.0)  # en sık kategori
    m = s("poisson c x z, vce(robust) irr")
    assert m.kovaryans_turu == "saglam"
    m = s("nbreg c x, vce(cluster g)")
    assert m.kovaryans_turu == "kume"
    t = s("tobit t x z, ll(0)")
    assert np.allclose(t.katsayilar.values, ek.tobit("t ~ x + z", c8, alt=0).katsayilar.values)
    t2 = s("tobit t x z, ll")
    assert np.allclose(t2.katsayilar.values, t.katsayilar.values)
    ame = s("margins, dydx(*)")
    assert {"x", "z"} <= set(ame["değişken"])
    with pytest.raises(ek.EkonometriHatasi, match="sansür"):
        s("tobit t x z")


def test_margins_at_ve_grafik(s, tmp_path):
    s("sysuse ucret")
    s("regress ucret egitim tecrube i.cinsiyet")
    m = s("margins, at(egitim=(8(4)16))")
    assert list(m.tablo["x:egitim"]) == [8, 12, 16]
    dogrudan = ek.marjlar(s.son, at={"egitim": [8, 12, 16]})
    assert np.allclose(m.tablo["Marj"], dogrudan.tablo["Marj"]) and np.allclose(m.tablo["Std. hata"], dogrudan.tablo["Std. hata"])
    m = s("margins cinsiyet, at(egitim=(8 16))")
    assert len(m.tablo) == 4
    m = s("margins, dydx(egitim) at(tecrube=(5 20))")
    assert len(m.tablo) == 2 and np.allclose(m.tablo["Marj"], s.son.katsayilar["egitim"])  # doğrusal modelde sabit etki
    s("marginsplot")
    s(f"graph export {tmp_path / 'm.png'}")
    assert (tmp_path / "m.png").stat().st_size > 10_000
    with pytest.raises(ek.EkonometriHatasi, match="replace"):
        s(f"graph export {tmp_path / 'm.png'}")
    s(f"graph export {tmp_path / 'm.png'}, replace")
    for k in (
        "rvfplot",
        "qnorm",
        "qnorm ucret",
        "histogram ucret, normal",
        "scatter ucret egitim",
        "twoway (scatter ucret egitim) (lfit ucret egitim)",
    ):
        assert s(k) is s._son_sekil
    s("estimates store m1")
    s("regress ucret egitim tecrube")
    s("estimates store m2")
    s("coefplot m1 m2")
    for uz in ("docx", "tex", "xlsx"):
        s(f"esttab m1 m2 using {tmp_path / ('t.' + uz)}")
        assert (tmp_path / f"t.{uz}").exists()
    with pytest.raises(ek.EkonometriHatasi, match="replace"):
        s(f"esttab m1 m2 using {tmp_path / 't.docx'}")


def test_logit_sonrasi(s, c8):
    s.veri = c8.copy()
    s("logit b x z")
    t = s("estat classification")
    assert t.values.sum() == len(c8)
    hl = s("estat gof, group(10)")
    assert hl["sd"] == 8 and 0 <= hl["p"] <= 1
    s("lroc")
    ame = s("margins, dydx(*)")
    assert "AME" in ame.columns
    m = s("margins, at(x=(-1 0 1))")
    assert m.tablo["Marj"].is_monotonic_increasing and "z" in m.tablo.columns
    with pytest.raises(ek.EkonometriHatasi, match="group"):
        s("estat gof")


def test_zaman_serisi_komutlari(s):
    rng = np.random.default_rng(5)
    T = 200
    e = rng.normal(size=(T, 2))
    y = np.zeros((T, 2))
    for t in range(1, T):
        y[t] = [0.5 * y[t - 1, 0] + 0.2 * y[t - 1, 1], 0.3 * y[t - 1, 1]] + e[t]
    x = np.cumsum(rng.normal(size=T))
    s.veri = pd.DataFrame({"t": range(T), "a": y[:, 0], "b": y[:, 1], "x": x, "w": 1 + 0.8 * x + rng.normal(size=T)})
    s("tsset t")
    v = s("var a b, lags(1/2)")
    assert v.gecikme == 2 and np.allclose(v.katsayilar.values, ek.var(s.veri, ["a", "b"], 2).katsayilar.values)
    t = s("varsoc a b, maxlag(4)")
    assert t.attrs["secim"]["SBIC"] == 1
    assert s("varstable").attrs["kararli"]
    s("vargranger")
    s("irf graph oirf, step(10)")
    assert s("irf table oirf").shape == (9, 4)
    m = s("ardl w x, lags(1)")
    assert m is s.son
    s("estat ectest")
    j = s("vecrank w x, lags(2)")
    assert j.gecikme == 1
    s("tsline a b")


def test_panel_komutlari(s):
    rng = np.random.default_rng(4)
    N, T = 60, 8
    rows = []
    for i in range(N):
        a = rng.normal()
        yy, xx = 0.0, rng.normal()
        for t in range(T):
            xx = 0.5 * xx + rng.normal()
            yy = 0.5 * yy + 0.8 * xx + a + rng.normal()
            rows.append((i, t, yy, xx))
    s.veri = pd.DataFrame(rows, columns=["id", "yil", "y", "x"])
    s("xtset id yil")
    m = s("xtabond y x, vce(robust)")
    assert m.kovaryans_turu == "saglam" and 0.3 < m.katsayilar.iloc[0] < 0.7
    assert set(s("estat abond")) == {1, 2}
    s("estat sargan")
    for k in (
        "xtunitroot llc x",
        "xtunitroot ips y, lags(1)",
        "xtunitroot fisher x, dfuller lags(0)",
        "xtunitroot ips x, lags(aic 2) demean",
    ):
        r = s(k)
        assert 0 <= r.p <= 1
    assert s("xtunitroot llc x").p < 0.05  # x durağan AR(0,5)
    with pytest.raises(ek.EkonometriHatasi, match="dfuller"):
        s("xtunitroot fisher x")


def test_sayi_listesi():
    from ekonometri.stata import _sayi_listesi

    assert _sayi_listesi("8 12 16") == [8, 12, 16]
    assert _sayi_listesi("0(5)20") == [0, 5, 10, 15, 20]
    assert _sayi_listesi("1/4") == [1, 2, 3, 4]
    assert _sayi_listesi("0.5(0.25)1") == [0.5, 0.75, 1]


# --------------------------------------------------------------------------- hata düzeltmeleri için regresyon testleri
def test_eksen_bicimi():
    from ekonometri.grafik import _virgul

    assert _virgul(1.4e-05) == "1,4×10⁻⁵" and _virgul(2.7e-17) == "0" and _virgul(400000) == "400.000" and _virgul(0.25) == "0,25"


def test_tarih_kosulu_ve_desteklenmeyen_sonrasi(s):
    s("sysuse ihracat")  # donem: çeyreklik Period sütunu
    assert s("count if donem >= tq(2010q1)") == s("count if donem >= 2010q1") == 60
    assert s("var ihracat reel_kur if donem >= 2010q1, lags(1)").T == 59
    for k in ("yorum", "rapor v.docx", "predict w"):
        with pytest.raises(ek.EkonometriHatasi):
            s(k)


def test_faktor_marginsplot(s):
    s("sysuse ucret")
    s("regress ucret egitim i.bolge")
    s("margins bolge")
    assert s("marginsplot") is s._son_sekil


def test_ml_predict_ve_maruziyet(s, c8):
    d = c8.copy()
    d["e"] = np.random.default_rng(2).uniform(0.5, 3, len(d))
    s.veri = d
    s("ologit o x z")
    p = s("predict p1 p2 p3 p4")
    assert np.allclose(p.sum(axis=1), 1)
    s("mlogit m x z")
    assert list(s("predict m1 m2 m3").columns) == ["Pr(m=1)", "Pr(m=2)", "Pr(m=3)"]  # sonuç sırası, temel önde değil
    m = s("poisson c x, exposure(e)")
    n = m.tahminler()["n"]
    assert np.allclose(n, d.e * np.exp(m.katsayilar["Intercept"] + m.katsayilar["x"] * d.x))
    # AME gözlem başına maruziyetle: ortalama(β·n_i)
    assert np.isclose(m.marjinal_etkiler()["AME"].iloc[0], (m.katsayilar["x"] * n).mean())


def test_predict_outcome_tam_eslesir(s):
    rng = np.random.default_rng(0)
    n = 300
    x = rng.normal(size=n)
    xb = np.column_stack([np.zeros(n), 0.8 * x, -0.4 * x])
    y = np.array([1, 11, 12])[np.argmax(xb + rng.gumbel(size=(n, 3)), axis=1)]
    s.veri = pd.DataFrame({"y": y, "x": x})
    s("mlogit y x")
    beklenen = s.son.tahminler()
    for kod in (1, 11, 12):
        p = s(f"predict p{kod}, outcome({kod})")
        assert list(p.columns) == [f"Pr(y={kod})"]
        assert np.allclose(p.iloc[:, 0], beklenen[f"Pr(y={kod})"])


def test_nbreg_asiri_yayilim_yokken():
    rng = np.random.default_rng(1)
    d = pd.DataFrame({"x": rng.normal(size=800)})
    d["c"] = rng.poisson(np.exp(0.3 + 0.4 * d.x))  # saf Poisson: α = 0 sınırında
    nb, po = ek.nbreg("c ~ x", d), ek.poisson("c ~ x", d)
    assert np.all(np.isfinite(nb.katsayilar)) and np.allclose(nb.katsayilar, po.katsayilar, atol=1e-4)
    assert nb.ek_parametreler.loc["α", "Değer"] < 1e-3 and nb.ek_bilgi["alfa_lr"][1] > 0.4
