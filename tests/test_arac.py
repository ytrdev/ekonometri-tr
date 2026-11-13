"""Araç değişken (2SLS): linearmodels ile birebir karşılaştırma, tanı testleri ve rapor."""

import numpy as np
import pandas as pd
import pytest
from linearmodels.iv import IV2SLS

import ekonometri as ek

F = "np.log(ucret) ~ tecrube + I(tecrube**2) + cinsiyet + yerlesim + [egitim ~ uzaklik + kardes]"
ESLE = {
    "Intercept": "const",
    "tecrube": "tecrube",
    "I(tecrube ** 2)": "tec2",
    "cinsiyet[T.Kadın]": "kadin",
    "yerlesim[T.Kent]": "kent",
    "egitim": "egitim",
}


@pytest.fixture(scope="module")
def veri():
    return ek.veri.yukle("getiri")


def _lm(v, kov, debiased=False, araclar=("uzaklik", "kardes")):
    d = v.copy()
    d["lw"], d["tec2"] = np.log(d.ucret), d.tecrube**2
    d["kadin"], d["kent"], d["const"] = (d.cinsiyet == "Kadın").astype(float), (d.yerlesim == "Kent").astype(float), 1.0
    return IV2SLS(d.lw, d[["const", "tecrube", "tec2", "kadin", "kent"]], d.egitim, d[list(araclar)]).fit(cov_type=kov, debiased=debiased)


@pytest.mark.parametrize("kov,lmkov", [("klasik", "unadjusted"), ("saglam", "robust")])
@pytest.mark.parametrize("kucuk", [False, True])
def test_linearmodels_ile_ayni(veri, kov, lmkov, kucuk):
    """kucuk_orneklem=False: Stata ivregress varsayılanı (z, e'e/N); True: Stata 'small' (t, e'e/(N−k))."""
    m = ek.iv2sls(F, veri, kovaryans=kov, kucuk_orneklem=kucuk)
    r = _lm(veri, lmkov, debiased=kucuk)
    b, s = m.katsayilar.rename(index=ESLE), m.standart_hatalar.rename(index=ESLE)
    assert np.allclose(b.values, r.params[b.index].values, rtol=1e-8)
    assert np.allclose(s.values, r.std_errors[b.index].values, rtol=1e-8)
    assert np.allclose(m.p_degerleri.rename(index=ESLE)[b.index].values, r.pvalues[b.index].values, rtol=1e-6, atol=1e-300)
    assert np.isclose(m.r2, r.rsquared)
    assert m.ist_adi == ("t" if kucuk else "z")
    d = r.first_stage.diagnostics
    assert np.isclose(m.birinci_asama["egitim"]["kismi_r2"], d["partial.rsquared"].iloc[0])
    if kov == "klasik" and kucuk:  # Stata estat firststage F'yi her zaman küçük örneklem formülüyle verir
        assert np.isclose(m.birinci_asama["egitim"]["F"], d["f.stat"].iloc[0])


def test_tani_testleri(veri):
    m = ek.iv2sls(F, veri)
    r = _lm(veri, "unadjusted")
    birinci, durbin, wh, sg, bas = m.tanilar()
    assert np.isclose(durbin.istatistik, r.durbin().stat) and np.isclose(durbin.p_degeri, r.durbin().pval)
    assert np.isclose(wh.istatistik, r.wu_hausman().stat) and np.isclose(wh.p_degeri, r.wu_hausman().pval)
    assert np.isclose(sg.istatistik, r.sargan.stat) and np.isclose(sg.p_degeri, r.sargan.pval)
    assert np.isclose(bas.istatistik, r.basmann.stat) and np.isclose(bas.p_degeri, r.basmann.pval)
    assert not birinci.sorun_var and wh.sorun_var and durbin.sorun_var and not sg.sorun_var
    # tam tanımlı modelde Sargan / Basmann yok
    assert len(ek.iv2sls("np.log(ucret) ~ tecrube + [egitim ~ uzaklik]", veri).tanilar()) == 3


def test_kumelenmis(veri):
    v = veri.assign(grup=np.arange(len(veri)) % 40)
    m = ek.iv2sls(F, v, kume="grup", kucuk_orneklem=True)
    assert m.kovaryans_turu == "kume" and "40 küme" in m.kovaryans_aciklama
    assert m.ekk.kovaryans_turu == "kume"


def test_2sls_gercek_degeri_bulur_ekk_yanli(veri):
    m = ek.iv2sls(F, veri)
    assert abs(m.katsayilar["egitim"] - 0.075) < 0.02  # gerçek getiri 0,075
    assert m.ekk.katsayilar["egitim"] > m.katsayilar["egitim"] + 0.015  # yetenek yanlılığı


def test_zayif_arac_uyarisi():
    rng = np.random.default_rng(4)
    n = 500
    u = rng.normal(size=n)
    z = rng.normal(size=n)
    x = 0.03 * z + u + rng.normal(size=n)
    d = pd.DataFrame({"y": 1 + x + u + rng.normal(size=n), "x": x, "z": z})
    t = ek.iv2sls("y ~ [x ~ z]", d).tanilar()[0]
    assert t.sorun_var and "zayıf" in t.karar


def test_yorum_ve_rapor(veri, tmp_path):
    m = ek.iv2sls(F, veri, birimler={"egitim": "yıl"})
    y = str(m.yorumla())
    assert "EKK tahmini 'egitim' için" in y and "olduğundan büyük" in y
    r = m.rapor(tmp_path / "iv.docx", etiketler={"egitim": "Eğitim süresi", "uzaklik": "Üniversiteye uzaklık"})
    metin = str(r)
    for parca in (
        "## Birinci Aşama",
        "## Tanı Testleri",
        "Wu-Hausman testi (F(1, 1993)",
        "Üniversiteye uzaklık",
        "Sargan, J. D. (1958)",
        "yukarı yönlü bir yanlılığa",
    ):
        assert parca in metin, parca
    assert "Eğitim süresi değişkeni 1 yıl arttığında" in metin
    assert [t.no for t in r.tablolar] == [1, 2, 3, 4, 5]
    assert (tmp_path / "iv.docx").stat().st_size > 5000


def test_hatalar(veri):
    with pytest.raises(ek.EkonometriHatasi, match="köşeli parantez"):
        ek.iv2sls("np.log(ucret) ~ egitim + tecrube", veri)
    with pytest.raises(ek.EkonometriHatasi, match="eksik tanımlı"):
        ek.iv2sls("np.log(ucret) ~ [egitim + tecrube ~ uzaklik]", veri)
    with pytest.raises(ek.EkonometriHatasi):
        ek.iv2sls("np.log(ucret) ~ uzaklik + [egitim ~ uzaklik + kardes]", veri)
    with pytest.raises(ek.EkonometriHatasi):
        ek.iv2sls(F, veri, kovaryans="HAC")
    with pytest.raises(ek.EkonometriHatasi):
        ek.iv2sls(F, veri, kovaryans="kume")
