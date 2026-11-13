"""lincom: R (lm / estimatr se_type="stata" / glm vcov ile c'b, √(c'Vc)) ile karşılaştırma ve tüm model türlerinde tutarlılık."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import ekonometri as ek

F = "np.log(ucret) ~ egitim + tecrube + I(tecrube**2) + cinsiyet"
IFADE = ["egitim + 2*tecrube", "egitim - tecrube", "_cons + cinsiyet[T.Kadın]", "tecrube + 2*40*I(tecrube ** 2)", "3*egitim - 0.5"]
# R: lm / lm_robust(se_type = "stata") / lm_robust(clusters = grup, se_type = "stata")
REF = {
    ("klasik", None): [0.0067547843, 0.0041494357, 0.0547807804, 0.0022161165, 0.0094306099],
    ("HC1", None): [0.0067511618, 0.0041782734, 0.0528336763, 0.0021493928, 0.0092019492],
    ("klasik", "grup"): [0.0051677894, 0.0038764496, 0.0467175529, 0.0018464451, 0.0090124654],
}
TAHMIN = [0.1637466711, 0.0393382551, 9.4797804958, -0.0154930457, -0.2575768187]


@pytest.fixture(scope="module")
def u():
    v = ek.veri.yukle("ucret")
    v["grup"] = np.arange(len(v)) % 40
    return v


@pytest.mark.parametrize("kov,kume", list(REF))
def test_ekk_r_ile_ayni(u, kov, kume):
    m = ek.ekk(F, u, kovaryans=kov, kume=kume)
    for e, t, se in zip(IFADE, TAHMIN, REF[(kov, kume)]):
        r = ek.lincom(m, e)
        assert np.isclose(r.tahmin, t, atol=1e-9) and np.isclose(r.std_hata, se, rtol=1e-8)
        assert r.ist_adi == "t" and r.sd == (39 if kume else m.sd)
    r = ek.lincom(ek.ekk(F, u, kume="grup"), "tecrube + 2*40*I(tecrube ** 2)")
    assert np.isclose(r.p, 2 * stats.t.sf(abs(r.istatistik), 39))


def test_logit_r_ile_ayni():
    s = ek.veri.yukle("sinav")
    r = ek.lincom(ek.logit("gecti ~ calisma_saati + uyku + burs", s), "calisma_saati - uyku")
    assert np.isclose(r.tahmin, -0.0922979189, atol=1e-8) and np.isclose(r.std_hata, 0.0954395223, rtol=1e-7)
    assert r.ist_adi == "z" and np.isclose(r.p, 0.3335026512, atol=1e-7)


def _tek_katsayi_tutarli(m, ad, ist):
    r = ek.lincom(m, ad)
    assert np.isclose(r.tahmin, m.katsayilar[ad]) and np.isclose(r.std_hata, m.standart_hatalar[ad], rtol=1e-10)
    assert r.ist_adi == ist


def test_tum_model_turleri(u):
    f = ek.xtset(ek.veri.yukle("firma"), "firma", "yil")
    pf = "np.log(satis) ~ np.log(calisan) + arge"
    _tek_katsayi_tutarli(ek.panel(pf, f, yontem="sabit", kovaryans="kume"), "arge", "t")
    _tek_katsayi_tutarli(ek.panel(pf, f, yontem="rassal"), "arge", "z")
    g = ek.veri.yukle("getiri")
    _tek_katsayi_tutarli(ek.iv2sls("np.log(ucret) ~ tecrube + [egitim ~ uzaklik]", g), "egitim", "z")
    _tek_katsayi_tutarli(ek.iv2sls("np.log(ucret) ~ tecrube + [egitim ~ uzaklik]", g, kucuk_orneklem=True), "egitim", "t")
    s = ek.veri.yukle("sinav")
    s["not_"] = pd.cut(s.puan, [0, 50, 65, 80, 101], labels=False) + 1
    _tek_katsayi_tutarli(ek.ologit("not_ ~ uyku + calisma_saati", s), "uyku", "z")
    _tek_katsayi_tutarli(ek.tobit("puan ~ uyku", s, ust=100), "uyku", "z")
    d = ek.xtabond("np.log(satis) ~ np.log(calisan)", f)
    _tek_katsayi_tutarli(d, d.katsayilar.index[0], "z")
    # sözlük, _b[], boşluksuz ad, bölme
    m = ek.ekk(F, u)
    a = ek.lincom(m, {"egitim": 1, "tecrube": -1})
    b = ek.lincom(m, "_b[egitim] - _b[tecrube]")
    c = ek.lincom(m, "egitim/2 + egitim/2 - tecrube")
    assert np.isclose(a.std_hata, b.std_hata) and np.isclose(b.tahmin, c.tahmin) and np.isclose(b.std_hata, c.std_hata)
    assert np.isclose(ek.lincom(m, "tecrube + 80*I(tecrube**2)").tahmin, TAHMIN[3])
    assert "güven aralığı" in str(a)


def test_hatalar(u):
    m = ek.ekk(F, u)
    with pytest.raises(ek.EkonometriHatasi, match="olmayan terim"):
        ek.lincom(m, "egitim + yas")
    with pytest.raises(ek.EkonometriHatasi, match="Doğrusal olmayan"):
        ek.lincom(m, "egitim*tecrube")
    with pytest.raises(ek.EkonometriHatasi, match="katsayı yok"):
        ek.lincom(m, "2 + 3")
    with pytest.raises(ek.EkonometriHatasi, match="kullanılamaz"):
        ek.lincom(ek.var(ek.veri.yukle("ihracat"), ["ihracat", "reel_kur"]), "x")
