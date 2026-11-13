"""Model karşılaştırma tablosu, korelasyon matrisi, tanımlayıcı istatistikler ve etkileşim yorumu."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import ekonometri as ek


@pytest.fixture(scope="module")
def modeller():
    v = ek.veri.yukle("ucret")
    return (
        ek.ekk("np.log(ucret) ~ egitim", v),
        ek.ekk("np.log(ucret) ~ egitim + tecrube + I(tecrube**2)", v),
        ek.ekk("np.log(ucret) ~ egitim + tecrube + I(tecrube**2) + cinsiyet", v),
    )


def _tr(x, b):
    return f"{x:.{b}f}".replace(".", ",")


def test_karsilastirma_tablosu(modeller):
    k = ek.karsilastir(*modeller)
    t = k.tablo()
    assert list(t.columns) == ["Değişken", "Model 1", "Model 2", "Model 3"]
    egitim = t[t["Değişken"] == "egitim"].iloc[0]
    for i, m in enumerate(modeller, 1):
        assert egitim[f"Model {i}"] == _tr(m.katsayilar["egitim"], 4) + "***"
    i = t.index[t["Değişken"] == "egitim"][0]
    assert t.loc[i + 1, "Model 2"] == f"({_tr(modeller[1].standart_hatalar['egitim'], 4)})"
    kadin = t[t["Değişken"] == "cinsiyet: Kadın"].iloc[0]
    assert kadin["Model 1"] == "" and kadin["Model 3"].startswith("-0,")
    # sabit, katsayıların sonunda; ardından uyum istatistikleri
    adlar = list(t["Değişken"])
    assert adlar.index("Sabit") > adlar.index("cinsiyet: Kadın") and adlar.index("Gözlem sayısı") > adlar.index("Sabit")
    r2 = t[t["Değişken"] == "R²"].iloc[0]
    assert r2["Model 3"] == _tr(modeller[2].r2, 3)
    assert "Model 1" in str(k) and "<table" in k._repr_html_()


def test_karsilastirma_raporu(modeller, tmp_path):
    r = ek.karsilastir(*modeller, adlar=["Temel", "Tecrübeli", "Tam"]).rapor(
        tmp_path / "m.docx", etiketler={"egitim": "Eğitim süresi"}, tablo_no=4
    )
    metin = str(r)
    assert (tmp_path / "m.docx").stat().st_size > 5000
    assert "Tablo 4 ile" in metin and "Tam ile elde edilmiştir" in metin
    assert "aynı işaretli ve %5 düzeyinde istatistiksel olarak anlamlıdır: Eğitim süresi." in metin
    assert r.tablolar[0].sutunlar == ["Değişken", "Temel", "Tecrübeli", "Tam"]


def test_karsilastirma_hatalari(modeller):
    with pytest.raises(ek.EkonometriHatasi):
        ek.karsilastir()
    with pytest.raises(ek.EkonometriHatasi):
        ek.karsilastir(modeller[0], "model değil")
    with pytest.raises(ek.EkonometriHatasi):
        ek.karsilastir(*modeller, adlar=["A"])
    assert len(ek.karsilastir(list(modeller)).modeller) == 3


def test_korelasyon_scipy_ile_ayni():
    v = ek.veri.yukle("konut")
    s = ["fiyat", "metrekare", "oda", "bina_yasi"]
    for yontem, f in (("pearson", stats.pearsonr), ("spearman", stats.spearmanr)):
        k = ek.korelasyon(v, s, yontem=yontem)
        for a in s:
            for b in s:
                if a != b:
                    r, p = f(v[a], v[b])
                    assert np.isclose(k.r.loc[a, b], r, rtol=1e-10)
                    assert np.isclose(k.p.loc[a, b], p, rtol=1e-6, atol=1e-300)
    t = ek.korelasyon(v, s).tablo()
    assert t.iloc[0, 1] == "1" and t.iloc[0, 2] == "" and t.iloc[1, 1].endswith("***")
    metin = str(ek.korelasyon(v, s).rapor(etiketler={"oda": "Oda sayısı"}))
    assert "En yüksek" not in metin and "Mutlak değerce en yüksek korelasyon" in metin and "| 3. Oda sayısı |" in metin


def test_korelasyon_hatalari():
    v = ek.veri.yukle("ucret")
    with pytest.raises(ek.EkonometriHatasi):
        ek.korelasyon(v, ["ucret", "cinsiyet"])
    with pytest.raises(ek.EkonometriHatasi):
        ek.korelasyon(v, ["ucret", "yok"])
    with pytest.raises(ek.EkonometriHatasi):
        ek.korelasyon(v, yontem="kendall")
    assert list(ek.korelasyon(v).r.columns) == ["ucret", "egitim", "tecrube", "yas"]


def test_tanimlayici_stata_formulleri():
    v = ek.veri.yukle("ucret")
    t = ek.tanimlayici(v, ["ucret", "egitim"], detay=True)
    x = v["ucret"].values.astype(float)
    assert np.isclose(t.loc["ucret", "Ortalama"], x.mean()) and t.loc["egitim", "N"] == 1500
    assert np.isclose(t.loc["ucret", "Std. sapma"], x.std(ddof=1))
    # Stata summarize, detail: çarpıklık ve basıklık yanlı (n paydalı) momentlerle
    assert np.isclose(t.loc["ucret", "Çarpıklık"], stats.skew(x, bias=True))
    assert np.isclose(t.loc["ucret", "Basıklık"], stats.kurtosis(x, bias=True, fisher=False))
    # Stata yüzdelik tanımı: n·p/100 tam sayıysa iki gözlemin ortalaması, değilse yukarı yuvarla
    xs = np.sort(x)
    assert t.loc["ucret", "Medyan"] == (xs[749] + xs[750]) / 2  # 1500·0,50 = 750
    assert t.loc["ucret", "%1"] == (xs[14] + xs[15]) / 2  # 1500·0,01 = 15
    tek = pd.DataFrame({"a": [1.0, 2, 3, 4, 5, 6, 7]})
    assert ek.tanimlayici(tek, detay=True).loc["a", "%25"] == 2  # 7·0,25 = 1,75 → 2. gözlem
    assert ek.tanimlayici(tek, detay=True).loc["a", "Medyan"] == 4


def test_pwcorr_cift_bazinda():
    v = ek.veri.yukle("konut")[["fiyat", "metrekare", "oda"]].astype(float).copy()
    v.loc[:99, "oda"] = np.nan
    liste = ek.korelasyon(v)
    cift = ek.korelasyon(v, eksik="cift")
    assert liste.n == len(v) - 100
    assert np.isclose(cift.r.loc["fiyat", "metrekare"], stats.pearsonr(v["fiyat"], v["metrekare"])[0])
    assert np.isclose(cift.p.loc["fiyat", "metrekare"], stats.pearsonr(v["fiyat"], v["metrekare"])[1])
    assert cift.n_matris.loc["fiyat", "metrekare"] == len(v) and cift.n_matris.loc["fiyat", "oda"] == len(v) - 100


def test_etkilesim_terimi_yorumu():
    v = ek.veri.yukle("ucret")
    m = ek.ekk("np.log(ucret) ~ egitim * cinsiyet + egitim:tecrube + tecrube", v)
    turler = {t.sutun: t.tur for t in m._terimler}
    assert turler["egitim:cinsiyet[T.Kadın]"] == "etkilesim" and turler["egitim:tecrube"] == "etkilesim"
    y = str(m.yorumla())
    assert "'tecrube' değişkeninin düzeyine bağlı" in y and "ana etkiler" in y
    r = str(m.rapor(yontem=False))
    assert "egitim × cinsiyet: Kadın" in r and "ana etki katsayıları" in r
    # anlamlı kukla × sürekli etkileşimi: grup eğimi farkı doğru yönde anlatılır
    rng = np.random.default_rng(5)
    import pandas as pd

    d = pd.DataFrame({"x": rng.normal(size=400), "g": rng.choice(["A", "B"], size=400)})
    d["y"] = 1 + d["x"] + 2 * d["x"] * (d["g"] == "B") + rng.normal(size=400)
    metin = str(ek.ekk("y ~ x * g", d).yorumla())
    assert "'g' değişkeninin 'B' grubunda referans gruba ('A') kıyasla daha yüksek" in metin
