"""Grafikler, tablo dışa aktarma (docx / tex / xlsx / csv), margins, ROC / Hosmer-Lemeshow ve VAR."""

import numpy as np
import pandas as pd
import pytest

import ekonometri as ek


@pytest.fixture(scope="module")
def ucret():
    return ek.veri.yukle("ucret")


@pytest.fixture(scope="module")
def model(ucret):
    return ek.ekk("np.log(ucret) ~ egitim * cinsiyet + tecrube + I(tecrube**2)", ucret)


def test_grafikler_ve_rapora_ekleme(model, ucret, tmp_path):
    sekiller = [
        ek.kalinti_grafigi(model),
        ek.normallik_grafigi(model),
        ek.katsayi_grafigi(model, ek.ekk("np.log(ucret) ~ egitim", ucret)),
        ek.zaman_grafigi(ek.veri.yukle("ihracat"), ["ihracat", "reel_kur"]),
        ek.histogram(ucret, "egitim"),
        ek.serpme(ucret, "ucret", "egitim"),
        ek.marjinal_grafik(ek.marjlar(model, at={"egitim": [8, 12, 16]}, grup="cinsiyet")),
        ek.roc_grafigi(ek.logit("gecti ~ calisma_saati + devam", ek.veri.yukle("sinav"))),
    ]
    for i, s in enumerate(sekiller):
        assert s.png(dpi=50)[:4] == b"\x89PNG" and s.kaydet(tmp_path / f"g{i}.png").stat().st_size > 1000
    r = model.rapor()
    r.sekil_ekle(sekiller[0], "Kalıntılar").sekil_ekle(sekiller[2], "Katsayılar")
    assert [s.no for s in r.sekiller] == [1, 2] and "**Şekil 2.** *Katsayılar*" in r.markdown()
    docx = pytest.importorskip("docx")
    r.kaydet(tmp_path / "r.docx")
    assert len(docx.Document(str(tmp_path / "r.docx")).inline_shapes) == 2
    assert "data:image/png;base64" in r.html()


def test_tablo_disa_aktarma(model, ucret, tmp_path):
    k = ek.karsilastir(ek.ekk("np.log(ucret) ~ egitim", ucret), model)
    for uz in ("docx", "tex", "xlsx", "csv", "md", "html"):
        assert k.kaydet(tmp_path / f"t.{uz}").stat().st_size > 200
    tex = (tmp_path / "t.tex").read_text(encoding="utf-8")
    assert "\\toprule" in tex and "\\bottomrule" in tex and "R$^{2}$" in tex and "$-$0," in tex
    from openpyxl import load_workbook

    sayfa = load_workbook(tmp_path / "t.xlsx").active
    assert sayfa["A1"].value.startswith("Tablo 1.") and sayfa["B2"].value == "Model 1"
    assert pd.read_csv(tmp_path / "t.csv")["Değişken"].iloc[0] == "egitim"
    r = model.rapor()
    r.kaydet(tmp_path / "r.tex")
    belge = (tmp_path / "r.tex").read_text(encoding="utf-8")
    assert belge.startswith("\\documentclass") and "\\end{document}" in belge and "[turkish]{babel}" in belge
    r.kaydet(tmp_path / "r.xlsx")
    assert len(load_workbook(tmp_path / "r.xlsx").sheetnames) == len(r.tablolar)


def test_latex_kacis():
    from ekonometri.rapor import latex_kacis

    assert latex_kacis("β = 0,5; %5 & χ² — x_1") == r"$\beta$ = 0,5; \%5 \& $\chi^{2}$ --- x\_1"


def test_marjlar_r_marginaleffects_ile_ayni(model):
    """Referans: R marginaleffects avg_predictions / avg_comparisons (aynı model ve veri)."""
    t = ek.marjlar(model, at={"egitim": [8, 12, 16]}).tablo
    assert np.allclose(t["Marj"], [10.6297498265, 10.9530907924, 11.2764317583], rtol=1e-10)
    assert np.allclose(t["Std. hata"], [0.0180451001, 0.0112924669, 0.0156736208], rtol=1e-7)
    g = ek.marjlar(model, grup="cinsiyet").tablo
    assert np.allclose(g["Marj"], [11.0435657231, 10.9203919739], rtol=1e-10) and list(g["grup:cinsiyet"]) == ["Erkek", "Kadın"]
    lg = ek.logit("gecti ~ calisma_saati + devam + burs", ek.veri.yukle("sinav"))
    lt = ek.marjlar(lg, at={"calisma_saati": [5, 10, 20]}).tablo
    assert np.allclose(lt["Marj"], [0.7012479647, 0.9014933741, 0.994418148], rtol=1e-8)
    assert np.allclose(lt["Std. hata"], [0.0216069893, 0.0128851555, 0.0025969942], rtol=1e-6)
    b = ek.marjlar(lg, dydx="burs").tablo
    assert np.isclose(b["Marj"].iloc[0], 0.0636193914, rtol=1e-8) and np.isclose(b["Std. hata"].iloc[0], 0.0241272716, rtol=1e-6)


def test_kosullu_etki_analitik(model, ucret):
    """Etkileşimli modelde eğitimin etkisi: Erkek grubunda katsayının kendisi, ortalamada b_e + b_ek·pay(Kadın)."""
    t = ek.marjlar(model, dydx="egitim", grup="cinsiyet").tablo
    b, V = model.katsayilar, model.kovaryans_matrisi
    assert np.isclose(t["Marj"].iloc[0], b["egitim"], rtol=1e-7) and np.isclose(
        t["Std. hata"].iloc[0], model.standart_hatalar["egitim"], rtol=1e-7
    )
    assert np.isclose(t["Marj"].iloc[1], b["egitim"] + b["egitim:cinsiyet[T.Kadın]"], rtol=1e-7)
    pay = (ucret.cinsiyet == "Kadın").mean()
    g = np.zeros(model.k)
    g[list(b.index).index("egitim")], g[list(b.index).index("egitim:cinsiyet[T.Kadın]")] = 1, pay
    o = ek.marjlar(model, dydx="egitim").tablo
    assert np.isclose(o["Std. hata"].iloc[0], np.sqrt(g @ V.values @ g), rtol=1e-7)
    assert o.columns.tolist()[-3] == "p" and "t" in o.columns  # EKK sonrası t


def test_roc_ve_hosmer_lemeshow():
    lg = ek.logit("gecti ~ calisma_saati + devam + burs", ek.veri.yukle("sinav"))
    assert np.isclose(lg.roc().attrs["auc"], 0.839454649827784, rtol=1e-12)  # R pROC::auc
    h = lg.hosmer_lemeshow()
    assert h["sd"] == 8 and len(h["gruplar"]) == 10 and int(h["gruplar"]["N"].sum()) == lg.n
    e = h["gruplar"]
    elle = (((e["Gözlenen 1"] - e["Beklenen 1"]) ** 2) / (e["Beklenen 1"] * (1 - e["Beklenen 1"] / e["N"]))).sum()
    assert np.isclose(h["ki_kare"], elle)
    s = lg.siniflandirma().attrs
    assert {"pozitif_ongoru", "negatif_ongoru"} <= set(s)


def test_var_statsmodels_ile_ayni():
    from statsmodels.tsa.api import VAR

    v = ek.veri.yukle("ihracat")
    d = np.log(v[["ihracat", "reel_kur"]]).assign(b=v["dunya_buyumesi"]).reset_index(drop=True)
    cols = ["ihracat", "reel_kur", "b"]
    m = ek.var(d, cols, gecikme=2)
    r = VAR(d[cols]).fit(2)
    olcek = np.sqrt((m.T - 7) / m.T)  # statsmodels Σ'yı serbestlik derecesiyle, Stata ML ile ölçekler
    assert np.allclose(m.katsayilar.values, np.vstack([r.params.values[1:], r.params.values[:1]]), atol=1e-10)
    assert np.allclose(m.standart_hatalar.values, np.vstack([r.bse.values[1:], r.bse.values[:1]]) * olcek, rtol=1e-9)
    assert np.isclose(m.log_olabilirlik, r.llf)
    oi = r.irf(6).orth_irfs * olcek
    it = m.etki_tepki(6)
    assert all(np.allclose(it[(e, t)], oi[:, cols.index(t), cols.index(e)], atol=1e-12) for e in cols for t in cols)
    s = ek.var_gecikme_secimi(d, cols, 4).attrs["secim"]
    so = VAR(d[cols]).select_order(4).selected_orders
    assert (s["AIC"], s["SBIC"], s["HQIC"], s["FPE"]) == (so["aic"], so["bic"], so["hqic"], so["fpe"])
    assert m.kararlilik().attrs["kararli"]
    assert m.etki_tepki_grafigi(5).png(dpi=40)[:4] == b"\x89PNG"
