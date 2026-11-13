"""Bulgu raporu: sayılar modelle tutarlı mı, metin doğru dalı seçiyor mu, dosyalar yazılıyor mu?"""

import re

import numpy as np
import pandas as pd
import pytest

import ekonometri as ek

FORMUL = "np.log(ucret) ~ egitim + tecrube + I(tecrube**2) + cinsiyet + sektor + bolge"


@pytest.fixture(scope="module")
def model():
    return ek.ekk(FORMUL, ek.veri.yukle("ucret"), kovaryans="HC1", birimler={"egitim": "yıl", "tecrube": "yıl"})


def _tr(x, basamak):
    return f"{x:.{basamak}f}".replace(".", ",")


def test_bolumler_ve_tablolar(model):
    r = model.rapor()
    basliklar = [b[1] for b in r.bloklar if b[0] == "baslik"]
    assert basliklar == ["Yöntem", "Tanımlayıcı İstatistikler", "Regresyon Bulguları", "Varsayım Testleri", "Kaynakça"]
    assert [t.no for t in r.tablolar] == [1, 2, 3, 4]
    assert isinstance(r, ek.Rapor) and "4 tablo" in repr(r)


def test_regresyon_tablosu_modelle_ayni(model):
    t = next(t for t in model.rapor().tablolar if "regresyon" in t.baslik)
    assert len(t.satirlar) == model.k
    satir = next(s for s in t.satirlar if s[0] == "egitim")
    assert satir[1] == _tr(model.katsayilar["egitim"], 4) + "***"
    assert satir[2] == _tr(model.standart_hatalar["egitim"], 4)
    assert satir[3] == _tr(model.t_degerleri["egitim"], 2)
    assert f"R² = {_tr(model.r2, 3)}" in t.notu and "HC1" in t.notu and "cinsiyet = Erkek" in t.notu


def test_tanimlayici_istatistikler_veriyle_ayni(model):
    veri = ek.veri.yukle("ucret")
    r = model.rapor()
    t1, t2 = r.tablolar[0], r.tablolar[1]
    satir = next(s for s in t1.satirlar if s[0] == "egitim")
    assert satir[2] == _tr(veri["egitim"].mean(), 2) and satir[3] == _tr(veri["egitim"].std(), 2)
    assert {s[0] for s in t1.satirlar} == {"ucret", "egitim", "tecrube"}  # formülde olmayan 'yas' yok
    kadin = next(s for s in t2.satirlar if s[1] == "Kadın")
    assert kadin[2] == str((veri["cinsiyet"] == "Kadın").sum())


def test_metin_yuzde_etki_ve_donum_noktasi(model):
    metin = str(model.rapor())
    b = model.katsayilar
    assert f"%{_tr(100 * (np.exp(b['egitim']) - 1), 2)} artmaktadır" in metin
    assert f"dönüm noktası yaklaşık {_tr(-b['tecrube'] / (2 * b['I(tecrube ** 2)']), 2)} yıl" in metin
    assert "ters U" in metin and "referans grup 'Erkek'" in metin


def test_saglam_ve_klasik_standart_hata_cumleleri(model):
    assert "standart hatalara (HC1) dayandırılmıştır" in str(model.rapor())
    klasik = ek.ekk(FORMUL, ek.veri.yukle("ucret"))
    metin = str(klasik.rapor())
    assert "yeniden tahmin edilmesi gerekir" in metin and "dayandırılmıştır" not in metin


def test_zaman_serisi_otokorelasyon_ve_kaynakca():
    v = ek.veri.yukle("ihracat")
    model = ek.ekk("np.log(ihracat) ~ np.log(reel_kur) + dunya_buyumesi + C(ceyrek)", v, kovaryans="HAC")
    metin = str(model.rapor())
    assert "Breusch-Godfrey" in metin and "Newey, W. K." in metin and "Durbin, J." in metin
    assert "Newey ve West, 1987" in metin
    en = str(model.rapor(dil="en"))
    assert "Newey-West (HAC) standard errors" in en
    assert "Newey ve West" not in en and "Standart hatalar" not in en
    assert "'reel_kur' değişkeni %1 arttığında" in metin
    # C(ceyrek) sayısal sütun olsa da kategorik tabloda yer alır
    r = ek.ekk("np.log(ihracat) ~ C(ceyrek)", v).rapor()
    assert "ceyrek" not in {s[0] for s in r.tablolar[0].satirlar}
    assert "ceyrek" in {s[0] for s in r.tablolar[1].satirlar}


def test_anlamsiz_degisken_etki_iddia_etmez():
    rng = np.random.default_rng(2)
    v = pd.DataFrame({"x": rng.normal(size=200), "gurultu": rng.normal(size=200)})
    v["y"] = 1 + 2 * v["x"] + rng.normal(size=200)
    metin = str(ek.ekk("y ~ x + gurultu", v).rapor())
    assert "'gurultu' değişkeninin 'y' değişkeni üzerindeki etkisi istatistiksel olarak anlamlı bulunmamıştır" in metin
    assert "'x' değişkeni 1 birim arttığında 'y' değişkeni ortalama" in metin


def test_etiket_tablo_no_ve_secenekler(model):
    r = model.rapor(etiketler={"egitim": "Eğitim süresi", "ucret": "Aylık ücret"}, tablo_no=5, yontem=False)
    metin = str(r)
    assert [t.no for t in r.tablolar] == [5, 6, 7, 8]
    assert "Eğitim süresi değişkeni 1 yıl arttığında Aylık ücret değişkeni" in metin
    assert "## Yöntem" not in metin and "## Kaynakça" not in metin
    assert "Tablo 7 ile sunulmuştur" in metin


def test_turkce_ek_hatasi_yok(model):
    """Tırnaklı adlardan hemen sonra ek gelmemeli ('ucret'in gibi)."""
    for m in (model, ek.ekk("puan ~ calisma_saati + np.log(devam) + burs", ek.veri.yukle("sinav"))):
        assert not re.search(r"\w'[a-zçğıöşü]", str(m.rapor()))


def test_dosya_cikti_bicimleri(model, tmp_path):
    docx = pytest.importorskip("docx")
    r = model.rapor(tmp_path / "bulgular.docx")
    belge = docx.Document(str(tmp_path / "bulgular.docx"))
    assert len(belge.tables) == 4
    tum = "\n".join(p.text for p in belge.paragraphs)
    assert "Regresyon Bulguları" in tum and "Wooldridge" in tum
    hucreler = [c.text for c in belge.tables[2].rows[1].cells]
    assert hucreler[0] == "Sabit" and hucreler[1].endswith("***")
    assert "| Değişken |" in r.kaydet(tmp_path / "b.md").read_text(encoding="utf-8")
    assert "<table" in r.kaydet(tmp_path / "b.html").read_text(encoding="utf-8")
    assert "<table" in r._repr_html_()
    with pytest.raises(ek.EkonometriHatasi):
        r.kaydet(tmp_path / "b.pdf")


def test_ingilizce_rapor_ve_tablo_bicimi(model, tmp_path):
    en = model.rapor(dil="en", bicim="izgara")
    metin = en.markdown()
    assert en.baslik == "Findings"
    assert "ordinary least squares" in metin
    assert "Table 1" in metin and "Variable" in metin
    assert "Wooldridge" in metin
    tr = model.rapor()
    assert "Regresyon Bulguları" in tr.markdown() and "Tablo 1" in tr.markdown()
    assert "ordinary least squares" not in tr.markdown()
    en.kaydet(tmp_path / "en.docx")
    tr.kaydet(tmp_path / "tr.docx")
    import zipfile

    def tablo_xml(yol):
        with zipfile.ZipFile(yol) as z:
            return z.read("word/document.xml").decode("utf-8")

    izgara = tablo_xml(tmp_path / "en.docx")
    apa = tablo_xml(tmp_path / "tr.docx")
    assert "<w:left " in izgara
    # APA gövde hücrelerinde sol kenarlık yok; ızgara tablosunda var
    assert izgara.count("<w:left ") > apa.count("<w:left ")


def test_ingilizce_katsayi_ve_varsayim_cumleleri(model):
    metin = model.rapor(dil="en").markdown()
    assert "inverted-U" in metin and "turning point" in metin
    assert "The reference group for" in metin
    assert "Heteroskedasticity was tested" in metin and "Homoskedasticity" in metin
    assert "the assumption holds" in metin
    for tr in ("ters U", "referans grup", "değişkeni", "sınanmıştır", "dönüm noktası"):
        assert tr not in metin

    rng = np.random.default_rng(2)
    v = pd.DataFrame({"x": rng.normal(size=200), "gurultu": rng.normal(size=200)})
    v["y"] = 1 + 2 * v["x"] + rng.normal(size=200)
    duz = str(ek.ekk("y ~ x + gurultu", v).rapor(dil="en"))
    assert "an increase of one unit" in duz and "birim" not in duz

    etk = ek.ekk("np.log(ucret) ~ egitim * cinsiyet + egitim:tecrube + tecrube", ek.veri.yukle("ucret"))
    etk_en = str(etk.rapor(dil="en", yontem=False))
    assert "main-effect coefficients" in etk_en and "ana etki" not in etk_en
    assert "değişkeni" not in etk_en


def test_sayi_bilimsel_gosterim_dile_uyar():
    from ekonometri._yardimci import DIL
    from ekonometri._yardimci import sayi as sayi_yaz

    token = DIL.set("en")
    try:
        assert sayi_yaz(1.2345e-5, 4) == "1.2345e-05"
    finally:
        DIL.reset(token)
    assert sayi_yaz(1.2345e-5, 4) == "1,2345e-05"
