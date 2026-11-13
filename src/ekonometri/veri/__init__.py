"""Eğitim amaçlı hazır veri setleri.

⚠️ Bu veri setleri SİMÜLE edilmiştir; gerçek kişi, konut ya da ülke verisi DEĞİLDİR.
Bilinen bir veri üretim süreciyle oluşturuldukları için, tahmin sonuçlarını "gerçek"
katsayılarla karşılaştırarak öğrenmek mümkündür (bkz. ``aciklama(ad)``).

>>> import ekonometri as ek
>>> ek.veri.listele()
>>> veri = ek.veri.yukle("ucret")
>>> print(ek.veri.aciklama("ucret"))
"""

from __future__ import annotations

from importlib import resources

import pandas as pd

KATALOG: dict[str, dict] = {
    "ucret": {
        "baslik": "Ücret ve eğitim (Mincer denklemi)",
        "konu": "Çoklu regresyon, log-düz model, kare terim, kukla değişken, değişen varyans",
        "degiskenler": {
            "ucret": "Aylık net ücret (TL)",
            "egitim": "Eğitim süresi (yıl)",
            "tecrube": "Potansiyel iş tecrübesi (yıl)",
            "yas": "Yaş",
            "cinsiyet": "Erkek / Kadın",
            "bolge": "İstatistiki bölge (7 kategori)",
            "sektor": "Özel / Kamu",
        },
        "gercek_model": "log(ucret) = 9,55 + 0,085·egitim + 0,036·tecrube − 0,0006·tecrube² − 0,13·Kadın + bölge etkisi + 0,06·Kamu + u;  Var(u) eğitimle artar (değişen varyans).",
        "ornek": 'ek.ekk("np.log(ucret) ~ egitim + tecrube + I(tecrube**2) + cinsiyet + sektor + bolge", veri)',
        "kategoriler": {
            "cinsiyet": ["Erkek", "Kadın"],
            "sektor": ["Özel", "Kamu"],
            "bolge": ["İstanbul", "Batı Anadolu", "Ege", "Akdeniz", "Orta Anadolu", "Karadeniz", "Doğu Anadolu"],
        },
    },
    "konut": {
        "baslik": "Konut fiyatları (hedonik fiyat modeli)",
        "konu": "Log-log model, esneklik, çoklu doğrusal bağlantı (metrekare–oda), kategorik değişken",
        "degiskenler": {
            "fiyat": "Satış fiyatı (TL)",
            "metrekare": "Brüt alan (m²)",
            "oda": "Oda sayısı",
            "bina_yasi": "Bina yaşı (yıl)",
            "kat": "Bulunduğu kat (−1 = bodrum)",
            "asansor": "Var / Yok",
            "site": "Site içinde mi (Evet / Hayır)",
            "il": "İl (5 büyük il)",
        },
        "gercek_model": "log(fiyat) = 12,6 + 0,95·log(metrekare) + 0,03·oda − 0,009·bina_yasi + 0,012·kat + 0,07·asansör + 0,12·site + il etkisi + u",
        "ornek": 'ek.ekk("np.log(fiyat) ~ np.log(metrekare) + oda + bina_yasi + asansor + site + il", veri)',
        "kategoriler": {"asansor": ["Yok", "Var"], "site": ["Hayır", "Evet"], "il": ["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"]},
    },
    "sinav": {
        "baslik": "Öğrenci sınav başarısı",
        "konu": "Azalan getiri (kare terim), düz-düz yorum; ileride logit/probit (gecti)",
        "degiskenler": {
            "puan": "Sınav puanı (0–100)",
            "gecti": "Geçti mi (1 = evet)",
            "calisma_saati": "Haftalık çalışma saati",
            "devam": "Derse devam oranı (%)",
            "uyku": "Ortalama günlük uyku (saat)",
            "anne_egitim": "Annenin eğitim süresi (yıl)",
            "burs": "Burslu mu (Evet / Hayır)",
        },
        "gercek_model": "puan = 18 + 2,4·çalışma − 0,035·çalışma² + 0,28·devam + 1,8·(uyku−7) − 0,4·(uyku−7)² + 0,5·anne_eğitim + 3·burs + u",
        "ornek": 'ek.ekk("puan ~ calisma_saati + I(calisma_saati**2) + devam + anne_egitim + burs", veri)',
        "kategoriler": {"burs": ["Hayır", "Evet"]},
    },
    "ihracat": {
        "baslik": "Çeyreklik ihracat (zaman serisi)",
        "konu": "Zaman serisi regresyonu, trend, mevsimsellik, otokorelasyon, HAC standart hata",
        "degiskenler": {
            "donem": "Çeyrek (2005Q1–2024Q4); indeks olarak kullanılır",
            "ihracat": "İhracat (milyar $, simüle)",
            "reel_kur": "Reel efektif kur endeksi (2005Q1 = 100, simüle)",
            "dunya_buyumesi": "Dünya büyümesi (%, simüle; 2009 ve 2020 şokları içerir)",
            "ceyrek": "Yılın çeyreği (1–4)",
        },
        "gercek_model": "log(ihracat) = 2,65 + 0,011·trend + 0,35·log(reel_kur/100) + 0,018·dünya_büyümesi + mevsim + u;  u = 0,65·u(−1) + ε (otokorelasyon).",
        "ornek": 'ek.ekk("np.log(ihracat) ~ np.log(reel_kur) + dunya_buyumesi + C(ceyrek)", veri)',
    },
    "firma": {
        "baslik": "Firma üretim fonksiyonu (panel veri)",
        "konu": "Panel veri: havuzlanmış EKK, sabit ve rassal etkiler, F / Breusch-Pagan LM / Hausman testleri, kümelenmiş standart hata",
        "degiskenler": {
            "firma": "Firma kimliği (150 firma) — birim",
            "yil": "Yıl (2015–2024) — zaman",
            "satis": "Yıllık satış (TL)",
            "calisan": "Çalışan sayısı",
            "sermaye": "Sermaye stoku (TL)",
            "arge": "Ar-Ge harcamasının satışa oranı (%)",
            "ihracatci": "İhracatçı firma mı (Evet / Hayır; zamanla değişmez)",
            "sektor": "İmalat / Hizmet / Ticaret (zamanla değişmez)",
        },
        "gercek_model": "log(satis) = aᵢ + 0,62·log(calisan) + 0,31·log(sermaye) + 0,045·arge + 0,15·ihracatçı + sektör + yıl etkisi + u;  "
        "firma etkisi aᵢ sermaye ve çalışanla ilişkilidir (rassal etkiler tutarsız, Hausman testi sabit etkileri seçmeli).",
        "ornek": 'ek.panel_analizi("np.log(satis) ~ np.log(calisan) + np.log(sermaye) + arge", veri, birim="firma", zaman="yil")',
        "kategoriler": {"ihracatci": ["Hayır", "Evet"], "sektor": ["İmalat", "Hizmet", "Ticaret"]},
    },
    "getiri": {
        "baslik": "Eğitimin getirisi (araç değişken)",
        "konu": "İçsellik, araç değişken (2SLS), zayıf araç, Wu-Hausman içsellik ve Sargan aşırı tanımlama testleri",
        "degiskenler": {
            "ucret": "Aylık ücret (TL)",
            "egitim": "Eğitim süresi (yıl) — içsel: gözlenmeyen yetenekle ilişkili",
            "tecrube": "İş tecrübesi (yıl)",
            "cinsiyet": "Erkek / Kadın",
            "yerlesim": "Kent / Kır (büyüdüğü yer)",
            "uzaklik": "Büyüdüğü yerin en yakın üniversiteye uzaklığı (km) — araç",
            "kardes": "Kardeş sayısı — araç",
        },
        "gercek_model": "log(ucret) = 9 + 0,075·egitim + 0,03·tecrube − 0,0005·tecrube² − 0,12·Kadın + 0,06·Kent + 0,12·yetenek + u;  "
        "yetenek gözlenmez ve eğitimi de artırır, bu yüzden EKK eğitimin getirisini olduğundan yüksek tahmin eder.",
        "ornek": 'ek.iv2sls("np.log(ucret) ~ tecrube + I(tecrube**2) + cinsiyet + yerlesim + [egitim ~ uzaklik + kardes]", veri)',
        "kategoriler": {"cinsiyet": ["Erkek", "Kadın"], "yerlesim": ["Kır", "Kent"]},
    },
}


def listele() -> pd.DataFrame:
    """Mevcut veri setlerinin listesi."""
    return pd.DataFrame([{"ad": ad, "başlık": m["baslik"], "konu": m["konu"]} for ad, m in KATALOG.items()]).set_index("ad")


def yukle(ad: str) -> pd.DataFrame:
    """Veri setini pandas DataFrame olarak yükler. Zaman serileri dönem indeksiyle gelir."""
    if ad not in KATALOG:
        raise KeyError(f"'{ad}' adında veri seti yok. Mevcut: {', '.join(KATALOG)}")
    with resources.files(__package__).joinpath(f"{ad}.csv").open("r", encoding="utf-8") as f:
        df = pd.read_csv(f)
    # ilk kategori = referans grup (yorumlar buna göre yapılır)
    for sutun, sira in KATALOG[ad].get("kategoriler", {}).items():
        df[sutun] = pd.Categorical(df[sutun], categories=sira)
    if "donem" in df.columns:
        df.index = pd.PeriodIndex(df.pop("donem"), freq="Q")
        df.index.name = "donem"
    df.attrs["aciklama"] = KATALOG[ad]["baslik"] + " — SİMÜLE veri (eğitim amaçlı)"
    return df


class _Aciklama(str):
    def __repr__(self) -> str:  # notebook'ta düzgün görünsün
        return str(self)


def aciklama(ad: str) -> str:
    """Veri setinin değişkenleri, gerçek veri üretim süreci ve örnek model."""
    m = KATALOG[ad]
    satirlar = [
        f"📁 {ad} — {m['baslik']}",
        "⚠️ Simüle veri: gerçek gözlem değildir, eğitim amaçlıdır.",
        "",
        f"Konu: {m['konu']}",
        "",
        "Değişkenler:",
    ]
    satirlar += [f"  • {k}: {v}" for k, v in m["degiskenler"].items()]
    satirlar += ["", "Veriyi üreten gerçek model (cevap anahtarı):", f"  {m['gercek_model']}", "", "Örnek:", f"  {m['ornek']}"]
    return _Aciklama("\n".join(satirlar))


__all__ = ["listele", "yukle", "aciklama", "KATALOG"]
