"""Eğitim amaçlı SİMÜLE veri setlerini üretir (gerçek veri DEĞİLDİR).

Her veri seti bir ekonometri konusunu öğretmek için bilinen bir veri üretim süreciyle (DGP)
oluşturulur; böylece öğrenci tahmin sonucunu "doğru cevapla" karşılaştırabilir.

Çalıştırma:  python scripts/veri_uret.py   ->  src/ekonometri/veri/*.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

HEDEF = Path(__file__).resolve().parents[1] / "src" / "ekonometri" / "veri"
BOLGELER = ["İstanbul", "Batı Anadolu", "Ege", "Akdeniz", "Orta Anadolu", "Karadeniz", "Doğu Anadolu"]
BOLGE_ETKI = {
    "İstanbul": 0.18,
    "Batı Anadolu": 0.10,
    "Ege": 0.06,
    "Akdeniz": 0.02,
    "Orta Anadolu": -0.03,
    "Karadeniz": -0.04,
    "Doğu Anadolu": -0.10,
}


def ucret(n=1500, tohum=2026):
    r = np.random.default_rng(tohum)
    egitim = r.choice([5, 8, 11, 12, 14, 16, 18, 20], size=n, p=[0.08, 0.17, 0.05, 0.30, 0.10, 0.22, 0.06, 0.02])
    yas = r.integers(18, 65, size=n)
    tecrube = np.clip(yas - egitim - 6 + r.integers(-2, 3, size=n), 0, None)
    cinsiyet = r.choice(["Erkek", "Kadın"], size=n, p=[0.6, 0.4])
    bolge = r.choice(BOLGELER, size=n, p=[0.22, 0.14, 0.14, 0.13, 0.12, 0.13, 0.12])
    sektor = r.choice(["Özel", "Kamu"], size=n, p=[0.78, 0.22])
    # gerçek ilişki (Mincer): log(ücret) = 9,55 + 0,085·eğitim + 0,036·tecrübe − 0,0006·tecrübe² − 0,13·kadın + bölge + 0,06·kamu + u
    sigma = 0.22 + 0.016 * egitim  # bilerek değişen varyans: eğitim arttıkça ücret dağılımı genişler
    u = r.normal(0, sigma)
    logw = (
        9.55
        + 0.085 * egitim
        + 0.036 * tecrube
        - 0.0006 * tecrube**2
        - 0.13 * (cinsiyet == "Kadın")
        + np.vectorize(BOLGE_ETKI.get)(bolge)
        + 0.06 * (sektor == "Kamu")
        + u
    )
    return pd.DataFrame(
        {
            "ucret": np.round(np.exp(logw), -1).astype(int),
            "egitim": egitim,
            "tecrube": tecrube,
            "yas": yas,
            "cinsiyet": cinsiyet,
            "bolge": bolge,
            "sektor": sektor,
        }
    )


def konut(n=1000, tohum=7):
    r = np.random.default_rng(tohum)
    il = r.choice(["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"], size=n, p=[0.35, 0.2, 0.18, 0.12, 0.15])
    metrekare = np.round(np.clip(r.lognormal(np.log(115), 0.33, n), 45, 400))
    oda = np.clip(np.round(metrekare / 38 + r.normal(0, 0.45, n)), 1, 7).astype(int)  # metrekare ile güçlü ilişkili (VIF)
    bina_yasi = np.clip(r.gamma(2.2, 7, n).round(), 0, 60).astype(int)
    kat = np.clip(r.integers(-1, 16, n), -1, None)
    asansor = np.where((kat >= 4) | (r.random(n) < 0.35), "Var", "Yok")
    site = r.choice(["Evet", "Hayır"], size=n, p=[0.3, 0.7])
    il_etki = {"İstanbul": 0.55, "Ankara": 0.05, "İzmir": 0.30, "Bursa": 0.0, "Antalya": 0.25}
    logp = (
        12.6
        + 0.95 * np.log(metrekare)
        + 0.03 * oda
        - 0.009 * bina_yasi
        + 0.012 * np.clip(kat, 0, 10)
        + 0.07 * (asansor == "Var")
        + 0.12 * (site == "Evet")
        + np.vectorize(il_etki.get)(il)
        + r.normal(0, 0.18, n)
    )
    return pd.DataFrame(
        {
            "fiyat": np.round(np.exp(logp), -3).astype(int),
            "metrekare": metrekare.astype(int),
            "oda": oda,
            "bina_yasi": bina_yasi,
            "kat": kat,
            "asansor": asansor,
            "site": site,
            "il": il,
        }
    )


def sinav(n=800, tohum=11):
    r = np.random.default_rng(tohum)
    calisma = np.clip(r.gamma(3, 3.2, n), 0, 40).round(1)
    devam = np.clip(r.beta(6, 1.6, n) * 100, 20, 100).round()
    uyku = np.clip(r.normal(7, 1.1, n), 4, 10).round(1)
    anne_egitim = r.choice([5, 8, 12, 16], size=n, p=[0.3, 0.25, 0.3, 0.15])
    burs = r.choice(["Evet", "Hayır"], size=n, p=[0.25, 0.75])
    puan = (
        18
        + 2.4 * calisma
        - 0.035 * calisma**2
        + 0.28 * devam
        + 1.8 * (uyku - 7)
        - 0.4 * (uyku - 7) ** 2
        + 0.5 * anne_egitim
        + 3 * (burs == "Evet")
        + r.normal(0, 9, n)
    )
    puan = np.clip(puan, 0, 100).round(1)
    gecti = (puan + r.normal(0, 4, n) >= 50).astype(int)
    return pd.DataFrame(
        {"puan": puan, "gecti": gecti, "calisma_saati": calisma, "devam": devam, "uyku": uyku, "anne_egitim": anne_egitim, "burs": burs}
    )


def ihracat(tohum=3):
    r = np.random.default_rng(tohum)
    donem = pd.period_range("2005Q1", "2024Q4", freq="Q")
    T = len(donem)
    t = np.arange(T)
    reel_kur = 100 * np.exp(np.cumsum(r.normal(0.004, 0.035, T)))
    dunya = np.clip(3.2 + r.normal(0, 1.0, T), -4, 7)
    dunya[16:20] -= 3.5  # 2009 küresel kriz
    dunya[60:62] -= 6  # 2020 salgın şoku
    u = np.zeros(T)
    for i in range(1, T):  # bilerek otokorelasyon: AR(1), ρ = 0,65
        u[i] = 0.65 * u[i - 1] + r.normal(0, 0.035)
    mevsim = np.tile([-0.03, 0.01, 0.0, 0.02], T // 4)
    log_ihr = 2.65 + 0.011 * t + 0.35 * np.log(reel_kur / 100) + 0.018 * dunya + mevsim + u
    return pd.DataFrame(
        {
            "donem": donem.astype(str),
            "ihracat": np.exp(log_ihr).round(2),
            "reel_kur": reel_kur.round(2),
            "dunya_buyumesi": dunya.round(2),
            "ceyrek": donem.quarter,
        }
    )


def firma(N=150, T=10, tohum=21):
    """Panel veri: firmaların üretim fonksiyonu. Firma etkisi sermaye ile ilişkili (sabit etkiler tercih edilmeli)."""
    r = np.random.default_rng(tohum)
    sektorler = ["İmalat", "Hizmet", "Ticaret"]
    sektor = r.choice(sektorler, size=N, p=[0.45, 0.3, 0.25])
    ihracatci = r.choice(["Hayır", "Evet"], size=N, p=[0.6, 0.4])
    a_i = r.normal(0, 0.45, N)  # gözlenmeyen firma etkisi (yönetim kalitesi)
    satir = []
    for i in range(N):
        ln_k0 = 2.5 + 0.8 * a_i[i] + r.normal(0, 0.5)  # iyi yönetilen firma daha çok yatırım yapar -> E(a|x) ≠ 0
        ln_l0 = 3.6 + 0.4 * a_i[i] + r.normal(0, 0.6)
        for t in range(T):
            ln_k = ln_k0 + 0.04 * t + r.normal(0, 0.15)
            ln_l = ln_l0 + 0.02 * t + r.normal(0, 0.12)
            arge = max(0.0, round(r.normal(1.5 + 0.8 * (ihracatci[i] == "Evet"), 1.0), 2))
            ln_q = (
                1.2
                + a_i[i]
                + 0.62 * ln_l
                + 0.31 * ln_k
                + 0.045 * arge
                + 0.15 * (ihracatci[i] == "Evet")
                + {"İmalat": 0.0, "Hizmet": 0.1, "Ticaret": -0.05}[sektor[i]]
                + 0.02 * t
                - 0.08 * (2015 + t == 2020)
                + r.normal(0, 0.12)
            )
            satir.append(
                {
                    "firma": f"F{i + 1:03d}",
                    "yil": 2015 + t,
                    "satis": round(float(np.exp(ln_q + 9)), -3),
                    "calisan": int(round(np.exp(ln_l))),
                    "sermaye": round(float(np.exp(ln_k + 9)), -3),
                    "arge": arge,
                    "ihracatci": ihracatci[i],
                    "sektor": sektor[i],
                }
            )
    return pd.DataFrame(satir)


def getiri(n=2000, tohum=5):
    """Araç değişken: eğitim içsel (gözlenmeyen yetenekle ilişkili); araçlar: üniversiteye uzaklık ve kardeş sayısı."""
    r = np.random.default_rng(tohum)
    yetenek = r.normal(0, 1, n)
    uzaklik = np.round(r.gamma(2.0, 25, n), 1)  # km
    kardes = r.poisson(2.2, n)
    kadin = r.random(n) < 0.45
    kent = r.random(n) < 0.7
    egitim = np.clip(np.round(11.5 + 0.9 * yetenek - 0.025 * uzaklik - 0.35 * kardes + 0.8 * kent + r.normal(0, 1.6, n)), 5, 20)
    tecrube = r.integers(1, 30, n)
    ln_w = 9.0 + 0.075 * egitim + 0.03 * tecrube - 0.0005 * tecrube**2 - 0.12 * kadin + 0.06 * kent + 0.12 * yetenek + r.normal(0, 0.3, n)
    return pd.DataFrame(
        {
            "ucret": np.round(np.exp(ln_w), -1).astype(int),
            "egitim": egitim.astype(int),
            "tecrube": tecrube,
            "cinsiyet": np.where(kadin, "Kadın", "Erkek"),
            "yerlesim": np.where(kent, "Kent", "Kır"),
            "uzaklik": uzaklik,
            "kardes": kardes,
        }
    )


if __name__ == "__main__":
    HEDEF.mkdir(parents=True, exist_ok=True)
    for ad, f in {"ucret": ucret, "konut": konut, "sinav": sinav, "ihracat": ihracat, "firma": firma, "getiri": getiri}.items():
        df = f()
        df.to_csv(HEDEF / f"{ad}.csv", index=False)
        print(f"{ad}: {df.shape[0]} satır, {df.shape[1]} sütun")
