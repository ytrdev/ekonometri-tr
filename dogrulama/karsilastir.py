"""ekonometri-tr sonuçlarını bağımsız ücretsiz yazılımlarla (R paketleri ve gretl) karşılaştırır, dogrulama/SONUC.md yazar.

Gereken: R (estimatr, sandwich, lmtest, plm, AER, urca, tseries), gretl (gretlcli) ve pyfixest (pip install pyfixest).
Çalıştırma:  python dogrulama/karsilastir.py

Not: Stata'nın standart hata kuralları için R tarafında, belgelerinde Stata ile aynı formülü uyguladıkları
yazılan seçenekler kullanılır: estimatr ``se_type = "stata"`` (HC1, kümelenmiş CR1 ve G − 1 sd),
plm ``vcovHC(type = "sss")`` (Stata küçük örneklem düzeltmesi), sandwich ``NeweyWest(adjust = TRUE)``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

KOK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(KOK / "src"))
import ekonometri as ek  # noqa: E402


def r_adi(ad: str) -> str:
    """ekonometri-tr terim adını R'daki karşılığına çevirir."""
    ad = ad.replace("Intercept", "(Intercept)").replace(" ** ", "^").replace("**", "^").replace("np.log", "log")
    m = re.match(r"^(?:C\()?(\w+)\)?\[T\.(.+)\]$", ad)
    return f"{m.group(1)}{m.group(2)}" if m else ad


def ek_sonuclari() -> dict[str, float]:
    s: dict[str, float] = {}

    def yaz(on, seri):
        for k, v in seri.items():
            s[f"{on}[{r_adi(k)}]"] = float(v)

    u = ek.veri.yukle("ucret")
    f = "np.log(ucret) ~ egitim + tecrube + I(tecrube**2) + cinsiyet + bolge"
    m = ek.ekk(f, u)
    yaz("ekk_b", m.katsayilar)
    yaz("ekk_se", m.standart_hatalar)
    yaz("hc1_se", ek.ekk(f, u, kovaryans="HC1").standart_hatalar)
    u["grup"] = np.arange(len(u)) % 40
    k = ek.ekk(f, u, kume="grup")
    yaz("kume_se", k.standart_hatalar)
    yaz("kume_p", k.p_degerleri)
    u["w"] = 0.5 + (np.arange(1, len(u) + 1) % 7) / 3
    yaz("aw_se", ek.ekk(f, u, agirlik="w", agirlik_turu="aw").standart_hatalar)
    yaz("pw_se", ek.ekk(f, u, agirlik="w", agirlik_turu="pw").standart_hatalar)
    s["bp_koenker"] = ek.breusch_pagan(m).istatistik
    s["bp_stata"] = ek.breusch_pagan(m, "tahmin", koenker=False).istatistik
    s["reset_F"] = ek.reset(m).istatistik
    s["bg_lm"] = ek.breusch_godfrey(m, 2).istatistik

    fr = ek.veri.yukle("firma")
    pf = "np.log(satis) ~ np.log(calisan) + np.log(sermaye) + arge"
    fe = ek.panel(pf, fr, birim="firma", zaman="yil", yontem="sabit")
    re_ = ek.panel(pf, fr, birim="firma", zaman="yil", yontem="rassal")
    yaz("fe_b", fe.katsayilar.drop("Intercept"))
    yaz("fe_se", fe.standart_hatalar.drop("Intercept"))
    yaz("fe_kume_se", ek.panel(pf, fr, birim="firma", zaman="yil", yontem="sabit", kovaryans="kume").standart_hatalar.drop("Intercept"))
    yaz("re_b", re_.katsayilar)
    yaz("re_se", re_.standart_hatalar)
    s["hausman"] = ek.hausman(fe, re_, sigmamore=False).istatistik
    from ekonometri.panel import bp_lm_testi, f_testi

    h = ek.panel(pf, fr, birim="firma", zaman="yil", yontem="havuz")
    s["bp_lm"] = bp_lm_testi(h).istatistik
    s["f_birim"] = f_testi(h, fe).istatistik

    g = ek.veri.yukle("getiri")
    iv = ek.iv2sls("np.log(ucret) ~ tecrube + I(tecrube**2) + cinsiyet + [egitim ~ uzaklik + kardes]", g, kucuk_orneklem=True)
    yaz("iv_b", iv.katsayilar)
    yaz("iv_se", iv.standart_hatalar)
    tani = {t.ad.split()[0]: t.istatistik for t in iv.tanilar()}
    s["iv_zayif_F"] = iv.birinci_asama["egitim"]["F"]
    s["iv_wu_hausman"] = tani["Wu-Hausman"]
    s["iv_sargan"] = tani["Sargan"]

    sv = ek.veri.yukle("sinav")
    lg = ek.logit("gecti ~ calisma_saati + devam + burs", sv)
    yaz("logit_b", lg.katsayilar)
    yaz("logit_se", lg.standart_hatalar)
    n = lg.n
    yaz("logit_hc0", ek.logit("gecti ~ calisma_saati + devam + burs", sv, kovaryans="saglam").standart_hatalar / np.sqrt(n / (n - 1)))

    ih = ek.veri.yukle("ihracat")
    x = np.log(ih["ihracat"])
    s["adf_sabit_L3"] = ek.adf(x, "sabit", gecikme=3).istatistik
    s["adf_trend_L2"] = ek.adf(x, "trend", gecikme=2).istatistik
    s["kpss_sabit_L4"] = ek.kpss(x, "sabit", gecikme=4).istatistik
    nw = ek.ekk("np.log(ihracat) ~ np.log(reel_kur) + dunya_buyumesi", ih, kovaryans="newey", gecikme=3)
    for i, v in enumerate(nw.standart_hatalar.values, 1):
        s[f"newey_se[{['(Intercept)', 'log(reel_kur)', 'dunya_buyumesi'][i - 1]}]"] = float(v)

    t = ek.tanimlayici(u, ["ucret"], detay=True)
    for i, q in enumerate(["%1", "%5", "%25", "Medyan", "%75", "%95", "%99"], 1):
        s[f"yuzdelik[{['1%', '5%', '25%', '50%', '75%', '95%', '99%'][i - 1]}]"] = float(t.loc["ucret", q])
    s["ttest_t"] = ek.ttest(u, "ucret", by="cinsiyet").t
    w = ek.ttest(u, "ucret", by="cinsiyet", esit_varyans=False)
    s["welch_t"], s["welch_df"] = w.t, w.sd
    s["chi2"] = ek.ki_kare(u, "cinsiyet", "sektor").ki_kare
    a = ek.anova(u, "ucret", "bolge")
    s["anova_F"], s["bartlett"] = a.F, a.bartlett

    pw = ek.prais("np.log(ihracat) ~ np.log(reel_kur) + dunya_buyumesi", ih)
    s["pw_rho"] = pw.rho
    for i, v in enumerate(pw.katsayilar.values, 1):
        s[f"pw_b[{i}]"] = float(v)
    for i, v in enumerate(pw.standart_hatalar.values, 1):
        s[f"pw_se[{i}]"] = float(v)

    # sıralı / çok terimli / sayım / sansürlü
    sv = sv.copy()
    sv["not_"] = pd.cut(sv["puan"], [0, 50, 65, 80, 101], labels=False) + 1
    for ad, islev in (("ologit", ek.ologit), ("oprobit", ek.oprobit)):
        o = islev("not_ ~ calisma_saati + uyku + burs", sv)
        yaz(f"{ad}_b", o.katsayilar)
        for i, v in enumerate(o.ek_parametreler["Değer"].values, 1):
            s[f"{ad}_kesim[{i}|{i + 1}]"] = float(v)
        s[f"{ad}_ll"] = o.log_olabilirlik
    sv["kat3"] = np.array(["a", "b", "c"])[pd.cut(sv["puan"], [0, 55, 75, 101], labels=False)]
    ml = ek.mlogit("kat3 ~ calisma_saati + uyku", sv, temel="a")
    for i, (b, se) in enumerate(zip(ml.katsayilar.values, ml.standart_hatalar.values), 1):
        s[f"mlogit_b[{i}]"], s[f"mlogit_se[{i}]"] = float(b), float(se)
    s["mlogit_ll"] = ml.log_olabilirlik
    po = ek.poisson("kardes ~ egitim + tecrube", g)
    yaz("poisson_b", po.katsayilar)
    yaz("poisson_se", po.standart_hatalar)
    s["poisson_ll"] = po.log_olabilirlik
    nb = ek.nbreg("kardes ~ egitim + tecrube", g)
    yaz("nbreg_b", nb.katsayilar)
    s["nbreg_alfa"] = float(nb.ek_parametreler.loc["α", "Değer"])
    s["nbreg_ll"] = nb.log_olabilirlik
    sv["kesik"] = np.maximum(sv["puan"] - 70, 0)
    tb = ek.tobit("kesik ~ calisma_saati + uyku", sv, alt=0)
    yaz("tobit_b", tb.katsayilar)
    yaz("tobit_se", tb.standart_hatalar)
    s["tobit_sigma"] = float(tb.ek_parametreler.loc["/σ", "Değer"])
    s["tobit_ll"] = tb.log_olabilirlik
    s["auc"] = lg.roc().attrs["auc"]

    # panel birim kök (plm LLC tabloyu T ile okur → _llc_tablo="T")
    import warnings

    fr["ls"] = np.log(fr["satis"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pbk = ek.panel_birim_kok
        s["llc_sabit_L0"] = pbk(fr, "ls", "llc", "firma", "yil", 0, _llc_tablo="T").istatistikler.loc["Düzeltilmiş t*", "Değer"]
        s["llc_trend_L1"] = pbk(fr, "ls", "llc", "firma", "yil", 1, deterministik="trend", _llc_tablo="T").istatistikler.loc[
            "Düzeltilmiş t*", "Değer"
        ]
        s["ips_sabit_L1"] = pbk(fr, "ls", "ips", "firma", "yil", 1).istatistikler.loc["W-t-bar", "Değer"]
        fi = pbk(fr, "ls", "fisher", "firma", "yil", 0).istatistikler["Değer"]
        s["fisher_Pm_L0"], s["fisher_Z_L0"] = fi["Pm"], fi["Z (ters normal)"]
    return s


GRUPLAR = [
    ("EKK katsayılar ve klasik SH", ["ekk_b", "ekk_se"], "R lm()"),
    ("EKK, Stata robust (HC1)", ["hc1_se"], 'R estimatr se_type="stata"'),
    ("EKK, kümelenmiş SH ve p (G − 1 sd)", ["kume_se", "kume_p"], 'R estimatr se_type="stata", clusters'),
    ("Ağırlıklı EKK [aw=] / [pw=]", ["aw_se", "pw_se"], 'R lm(weights) / estimatr se_type="stata"'),
    (
        "Breusch-Pagan (Koenker ve Stata varsayılanı), RESET (ŷ²–ŷ⁴), Breusch-Godfrey",
        ["bp_koenker", "bp_stata", "reset_F", "bg_lm"],
        "R lmtest",
    ),
    ("Sabit etkiler: katsayı ve klasik SH", ["fe_b", "fe_se"], "R plm within"),
    ("Sabit etkiler: kümelenmiş SH", ["fe_kume_se"], "pyfixest (Stata reghdfe kuralı)"),
    ("Rassal etkiler (Swamy-Arora)", ["re_b", "re_se"], 'R plm random, random.method="swar"'),
    ("Hausman, Breusch-Pagan LM, birim etkisi F testi", ["hausman", "bp_lm", "f_birim"], "R plm phtest / plmtest / pFtest"),
    (
        "2SLS (small) ve tanılar: zayıf araç F, Wu-Hausman, Sargan",
        ["iv_b", "iv_se", "iv_zayif_F", "iv_wu_hausman", "iv_sargan"],
        "R AER ivreg",
    ),
    ("Logit: katsayı, SH, sandviç SH", ["logit_b", "logit_se", "logit_hc0"], "R glm, sandwich"),
    ("ADF (sabit ve trend, sabit gecikme), KPSS", ["adf_sabit_L3", "adf_trend_L2", "kpss_sabit_L4"], "R urca"),
    ("Newey-West (Stata newey)", ["newey_se"], "R sandwich NeweyWest(adjust=TRUE)"),
    ("summarize, detail yüzdelikleri", ["yuzdelik"], "R quantile(type = 2)"),
    (
        "t testi, Welch, χ², ANOVA, Bartlett",
        ["ttest_t", "welch_t", "welch_df", "chi2", "anova_F", "bartlett"],
        "R t.test / chisq.test / oneway.test / bartlett.test",
    ),
    ("Prais-Winsten (yinelemeli)", ["pw_b", "pw_se", "pw_rho"], "gretl ar1 --pwe"),
    (
        "Sıralı logit / probit: katsayı, kesim noktaları, log-olabilirlik",
        ["ologit_b", "ologit_kesim", "ologit_ll", "oprobit_b", "oprobit_kesim", "oprobit_ll"],
        "R MASS polr",
    ),
    ("Çok terimli logit: katsayı, SH, log-olabilirlik", ["mlogit_b", "mlogit_se", "mlogit_ll"], "R nnet multinom"),
    (
        "Poisson ve negatif binom (NB2): katsayı, α, log-olabilirlik",
        ["poisson_b", "poisson_se", "poisson_ll", "nbreg_b", "nbreg_alfa", "nbreg_ll"],
        "R glm / MASS glm.nb",
    ),
    ("Tobit: katsayı, SH, σ, log-olabilirlik", ["tobit_b", "tobit_se", "tobit_sigma", "tobit_ll"], "R AER tobit"),
    ("ROC eğrisi altındaki alan (lroc)", ["auc"], "R pROC"),
    (
        "Panel birim kök: LLC, IPS W-t-bar, Fisher Pm ve Z",
        ["llc_sabit_L0", "llc_trend_L1", "ips_sabit_L1", "fisher_Pm_L0", "fisher_Z_L0"],
        "R plm purtest",
    ),
]


def disari_calistir(dizin: Path) -> dict[str, float]:
    for ad in ("ucret", "firma", "getiri", "sinav", "ihracat"):
        shutil.copy(KOK / "src" / "ekonometri" / "veri" / f"{ad}.csv", dizin / f"{ad}.csv")
    cikti = subprocess.run(
        ["Rscript", str(KOK / "dogrulama" / "r_hesapla.R"), str(dizin)], capture_output=True, text=True, check=True
    ).stdout
    sonuc = {}
    for satir in cikti.splitlines():
        if "=" in satir:
            k, v = satir.rsplit("=", 1)
            sonuc[k.strip()] = float(v)
    shutil.copy(KOK / "dogrulama" / "gretl_prais.inp", dizin)
    g = subprocess.run(["gretlcli", "-b", "gretl_prais.inp"], capture_output=True, text=True, cwd=dizin, check=True).stdout
    for satir in g.splitlines():
        m = re.match(r"^(pw_\w+)=(.+)$", satir.strip())
        if m:
            degerler = [float(x) for x in m.group(2).split(",")]
            if len(degerler) == 1:
                sonuc[m.group(1)] = degerler[0]
            else:
                for i, v in enumerate(degerler, 1):
                    sonuc[f"{m.group(1)}[{i}]"] = v
    # sabit etkiler kümelenmiş SH: pyfixest (Stata reghdfe / xtreg kuralı); plm'nin 'sss' türü K'yı farklı sayar (bkz. not)
    import pyfixest as pf

    v = ek.veri.yukle("firma")
    v["ls"], v["ll"], v["lk"] = np.log(v.satis), np.log(v.calisan), np.log(v.sermaye)
    fx = pf.feols("ls ~ ll + lk + arge | firma", data=v, vcov={"CRV1": "firma"})
    for ad, deger in zip(["log(calisan)", "log(sermaye)", "arge"], fx.se().values):
        sonuc[f"fe_kume_se[{ad}]"] = float(deger)
    return sonuc


def main() -> int:
    ek_s = ek_sonuclari()
    with tempfile.TemporaryDirectory() as d:
        dis = disari_calistir(Path(d))
    satirlar, hepsi_tamam = [], True
    for baslik, onekler, arac in GRUPLAR:
        anahtarlar = [k for k in dis if any(k == o or k.startswith(o + "[") for o in onekler)]
        eksik = [k for k in anahtarlar if k not in ek_s]
        farklar = [abs(ek_s[k] - dis[k]) / max(abs(dis[k]), 1e-12) for k in anahtarlar if k in ek_s]
        en = max(farklar) if farklar else float("nan")
        tamam = bool(farklar) and not eksik and en < 1e-6
        hepsi_tamam &= tamam
        satirlar.append((baslik, arac, len(farklar), en, "✅" if tamam else "❌", eksik))
    md = [
        "# Bağımsız yazılımlarla doğrulama",
        "",
        "Bu tablo `python dogrulama/karsilastir.py` ile üretilir. Her satırda ekonometri-tr'nin sonucu, kodumuzdan bağımsız yazılmış ücretsiz",
        "bir yazılımın sonucuyla karşılaştırılır. Stata'nın standart hata kuralları için R tarafında belgelerinde Stata ile aynı formülü",
        'uyguladıkları yazılan seçenekler kullanılır (estimatr `se_type = "stata"`, sandwich `NeweyWest(adjust = TRUE)`, pyfixest).',
        "",
        "| Karşılaştırma | Bağımsız araç | Değer sayısı | En büyük göreli fark | Sonuç |",
        "|---|---|---:|---:|:---:|",
    ]
    for baslik, arac, n, en, durum, eksik in satirlar:
        md.append(f"| {baslik} | {arac} | {n} | {en:.1e} | {durum}{' eksik: ' + ', '.join(eksik) if eksik else ''} |")
    md += [
        "",
        "Ölçüt: göreli fark < 10⁻⁶.",
        "",
        "**Bu tabloda olmayanlar:**",
        "- Logit/probit dayanıklı standart hatadaki Stata çarpanı N/(N−1) ve rassal etkilerde kümelenmiş standart hata çarpanı G/(G−1)",
        "  bağımsız bir yazılımla doğrulanamadı; tabloda logit için çarpansız sandviç karşılaştırılır.",
        "- Cochrane-Orcutt: gretl daha gevşek bir yakınsama ölçütüyle durduğundan ρ dördüncü basamakta ayrışır;",
        "  ekonometri-tr ρ değişimi 10⁻⁶'nın altına inene kadar (Stata prais varsayılanı) yineler.",
        "- `xttest0`'ın tek yönlü p-değeri ve `dfuller` kritik değer tablosu (bkz. REHBER.md).",
        "- Sıralı modellerin standart hataları: R `polr` Hessian'ı sayısal türevle hesapladığından ~10⁻⁵ göreli farkla tutar (testlerde).",
        "- `nbreg` standart hataları: R `glm.nb` α'yı sabit kabul eder; Stata gibi β ve α'yı birlikte (tam bilgi matrisi) kullanan",
        "  ekonometri-tr ile ~%0,3 farklıdır. Katsayılar, α ve log-olabilirlik birebir aynıdır.",
        "- LLC: plm düzeltme tablosunu T ile okur; ekonometri-tr varsayılan olarak makaledeki ve Stata'daki gibi T̃ = T − p̄ − 1 kullanır.",
        "  Karşılaştırmada aynı tablo satırı seçilir. Kısa panelde (T < 25) iki taraf da tablonun ilk satırını kullanır.",
        '- R `plm` paketinin `vcovHC(type = "sss")` seçeneği küçük örneklem düzeltmesinde sabit terimi saymaz; bu yüzden sabit etkiler',
        "  kümelenmiş standart hataları √((n−1)/(n−K)) farkıyla (bu veride %0,03) ayrışır. ekonometri-tr, pyfixest (Stata reghdfe kuralı) ile birebir aynıdır.",
    ]
    (KOK / "dogrulama" / "SONUC.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    for baslik, _arac, n, en, durum, eksik in satirlar:
        print(f"{durum} {baslik:<75} {n:>4} değer  en büyük fark {en:.1e}" + (f"  EKSİK: {eksik}" if eksik else ""))
    return 0 if hepsi_tamam else 1


if __name__ == "__main__":
    raise SystemExit(main())
