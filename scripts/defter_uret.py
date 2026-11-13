"""Ders defterlerini (Jupyter) üretir ve çalıştırıp çıktılarıyla kaydeder.

Çalıştırma:  python scripts/defter_uret.py
"""

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

KOK = Path(__file__).resolve().parents[1]
HEDEF = KOK / "defterler"


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def kod(s):
    return nbf.v4.new_code_cell(s.strip())


def defter_01():
    return [
        md("""
# 01 · Çoklu Regresyon: Eğitim Ücreti Ne Kadar Artırır?

Bu derste **Mincer ücret denklemini** tahmin edeceğiz: eğitimin, iş tecrübesinin ve bazı kişisel özelliklerin ücrete etkisi.

**Öğreneceklerin**
1. EKK (En Küçük Kareler) regresyonu kurmak ve sonuç tablosunu okumak
2. Log-düz modelde katsayıları **yüzde değişim** olarak yorumlamak
3. Kare terimle **azalan getiri** (ters U) ilişkisini modellemek
4. Kukla değişkenleri ve **referans grubu** yorumlamak
5. Varsayımları test etmek ve **değişen varyansa** karşı sağlam standart hata kullanmak

> ⚠️ Kullandığımız veri **simüle edilmiştir** (gerçek kişiler değildir). Avantajı: veriyi üreten gerçek modeli biliyoruz, tahminlerimizi "cevap anahtarı" ile karşılaştırabiliriz.
"""),
        kod("""
import numpy as np
import ekonometri as ek

veri = ek.veri.yukle("ucret")
veri.head()
"""),
        md("Veri setinin değişkenlerine ve onu üreten **gerçek modele** bakalım:"),
        kod('print(ek.veri.aciklama("ucret"))'),
        md("""
## 1. Basit bir başlangıç

Önce yalnızca eğitimi kullanalım. Bağımlı değişken olarak ücretin **logaritmasını** alıyoruz; böylece katsayılar yüzde değişim olarak yorumlanır ve ücretlerin sağa çarpık dağılımı dengelenir.
"""),
        kod("""
basit = ek.ekk("np.log(ucret) ~ egitim", veri, birimler={"egitim": "yıl"})
basit
"""),
        kod("basit.yorumla()"),
        md("""
## 2. Tam model

Eğitimin etkisi, tecrübe ve diğer özellikler kontrol edilmeden yanlı ölçülebilir (**atlanmış değişken yanlılığı**). Tecrübeyi kare terimiyle ekliyoruz: tecrübe ücreti artırır ama giderek azalan hızla.

Formülde:
- `I(tecrube**2)` → tecrübenin karesi
- `cinsiyet`, `sektor`, `bolge` → metin sütunları otomatik olarak **kukla değişkene** çevrilir; ilk kategori referans gruptur
"""),
        kod("""
model = ek.ekk(
    "np.log(ucret) ~ egitim + tecrube + I(tecrube**2) + cinsiyet + sektor + bolge",
    veri,
    birimler={"egitim": "yıl", "tecrube": "yıl"},
)
model
"""),
        kod("model.yorumla()"),
        md("""
**Gerçek modelle karşılaştır:** veriyi üreten modelde eğitimin katsayısı 0,085, kadın olmanın etkisi −0,13 idi.
Tahminlerimiz bu değerlere ne kadar yakın? Örneklem büyüdükçe EKK tahmincisi gerçek değere yaklaşır (**tutarlılık**).
"""),
        kod("model.guven_araliklari().loc[['egitim', 'cinsiyet[T.Kadın]']]"),
        md("""
## 3. Varsayımlar sağlanıyor mu?

EKK katsayılarının ve özellikle **standart hatalarının** güvenilir olması için bazı varsayımlar gerekir. `model.varsayimlar()` hepsini test eder ve ne yapman gerektiğini söyler.
"""),
        kod("model.varsayimlar()"),
        md("""
Breusch-Pagan testi **değişen varyans** buldu: eğitimli kişilerin ücretleri daha geniş bir aralıkta dağılıyor. Bu durumda katsayılar hâlâ yansızdır ama klasik standart hatalar yanlıştır.

Çözüm: **değişen varyansa dayanıklı (robust) standart hatalar.** Stata'daki `, robust` seçeneğinin karşılığı `kovaryans="HC1"`.
"""),
        kod("""
saglam = ek.ekk(
    "np.log(ucret) ~ egitim + tecrube + I(tecrube**2) + cinsiyet + sektor + bolge",
    veri,
    kovaryans="HC1",
    birimler={"egitim": "yıl", "tecrube": "yıl"},
)
saglam
"""),
        md("Katsayılar **aynı**, yalnızca standart hatalar (ve t/p değerleri) değişti. İki modeli yan yana koyalım:"),
        kod("""
import pandas as pd

pd.DataFrame({
    "katsayı": model.katsayilar,
    "klasik std. hata": model.standart_hatalar,
    "sağlam std. hata (HC1)": saglam.standart_hatalar,
}).round(4)
"""),
        md("""
## 4. Hipotez testi

"Eğitimin getirisi gerçekten %8,5 mi?" sorusunu **Wald testi** ile sınayalım (H0: katsayı = 0,085):
"""),
        kod('saglam.wald_testi("egitim = 0.085")'),
        md("Tecrübe ücreti hiç etkilemiyor mu? (H0: tecrübe ve karesinin katsayıları birlikte sıfır)"),
        kod('saglam.wald_testi(["tecrube", "I(tecrube ** 2)"])'),
        md("""
## 5. Tahmin

16 yıl eğitimli, 10 yıl tecrübeli, İstanbul'da özel sektörde çalışan bir kadının beklenen ücreti:
"""),
        kod("""
kisi = pd.DataFrame({"egitim": [16], "tecrube": [10], "cinsiyet": ["Kadın"], "sektor": ["Özel"], "bolge": ["İstanbul"]})
log_tahmin = saglam.tahmin(kisi)
print(f"Beklenen aylık ücret ≈ {np.exp(log_tahmin.iloc[0]):,.0f} TL".replace(",", "."))
"""),
        md("""
## 6. Bulgu raporu

`rapor()` Yöntem ve Bulgular metnini tablolarla üretir. Word dosyası için: `saglam.rapor("bulgular.docx")`

> Rapor, modelin hesapladığı sayıları cümleye döker. Metni okuyup kendi çalışmana göre düzelt.
"""),
        kod("""
saglam.rapor(etiketler={"ucret": "Aylık ücret", "egitim": "Eğitim süresi", "tecrube": "Tecrübe",
                        "cinsiyet": "Cinsiyet", "sektor": "Sektör", "bolge": "Bölge"})
"""),
    ]


def defter_02():
    return [
        md("""
# 02 · Zaman Serisi: Durağanlık, Sahte Regresyon ve Eşbütünleşme

Zaman serisi verisiyle regresyon kurmadan önce sorulması gereken soru: **seriler durağan mı?** Durağan olmayan iki seri arasında gerçekte hiçbir ilişki yokken bile yüksek R² ve "anlamlı" katsayılar çıkabilir (**sahte regresyon**, Granger ve Newbold, 1974).

**Öğreneceklerin**
1. ADF ve KPSS testleriyle durağanlığı sınamak ve iki testi birlikte okumak
2. Bütünleşme derecesine (I(0) / I(1)) karar vermek ve bunu tez tablosu olarak raporlamak
3. Sahte regresyonu kendi gözünle görmek
4. Engle-Granger testiyle eşbütünleşmeyi (uzun dönem ilişkisini) sınamak

> ⚠️ Kullanılan veriler **simüle edilmiştir**.
"""),
        kod("""
import numpy as np
import pandas as pd
import ekonometri as ek

veri = ek.veri.yukle("ihracat")
veri.head()
"""),
        md("""
## 1. Tek bir seri: ADF ve KPSS

- **ADF**: H0 = seride birim kök var (durağan değil). Reddedersen seri durağandır.
- **KPSS**: H0 = seri durağan. Reddedersen seri durağan değildir.

İki testin hipotezleri **ters** olduğu için birlikte kullanılır: ikisi aynı sonuca işaret ediyorsa karar güçlüdür.
"""),
        kod('ek.adf(np.log(veri["reel_kur"]), ad="ln(reel_kur)")'),
        kod('ek.kpss(np.log(veri["reel_kur"]), ad="ln(reel_kur)")'),
        md("""
## 2. Tez tablosu: tüm değişkenler, düzey ve birinci fark

`ek.duraganlik()` her değişken için düzeyde ve birinci farkta ADF + KPSS uygular ve bütünleşme derecesine karar verir. Gecikme sayısı AIC ile seçilir (köşeli parantez içinde).
"""),
        kod("""
d = ek.duraganlik(veri, ["ihracat", "reel_kur", "dunya_buyumesi"], log=["ihracat", "reel_kur"])
d
"""),
        kod("print(d.rapor().paragraflar[0])"),
        md("""
Bu bölüm Word'e doğrudan aktarılabilir: `d.rapor("duraganlik.docx")`.

## 3. Sahte regresyon deneyi

Birbirinden **tamamen bağımsız** iki rastgele yürüyüş üretelim ve birini diğerine regresleyelim:
"""),
        kod("""
rng = np.random.default_rng(1)
sahte = pd.DataFrame({"a": np.cumsum(rng.normal(size=200)), "b": np.cumsum(rng.normal(size=200))})
m = ek.ekk("a ~ b", sahte)
print(f"R² = {m.r2:.2f},  b katsayısının p-değeri = {m.p_degerleri['b']:.2g}")
"""),
        md('Aralarında hiçbir ilişki olmadığı hâlde katsayı "anlamlı" görünebilir. Engle-Granger testi bu tuzağı yakalar:'),
        kod('ek.engle_granger("a ~ b", sahte)'),
        md("""
## 4. Gerçek bir uzun dönem ilişkisi

Şimdi ortak bir stokastik trendi paylaşan iki seri üretelim: `y = 2 + 0,8·x + durağan hata`. Seriler tek tek durağan değil ama aralarındaki fark durağan: **eşbütünleşik**.
"""),
        kod("""
x = np.cumsum(rng.normal(size=200))
es = pd.DataFrame({"x": x, "y": 2 + 0.8 * x + rng.normal(size=200)})
eg = ek.engle_granger("y ~ x", es)
eg
"""),
        kod("print(eg.rapor().paragraflar[0])"),
    ]


def defter_03():
    return [
        md("""
# 03 · Logit ve Probit: Sınavı Geçme Olasılığı

Bağımlı değişken **0/1** olduğunda (geçti/kaldı, işe girdi/girmedi, kredi ödendi/ödenmedi) doğrusal regresyon 0–1 aralığının dışına çıkan olasılıklar üretebilir. Logit ve probit modelleri olasılığı her zaman 0 ile 1 arasında tutar.

**Öğreneceklerin**
1. Logit ve probit modellerini tahmin etmek
2. Katsayıları değil **ortalama marjinal etkileri (AME)** yorumlamak
3. Odds oranlarını ve sınıflandırma tablosunu okumak
4. Sonuçları tez formatında raporlamak

> ⚠️ Kullanılan veri **simüle edilmiştir**.
"""),
        kod("""
import ekonometri as ek

veri = ek.veri.yukle("sinav")
veri["gecti"].mean()
"""),
        md("## 1. Logit modeli"),
        kod("""
logit = ek.logit("gecti ~ calisma_saati + devam + anne_egitim + burs", veri, birimler={"calisma_saati": "saat", "devam": "puan"})
logit
"""),
        md("""
Tablonun altındaki uyarıya dikkat: logit katsayıları **olasılıktaki değişim değildir**, yalnızca yönü gösterir. Olasılık yorumu için marjinal etkilere bakılır:
"""),
        kod("logit.marjinal_etkiler()[['AME', 'std. hata', 'p']]"),
        kod("logit.yorumla()"),
        md("## 2. Odds oranları ve sınıflandırma"),
        kod("logit.odds_oranlari()"),
        kod("logit.siniflandirma()"),
        md("""
Doğru sınıflandırma oranı yüksek görünse de örneklemin büyük kısmı zaten sınavı geçmiş: **taban oran** ile karşılaştırmadan bu sayıya güvenme. Özgüllük (kalanları doğru bilme oranı) çok daha düşük.

## 3. Logit, probit ve doğrusal olasılık modeli yan yana

Katsayılar ölçek farkı nedeniyle farklıdır ama marjinal etkiler birbirine çok yakındır:
"""),
        kod("""
formul = "gecti ~ calisma_saati + devam + anne_egitim + burs"
probit = ek.probit(formul, veri)
dom = ek.ekk(formul, veri, kovaryans="HC1")   # doğrusal olasılık modeli
ek.karsilastir(dom, logit, probit, adlar=["DOM (EKK)", "Logit", "Probit"])
"""),
        kod("""
import pandas as pd
pd.DataFrame({"Logit AME": logit.marjinal_etkiler()["AME"], "Probit AME": probit.marjinal_etkiler()["AME"]}).round(4)
"""),
        md("## 4. Tez raporu"),
        kod('print(logit.rapor(etiketler={"calisma_saati": "Haftalık çalışma süresi", "gecti": "Geçme"}).paragraflar[4][:900])'),
        md(
            'Tam rapor (Yöntem, tanımlayıcı istatistikler, katsayı ve marjinal etki tabloları, sınıflandırma tablosu, kaynakça): `logit.rapor("logit.docx")`'
        ),
    ]


def defter_04():
    return [
        md("""
# 04 · Panel Veri: Sabit Etkiler mi, Rassal Etkiler mi?

Aynı birimleri (firma, il, ülke) birden çok dönem gözlediğimizde **gözlenmeyen ama zamanla değişmeyen** farkları (yönetim kalitesi, kültür, coğrafya) kontrol edebiliriz. Bu, panel verinin en büyük gücüdür.

**Öğreneceklerin**
1. Havuzlanmış EKK, sabit etkiler ve rassal etkiler modellerini kurmak
2. F, Breusch-Pagan LM ve Hausman testleriyle model seçmek
3. Kümelenmiş standart hata ve zaman etkilerini kullanmak
4. Panel analizini tez bölümü olarak raporlamak

> ⚠️ Kullanılan veri **simüle edilmiştir**. Gerçek modelde firma etkisi sermaye ile ilişkilidir; bu yüzden rassal etkiler tutarsızdır.
"""),
        kod("""
import ekonometri as ek

veri = ek.veri.yukle("firma")
veri.head()
"""),
        kod('print(ek.veri.aciklama("firma"))'),
        md("""
## 1. Tek komutla panel analizi

`ek.panel_analizi()` üç modeli tahmin eder, seçim testlerini uygular ve uygun modeli önerir:
"""),
        kod("""
formul = "np.log(satis) ~ np.log(calisan) + np.log(sermaye) + arge + ihracatci"
analiz = ek.panel_analizi(formul, veri, birim="firma", zaman="yil")
analiz
"""),
        md("""
**Ne gördük?**
- Havuzlanmış EKK sermayenin etkisini olduğundan çok büyük tahmin ediyor: iyi yönetilen firmalar hem daha çok yatırım yapıyor hem daha çok satıyor (atlanmış değişken yanlılığı).
- Hausman testi rassal etkileri reddediyor: firma etkisi açıklayıcı değişkenlerle ilişkili.
- `ihracatci` zamanla değişmediği için sabit etkiler modelinde tahmin edilemiyor.

## 2. Zaman etkileri: iki yönlü model

Tüm firmaları aynı anda etkileyen trendler ve şoklar (2020 gibi) ayıklanmazsa katsayılara karışır. Gerçek parametreler 0,62 (çalışan) ve 0,31 (sermaye):
"""),
        kod("""
iki_yonlu = ek.panel(formul, veri, birim="firma", zaman="yil", yontem="sabit", zaman_etkisi=True, kovaryans="kume")
iki_yonlu
"""),
        kod("iki_yonlu.yorumla()"),
        md("## 3. Tez raporu"),
        kod("""
analiz2 = ek.panel_analizi(formul, veri, birim="firma", zaman="yil", zaman_etkisi=True)
rapor = analiz2.rapor(etiketler={"calisan": "Çalışan sayısı", "sermaye": "Sermaye stoku", "arge": "Ar-Ge yoğunluğu"})
print(rapor.paragraflar[2])
"""),
        md(
            'Tam rapor (Yöntem, tanımlayıcı istatistikler, model seçim testleri tablosu, üç modelin karşılaştırma tablosu, yorum, kaynakça): `analiz2.rapor("panel.docx")`'
        ),
    ]


def defter_05():
    return [
        md("""
# 05 · Araç Değişken: Eğitimin Gerçek Getirisi

Daha yetenekli insanlar hem daha uzun okur hem daha çok kazanır. Yetenek gözlenmediği için EKK, eğitimin getirisini **olduğundan büyük** tahmin eder (yetenek yanlılığı). Çözüm: eğitimi etkileyen ama ücreti **yalnızca eğitim aracılığıyla** etkileyen bir **araç değişken**.

Bu derste Card (1995) yaklaşımını izliyoruz: büyüdüğün yerin üniversiteye uzaklığı ve kardeş sayısı.

**Öğreneceklerin**
1. 2SLS ile içsel değişkeni tahmin etmek
2. Zayıf araç, içsellik ve aşırı tanımlama testlerini okumak
3. EKK ve 2SLS sonuçlarını karşılaştırıp raporlamak

> ⚠️ Kullanılan veri **simüle edilmiştir**: gerçek getiri 0,075.
"""),
        kod("""
import ekonometri as ek

veri = ek.veri.yukle("getiri")
print(ek.veri.aciklama("getiri"))
"""),
        md("## 1. EKK: yanlı tahmin"),
        kod("""
ekk = ek.ekk("np.log(ucret) ~ egitim + tecrube + I(tecrube**2) + cinsiyet + yerlesim", veri)
ekk.katsayilar["egitim"]
"""),
        md("""
## 2. 2SLS

İçsel değişken ve araçları formülde köşeli parantezle yazılır: `[egitim ~ uzaklik + kardes]`.
"""),
        kod("""
iv = ek.iv2sls("np.log(ucret) ~ tecrube + I(tecrube**2) + cinsiyet + yerlesim + [egitim ~ uzaklik + kardes]", veri,
               birimler={"egitim": "yıl", "tecrube": "yıl"})
iv
"""),
        md("""
## 3. Tanı testleri

- **Birinci aşama F**: araçlar eğitimi yeterince güçlü açıklıyor mu? (F < 10 → zayıf araç) — Stata: `estat firststage`
- **Durbin ve Wu-Hausman**: eğitim gerçekten içsel mi? (içsel değilse EKK daha etkin) — Stata: `estat endogenous`
- **Sargan ve Basmann**: araç sayısı içsel değişken sayısından fazlaysa, araçlar birbiriyle tutarlı mı? — Stata: `estat overid`

Standart hatalar Stata `ivregress 2sls` varsayılanıyla (büyük örneklem, z istatistiği) hesaplanır; `small` seçeneği için `kucuk_orneklem=True`.
"""),
        kod("for t in iv.tanilar():\n    print(t, end='\\n\\n')"),
        kod("iv.yorumla()"),
        md("## 4. EKK ve 2SLS yan yana"),
        kod('ek.karsilastir(iv.ekk, iv, adlar=["EKK", "2SLS"])'),
        md("""
Tam tez bölümü (Yöntem, birinci aşama tablosu, tanı testleri, karşılaştırma ve yorum): `iv.rapor("arac_degisken.docx")`

> **Önemli:** Araçların geçerliliği (dışlama kısıtı) istatistiksel olarak tam sınanamaz. Sargan testi yalnızca araçların *birbiriyle* tutarlılığına bakar; hepsi aynı yönde geçersizse yakalayamaz. Araç seçimini kuramla savunmak gerekir.
"""),
    ]


def defter_06():
    do_metni = (
        "use firma_ornek.dta, clear\n"
        "gen lns = ln(satis)\n"
        "gen lnk = ln(sermaye)\n"
        "* havuzlanmış modeller\n"
        "quietly regress lns lnk, vce(cluster firma)\n"
        "estimates store m1\n"
        "quietly regress lns lnk arge i.ihracatci, ///\n"
        "    vce(cluster firma)\n"
        "estimates store m2\n"
        "esttab m1 m2\n"
    )
    return [
        md("""
# 06 · Kendi Verin ve Stata Komutları

Bu derste:
1. Kendi veri dosyanı (Stata `.dta`, Excel, CSV) yüklemeyi,
2. Değişken etiketlerinin raporlara nasıl yansıdığını,
3. **Stata komutlarıyla** (regress, xtreg, estat, esttab...) analiz yapmayı,
4. Hazır bir **do dosyasını** çalıştırmayı ve sonucu Word raporuna dönüştürmeyi öğreneceksin.

> Örnek dosyaları defterin içinde simüle veriden üretiyoruz; kendi çalışmanda `ek.oku("dosyan.dta")` yazman yeterli.
"""),
        kod("""
import ekonometri as ek

# örnek bir Stata dosyası hazırlayalım (kendi verin varsa bu adımı atla)
veri = ek.veri.yukle("firma")
ek.etiketle(veri, {"satis": "Yıllık satış (TL)", "calisan": "Çalışan sayısı", "sermaye": "Sermaye stoku (TL)", "arge": "Ar-Ge yoğunluğu (%)"})
ek.kaydet(veri, "firma_ornek.dta")
"""),
        md("""
## 1. Dosyayı oku

`ek.oku()` uzantıya bakar. `.dta` dosyalarında **değişken etiketleri** ve **değer etiketleri** korunur; Excel/CSV'de Türkçe biçimler (ondalık virgül, `;` ayırıcı, Türkçe karakterler) otomatik tanınır.
"""),
        kod("""
veri = ek.oku("firma_ornek.dta")
veri.attrs["etiketler"]
"""),
        md("## 2. Stata komutlarıyla analiz"),
        kod("""
s = ek.Stata(veri)
s("describe")
s("summarize satis calisan sermaye arge");
"""),
        kod("""
s("gen lns = ln(satis)")
s("gen lnl = ln(calisan)")
s("gen lnk = ln(sermaye)")
s("xtset firma yil")
"""),
        md("Sabit ve rassal etkiler, ardından Hausman testi — Stata'daki sırayla:"),
        kod("""
s("quietly xtreg lns lnl lnk arge i.yil, fe")
s("estimates store fe")
s("quietly xtreg lns lnl lnk arge i.yil, re")
s("estimates store re")
s("hausman fe re");
"""),
        kod('s("xtreg lns lnl lnk arge i.yil, fe vce(cluster firma)");'),
        md("`gen lns = ln(satis)` ile üretilen değişkenler hatırlandığı için katsayılar **esneklik** olarak yorumlanır:"),
        kod('s("yorum");'),
        md("""
## 3. Do dosyası

Elindeki do dosyalarını olduğu gibi çalıştırabilirsin. Yorumlar (`*`, `//`), satır devamı (`///`) ve `quietly` desteklenir. Önce örnek bir do dosyası yazalım:
"""),
        kod(f"""
do_metni = {do_metni!r}
with open("analiz.do", "w", encoding="utf-8") as f:
    f.write(do_metni)
print(do_metni)
"""),
        kod('ek.Stata().do("analiz.do");'),
        md("""
## 4. Türkçe tez raporu

Son tahmin edilen model için Yöntem + Bulgular metni ve tablolar (etiketler kullanılır):
"""),
        kod("""
s("quietly xtreg lns lnl lnk arge i.yil, fe vce(cluster firma)")
rapor = s.son.rapor()
print(rapor.paragraflar[0])
"""),
        md("""
Word dosyası için: `s("rapor bulgular.docx")`. Panel modelinin tam tez bölümü (model seçim testleriyle) için Python fonksiyonu: `ek.panel_analizi(...).rapor("panel.docx")`.

Tüm komutlar, Stata ile uyum kuralları ve bilinen farklar: [REHBER.md](../REHBER.md)
"""),
    ]


def defter_07():
    return [
        md("""
# 07 · Tez Araç Kutusu: Grafikler, Marjlar, Sıralı / Sayım / Sansürlü Modeller, Panel Birim Kök

Grafik, marj, tablo aktarma, sıralı ve sayım modelleri, panel birim kök ve VAR.

**Öğreneceklerin**
1. Grafik çizmek ve Word raporuna **Şekil** olarak eklemek
2. `margins, at()` ile **koşullu tahmin ve etkiler**; `marginsplot`
3. Tabloyu Word / LaTeX / Excel'e aktarmak (`esttab using`)
4. Sıralı logit (ologit), sayım modelleri (poisson, nbreg) ve tobit
5. Panel birim kök testleri (LLC, IPS, Fisher) ve VAR

> ⚠️ Veriler **simüle edilmiştir**.
"""),
        kod("""
import numpy as np
import pandas as pd
import ekonometri as ek

ucret = ek.veri.yukle("ucret")
m = ek.ekk("np.log(ucret) ~ egitim + tecrube + I(tecrube**2) + C(cinsiyet)", ucret, kovaryans="HC1")
ek.kalinti_grafigi(m)          # Stata: rvfplot
"""),
        md("""
## 1. Grafikler

Her grafik `Sekil` nesnesidir: `kaydet("dosya.png")` ile 300 dpi kaydedilir, `rapor.sekil_ekle()` ile Word raporuna numaralı şekil olarak girer. Eksenlerde ondalık virgül kullanılır.
"""),
        kod("""
m2 = ek.ekk("np.log(ucret) ~ egitim + tecrube + I(tecrube**2)", ucret, kovaryans="HC1")
ek.katsayi_grafigi(m, m2, adlar=["Tam model", "Kısa model"])   # Stata: coefplot
"""),
        md("""
## 2. Koşullu tahminler: `margins, at()`

"Eğitim 8, 12 ve 16 yıl iken, kadın ve erkek için beklenen log ücret ne?" Diğer değişkenler gözlenen değerlerinde tutulur, standart hatalar delta yöntemiyle hesaplanır.
"""),
        kod("""
mj = ek.marjlar(m, at={"egitim": [8, 12, 16]}, grup="cinsiyet")
mj
"""),
        kod("""
ek.marjinal_grafik(mj)          # Stata: marginsplot
"""),
        md("""
Koşullu marjinal etki: tecrübenin getirisi, tecrübe düzeyine göre nasıl değişiyor? (Kare terim olduğu için etki azalır.)
"""),
        kod("""
ek.marjlar(m, dydx="tecrube", at={"tecrube": [5, 20, 35]})
"""),
        md("""
## 3. Tabloyu dışa aktarmak

`ek.karsilastir(...).kaydet()` uzantıya göre Word (.docx), LaTeX (.tex, booktabs), Excel (.xlsx), CSV, Markdown ya da HTML yazar. Stata: `esttab m1 m2 using tablo.docx`.
"""),
        kod("""
k = ek.karsilastir(m2, m, adlar=["(1)", "(2)"])
k.kaydet("tablo.tex")
print(open("tablo.tex", encoding="utf-8").read()[:600])
"""),
        md("""
## 4. Sıralı, sayım ve sansürlü modeller

Sınav puanını dört harf notuna çevirip **sıralı logit** tahmin edelim. Kesim noktaları Stata `ologit` ile aynı biçimdedir: P(y ≤ j) = F(kesim_j − xb).
"""),
        kod("""
sinav = ek.veri.yukle("sinav")
sinav["not_"] = pd.cut(sinav["puan"], [0, 50, 65, 80, 101], labels=[1, 2, 3, 4]).astype(int)
o = ek.ologit("not_ ~ calisma_saati + uyku + C(burs)", sinav)
o
"""),
        kod("""
o.marjinal_etkiler().head(8)    # her not için olasılık değişimi (margins, dydx(*))
"""),
        md("""
**Sayım verisi** (ör. patent, ziyaret, kaza sayısı): `poisson` ve aşırı yayılım varsa `nbreg`. α = 0 için olabilirlik oranı testi otomatik raporlanır. **Sansürlü veri** (ör. 0'da yığılan harcama): `tobit`.
"""),
        kod("""
rng = np.random.default_rng(1)
sinav["ders_disi"] = rng.poisson(np.exp(0.2 + 0.05 * sinav["calisma_saati"]) * rng.gamma(2, 0.5, len(sinav)))
nb = ek.nbreg("ders_disi ~ calisma_saati + uyku", sinav)
print(nb)
nb.oranlar()                     # olay oranları (IRR)
"""),
        kod("""
sinav["harcama"] = np.maximum(-2 + 0.3 * sinav["calisma_saati"] + rng.normal(scale=2, size=len(sinav)), 0)
ek.tobit("harcama ~ calisma_saati + uyku", sinav, alt=0)
"""),
        md("""
## 5. Panel birim kök testleri

Panel zaman serisinde durağanlık: `xtunitroot llc` (ortak birim kök), `ips` ve `fisher` (bireysel birim kök). H0 tüm testlerde "birim kök var".
"""),
        kod("""
firma = ek.xtset(ek.veri.yukle("firma"), "firma", "yil")
firma["ln_satis"] = np.log(firma["satis"])
ek.panel_birim_kok(firma, "ln_satis", test="ips", gecikme="aic", en_fazla_gecikme=2)
"""),
        kod("""
ek.panel_duraganlik(firma, ["ln_satis"], gecikme=0)   # düzey ve birinci fark, IPS + Fisher
"""),
        md("""
## 6. Aynısını Stata komutlarıyla

Tüm bu işlemler `ek.Stata()` oturumunda Stata söz dizimiyle de yapılabilir.
"""),
        kod("""
s = ek.Stata()
s("sysuse ihracat")
s("varsoc ihracat reel_kur, maxlag(4)")
s("var ihracat reel_kur, lags(1/2)")
s("varstable")
"""),
    ]


if __name__ == "__main__":
    import sys
    import tempfile

    HEDEF.mkdir(exist_ok=True)
    DEFTERLER = {
        "01_coklu_regresyon": defter_01,
        "02_zaman_serisi": defter_02,
        "03_logit_probit": defter_03,
        "04_panel_veri": defter_04,
        "05_arac_degisken": defter_05,
        "06_kendi_veri_stata": defter_06,
        "07_tez_arac_kutusu": defter_07,
    }
    secilen = sys.argv[1:] or list(DEFTERLER)
    for ad in secilen:
        nb = nbf.v4.new_notebook()
        nb.cells = DEFTERLER[ad]()
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
        nb.metadata["language_info"] = {"name": "python"}
        with tempfile.TemporaryDirectory() as calisma:  # defterlerin ürettiği dosyalar (ör. .dta, .do) repoya düşmesin
            NotebookClient(nb, timeout=300, kernel_name="python3", resources={"metadata": {"path": calisma}}).execute()
        yol = HEDEF / f"{ad}.ipynb"
        nbf.write(nb, yol)
        print("yazıldı:", yol)
