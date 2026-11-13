# Bağımsız yazılımlarla doğrulama

Bu tablo `python dogrulama/karsilastir.py` ile üretilir. Her satırda ekonometri-tr'nin sonucu, kodumuzdan bağımsız yazılmış ücretsiz
bir yazılımın sonucuyla karşılaştırılır. Stata'nın standart hata kuralları için R tarafında belgelerinde Stata ile aynı formülü
uyguladıkları yazılan seçenekler kullanılır (estimatr `se_type = "stata"`, sandwich `NeweyWest(adjust = TRUE)`, pyfixest).

| Karşılaştırma | Bağımsız araç | Değer sayısı | En büyük göreli fark | Sonuç |
|---|---|---:|---:|:---:|
| EKK katsayılar ve klasik SH | R lm() | 22 | 5.9e-14 | ✅ |
| EKK, Stata robust (HC1) | R estimatr se_type="stata" | 11 | 1.5e-14 | ✅ |
| EKK, kümelenmiş SH ve p (G − 1 sd) | R estimatr se_type="stata", clusters | 22 | 4.2e-13 | ✅ |
| Ağırlıklı EKK [aw=] / [pw=] | R lm(weights) / estimatr se_type="stata" | 25 | 5.1e-14 | ✅ |
| Breusch-Pagan (Koenker ve Stata varsayılanı), RESET (ŷ²–ŷ⁴), Breusch-Godfrey | R lmtest | 4 | 1.1e-12 | ✅ |
| Sabit etkiler: katsayı ve klasik SH | R plm within | 6 | 3.8e-12 | ✅ |
| Sabit etkiler: kümelenmiş SH | pyfixest (Stata reghdfe kuralı) | 3 | 2.3e-12 | ✅ |
| Rassal etkiler (Swamy-Arora) | R plm random, random.method="swar" | 8 | 9.1e-14 | ✅ |
| Hausman, Breusch-Pagan LM, birim etkisi F testi | R plm phtest / plmtest / pFtest | 3 | 8.2e-11 | ✅ |
| 2SLS (small) ve tanılar: zayıf araç F, Wu-Hausman, Sargan | R AER ivreg | 13 | 1.3e-11 | ✅ |
| Logit: katsayı, SH, sandviç SH | R glm, sandwich | 12 | 9.2e-11 | ✅ |
| ADF (sabit ve trend, sabit gecikme), KPSS | R urca | 3 | 1.4e-12 | ✅ |
| Newey-West (Stata newey) | R sandwich NeweyWest(adjust=TRUE) | 3 | 3.4e-12 | ✅ |
| summarize, detail yüzdelikleri | R quantile(type = 2) | 7 | 0.0e+00 | ✅ |
| t testi, Welch, χ², ANOVA, Bartlett | R t.test / chisq.test / oneway.test / bartlett.test | 6 | 9.0e-14 | ✅ |
| Prais-Winsten (yinelemeli) | gretl ar1 --pwe | 18 | 3.2e-14 | ✅ |
| Sıralı logit / probit: katsayı, kesim noktaları, log-olabilirlik | R MASS polr | 14 | 9.3e-07 | ✅ |
| Çok terimli logit: katsayı, SH, log-olabilirlik | R nnet multinom | 13 | 4.2e-09 | ✅ |
| Poisson ve negatif binom (NB2): katsayı, α, log-olabilirlik | R glm / MASS glm.nb | 12 | 1.6e-08 | ✅ |
| Tobit: katsayı, SH, σ, log-olabilirlik | R AER tobit | 8 | 3.6e-08 | ✅ |
| ROC eğrisi altındaki alan (lroc) | R pROC | 1 | 0.0e+00 | ✅ |
| Panel birim kök: LLC, IPS W-t-bar, Fisher Pm ve Z | R plm purtest | 5 | 6.9e-12 | ✅ |

Ölçüt: göreli fark < 10⁻⁶.

**Bu tabloda olmayanlar:**
- Logit/probit dayanıklı standart hatadaki Stata çarpanı N/(N−1) ve rassal etkilerde kümelenmiş standart hata çarpanı G/(G−1)
  bağımsız bir yazılımla doğrulanamadı; tabloda logit için çarpansız sandviç karşılaştırılır.
- Cochrane-Orcutt: gretl daha gevşek bir yakınsama ölçütüyle durduğundan ρ dördüncü basamakta ayrışır;
  ekonometri-tr ρ değişimi 10⁻⁶'nın altına inene kadar (Stata prais varsayılanı) yineler.
- `xttest0`'ın tek yönlü p-değeri ve `dfuller` kritik değer tablosu (bkz. REHBER.md).
- Sıralı modellerin standart hataları: R `polr` Hessian'ı sayısal türevle hesapladığından ~10⁻⁵ göreli farkla tutar (testlerde).
- `nbreg` standart hataları: R `glm.nb` α'yı sabit kabul eder; Stata gibi β ve α'yı birlikte (tam bilgi matrisi) kullanan
  ekonometri-tr ile ~%0,3 farklıdır. Katsayılar, α ve log-olabilirlik birebir aynıdır.
- LLC: plm düzeltme tablosunu T ile okur; ekonometri-tr varsayılan olarak makaledeki ve Stata'daki gibi T̃ = T − p̄ − 1 kullanır.
  Karşılaştırmada aynı tablo satırı seçilir. Kısa panelde (T < 25) iki taraf da tablonun ilk satırını kullanır.
- R `plm` paketinin `vcovHC(type = "sss")` seçeneği küçük örneklem düzeltmesinde sabit terimi saymaz; bu yüzden sabit etkiler
  kümelenmiş standart hataları √((n−1)/(n−K)) farkıyla (bu veride %0,03) ayrışır. ekonometri-tr, pyfixest (Stata reghdfe kuralı) ile birebir aynıdır.
