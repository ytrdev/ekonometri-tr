# ekonometri-tr sonuçlarını bağımsız R paketleriyle yeniden hesaplar.
# Çalıştırma: Rscript dogrulama/r_hesapla.R <veri_dizini>   (dogrulama/karsilastir.py tarafından çağrılır)
suppressPackageStartupMessages({
  library(estimatr); library(sandwich); library(lmtest); library(plm); library(AER); library(urca); library(tseries)
  library(MASS); library(nnet); library(pROC)
})
args <- commandArgs(trailingOnly = TRUE); d <- args[1]
out <- c()
yaz <- function(ad, x) {
  isim <- names(x); x <- as.numeric(x)
  for (i in seq_along(x)) {
    ek <- if (length(x) == 1) "" else if (!is.null(isim) && nzchar(isim[i])) paste0("[", isim[i], "]") else paste0("[", i, "]")
    out[[paste0(ad, ek)]] <<- x[i]
  }
}

ucret <- read.csv(file.path(d, "ucret.csv"), stringsAsFactors = FALSE)
ucret$cinsiyet <- factor(ucret$cinsiyet, levels = c("Erkek", "Kadın"))
ucret$bolge <- factor(ucret$bolge, levels = c("İstanbul", "Batı Anadolu", "Ege", "Akdeniz", "Orta Anadolu", "Karadeniz", "Doğu Anadolu"))
f <- log(ucret) ~ egitim + tecrube + I(tecrube^2) + cinsiyet + bolge

# 1) EKK: klasik ve Stata robust (HC1)
m <- lm(f, data = ucret)
yaz("ekk_b", coef(m)); yaz("ekk_se", sqrt(diag(vcov(m))))
r <- lm_robust(f, data = ucret, se_type = "stata")
yaz("hc1_se", r$std.error)
# 2) kümelenmiş (Stata CR1, sd = G-1)
ucret$grup <- (seq_len(nrow(ucret)) - 1) %% 40
rc <- lm_robust(f, data = ucret, clusters = grup, se_type = "stata")
yaz("kume_se", rc$std.error); yaz("kume_p", rc$p.value)
# 3) ağırlıklar
ucret$w <- 0.5 + (seq_len(nrow(ucret)) %% 7) / 3
yaz("aw_se", sqrt(diag(vcov(lm(f, data = ucret, weights = w)))))
yaz("pw_se", lm_robust(f, data = ucret, weights = w, se_type = "stata")$std.error)
# 4) varsayım testleri
yaz("bp_koenker", bptest(m, studentize = TRUE)$statistic)
yaz("bp_stata", bptest(m, varformula = ~ fitted(m), studentize = FALSE)$statistic)
yaz("reset_F", resettest(m, power = 2:4, type = "fitted")$statistic)
yaz("bg_lm", bgtest(m, order = 2, fill = 0)$statistic)

# 5) panel
firma <- read.csv(file.path(d, "firma.csv"), stringsAsFactors = FALSE)
p <- pdata.frame(firma, index = c("firma", "yil"))
pf <- log(satis) ~ log(calisan) + log(sermaye) + arge
fe <- plm(pf, data = p, model = "within"); re <- plm(pf, data = p, model = "random", random.method = "swar")
yaz("fe_b", coef(fe)); yaz("fe_se", sqrt(diag(vcov(fe))))
yaz("fe_kume_se", sqrt(diag(vcovHC(fe, method = "arellano", type = "sss", cluster = "group"))))
yaz("re_b", coef(re)); yaz("re_se", sqrt(diag(vcov(re))))
yaz("hausman", phtest(fe, re)$statistic)
yaz("bp_lm", plmtest(plm(pf, data = p, model = "pooling"), type = "bp")$statistic)
yaz("f_birim", pFtest(fe, plm(pf, data = p, model = "pooling"))$statistic)

# 6) araç değişken (AER::ivreg, küçük örneklem)
getiri <- read.csv(file.path(d, "getiri.csv"), stringsAsFactors = FALSE)
getiri$cinsiyet <- factor(getiri$cinsiyet, levels = c("Erkek", "Kadın"))
iv <- ivreg(log(ucret) ~ tecrube + I(tecrube^2) + cinsiyet + egitim | tecrube + I(tecrube^2) + cinsiyet + uzaklik + kardes, data = getiri)
s <- summary(iv, diagnostics = TRUE)
yaz("iv_b", coef(iv)); yaz("iv_se", sqrt(diag(vcov(iv))))
yaz("iv_zayif_F", s$diagnostics["Weak instruments", "statistic"])
yaz("iv_wu_hausman", s$diagnostics["Wu-Hausman", "statistic"])
yaz("iv_sargan", s$diagnostics["Sargan", "statistic"])

# 7) logit: klasik
sinav <- read.csv(file.path(d, "sinav.csv"), stringsAsFactors = FALSE)
sinav$burs <- factor(sinav$burs, levels = c("Hayır", "Evet"))
lg <- glm(gecti ~ calisma_saati + devam + burs, data = sinav, family = binomial, control = glm.control(epsilon = 1e-14, maxit = 100))
yaz("logit_b", coef(lg)); yaz("logit_se", sqrt(diag(vcov(lg))))
yaz("logit_hc0", sqrt(diag(sandwich(lg))))

# 8) zaman serisi
ihr <- read.csv(file.path(d, "ihracat.csv"))
x <- log(ihr$ihracat)
a <- ur.df(x, type = "drift", lags = 3, selectlags = "Fixed")
yaz("adf_sabit_L3", a@teststat[1])
yaz("adf_trend_L2", ur.df(x, type = "trend", lags = 2, selectlags = "Fixed")@teststat[1])
yaz("kpss_sabit_L4", ur.kpss(x, type = "mu", use.lag = 4)@teststat)
mt <- lm(log(ihracat) ~ log(reel_kur) + dunya_buyumesi, data = ihr)
yaz("newey_se", sqrt(diag(NeweyWest(mt, lag = 3, prewhite = FALSE, adjust = TRUE))))

# 9) tanımlayıcı ve klasik testler
yaz("yuzdelik", quantile(ucret$ucret, c(.01, .05, .25, .5, .75, .95, .99), type = 2))
tt <- t.test(ucret ~ cinsiyet, data = ucret, var.equal = TRUE); yaz("ttest_t", tt$statistic)
tw <- t.test(ucret ~ cinsiyet, data = ucret, var.equal = FALSE); yaz("welch_t", tw$statistic); yaz("welch_df", tw$parameter)
yaz("chi2", chisq.test(table(ucret$cinsiyet, ucret$sektor), correct = FALSE)$statistic)
yaz("anova_F", oneway.test(ucret ~ bolge, data = ucret, var.equal = TRUE)$statistic)
yaz("bartlett", bartlett.test(ucret ~ bolge, data = ucret)$statistic)

# 10) sıralı, çok terimli, sayım ve sansürlü modeller
sinav$not_ <- as.integer(cut(sinav$puan, c(0, 50, 65, 80, 101)))
ol <- polr(factor(not_) ~ calisma_saati + uyku + burs, data = sinav, method = "logistic", control = list(reltol = 1e-14))
yaz("ologit_b", coef(ol)); yaz("ologit_kesim", ol$zeta); yaz("ologit_ll", logLik(ol))
op <- polr(factor(not_) ~ calisma_saati + uyku + burs, data = sinav, method = "probit", control = list(reltol = 1e-14))
yaz("oprobit_b", coef(op)); yaz("oprobit_kesim", op$zeta); yaz("oprobit_ll", logLik(op))
sinav$kat3 <- factor(c("a", "b", "c")[as.integer(cut(sinav$puan, c(0, 55, 75, 101)))], levels = c("a", "b", "c"))
ml <- multinom(kat3 ~ calisma_saati + uyku, data = sinav, Hess = TRUE, trace = FALSE, reltol = 1e-14, maxit = 1000)
yaz("mlogit_b", as.vector(t(coef(ml)))); yaz("mlogit_se", as.vector(t(sqrt(matrix(diag(solve(ml$Hessian)), nrow = 2, byrow = TRUE)))))
yaz("mlogit_ll", logLik(ml))
po <- glm(kardes ~ egitim + tecrube, data = getiri, family = poisson, control = glm.control(epsilon = 1e-14, maxit = 100))
yaz("poisson_b", coef(po)); yaz("poisson_se", sqrt(diag(vcov(po)))); yaz("poisson_ll", logLik(po))
nb <- glm.nb(kardes ~ egitim + tecrube, data = getiri, control = glm.control(epsilon = 1e-14, maxit = 100))
yaz("nbreg_b", coef(nb)); yaz("nbreg_alfa", 1 / nb$theta); yaz("nbreg_ll", logLik(nb))
sinav$kesik <- pmax(sinav$puan - 70, 0)
tb <- tobit(kesik ~ calisma_saati + uyku, left = 0, data = sinav)
yaz("tobit_b", coef(tb)); yaz("tobit_se", sqrt(diag(vcov(tb)))[1:3]); yaz("tobit_sigma", tb$scale); yaz("tobit_ll", logLik(tb))
yaz("auc", auc(roc(sinav$gecti, fitted(lg), quiet = TRUE, direction = "<")))

# 11) panel birim kök (plm purtest; Fisher için Stata dfuller'ın MacKinnon 1994 p-değeri)
p$ls <- log(p$satis)
suppressWarnings({
  yaz("llc_sabit_L0", purtest(p$ls, test = "levinlin", exo = "intercept", lags = 0)$statistic$statistic)
  yaz("llc_trend_L1", purtest(p$ls, test = "levinlin", exo = "trend", lags = 1)$statistic$statistic)
  yaz("ips_sabit_L1", purtest(p$ls, test = "ips", exo = "intercept", lags = 1, dfcor = TRUE)$statistic$statistic)
  yaz("fisher_Pm_L0", purtest(p$ls, test = "Pm", exo = "intercept", lags = 0, dfcor = TRUE, p.approx = "MacKinnon1994")$statistic$statistic)
  yaz("fisher_Z_L0", purtest(p$ls, test = "invnormal", exo = "intercept", lags = 0, dfcor = TRUE, p.approx = "MacKinnon1994")$statistic$statistic)
})

for (k in names(out)) cat(k, "=", format(out[[k]], digits = 17), "\n", sep = "")
