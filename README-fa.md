# راهنمای بسته توزیع آماده تایم‌لوپ (شاخه user)

این شاخه یک **بسته توزیع سبک و از پیش کامپایل‌شده** از ابزارهای شبیه‌ساز تحلیلی معماری سخت‌افزار (Timeloop / LoopTree نسخه ۳) است.
**برای استفاده از این شاخه به هیچ‌گونه کامپایلر C++، ابزار SCons، یا فرآیند بیلد نیازی ندارید.**

---

## شروع سریع در ۳ مرحله

### ۱. نصب وابستگی‌های زمان اجرا (Runtime Libraries)
فقط کتابخانه‌های اشتراکی پایه لینوکس مورد نیاز است:
```bash
sudo apt update && sudo apt install -y \
    libyaml-cpp0.8 \
    libboost-iostreams1.83.0 \
    libncurses6 \
    libtinfo6 \
    libisl23
```

### ۲. فعال‌سازی متغیرهای محیطی
در هر نشست ترمینال، دستور زیر را اجرا کنید:
```bash
source env/setup-env.bash
```
این اسکریپت پوشه `runner/` را به `PATH` و پوشه `lib/` را به `LD_LIBRARY_PATH` اضافه می‌کند.

### ۳. اجرای شبیه‌سازی

#### ارزیابی مستقیم عملکرد و انرژی یک نگاشت مشخص:
```bash
timeloop-model inputs/sample_model.cfg
```

#### کاوش خودکار فضای نگاشت برای یافتن بهترین تایلینگ:
```bash
timeloop-mapper inputs/sample_mapper.yaml
```

#### کاوش خودکار فضای نگاشت متصل به Accelergy:
```bash
timeloop-mapper inputs/sample_accelergy.yaml
```

> **نکته:** تمامی خروجی‌های شبیه‌سازی (`*.stats.txt`، `*.map.txt`، `*.map+stats.xml`) به صورت خودکار و بدون نیاز به فلگ اضافی درون پوشه **`outputs/`** ذخیره می‌شوند.

---

## ساختار پوشه‌های این شاخه

* **`runner/`**: باینری‌های اجرایی آماده:
  * `looptree-model`: موتور شبیه‌ساز لایه‌های فیوزشده و مدل LoopTree
  * `timeloop-model`: ارزیاب قطعی نگاشت
  * `timeloop-mapper`: جست‌وجوگر ابتکاری فضای نگاشت
  * `timeloop-metrics`: محاسبه مشخصات مستقل از ورکلود (مساحت و انرژی دسترسی)
  * `timeloop-design-space`: پیمایشگر فضای طراحی سخت‌افزار
  * `timeloop-simple-mapper`: نسخه سبک ماپر
  * `einsumgraph`: ابزار تحلیل گراف اینسام
* **`lib/`**: کتابخانه‌های اشتراکی کامپایل‌شده (`libtimeloop-model.so`, `libtimeloop-mapper.so`)
* **`inputs/`**: کلیه فایل‌های ورودی و پیکربندی:
  * فایل‌های نمونه ماپر و مدل: `inputs/sample_model.cfg`، `inputs/sample_mapper.yaml`، `inputs/sample_accelergy.yaml`
  * فرمولاسیون ریاضی و ابعاد تانسورها: `inputs/cnn_layer.yaml`، `inputs/gemm_ABZ.yaml`
  * کتابخانه فناوری نیمه‌هادی: `inputs/TechFile/` (Nangate 45nm)
  * شتاب‌دهنده سخت‌افزاری MMA و DSE: `inputs/mma_gemm_accelerator/`
* **`outputs/`**: مقصد پیش‌فرض تمامی فایل‌های نتایج شبیه‌سازی
* **`env/setup-env.bash`**: اسکریپت تنظیم خودکار مسیرها

---

## آیا نیاز به ویرایش کدهای C++ یا کامپایل از سورس دارید؟
اگر می‌خواهید شبیه‌ساز را ارتقا دهید، مدل سخت‌افزاری جدید اضافه کنید یا پروژه را از صفر کامپایل نمایید، به شاخه توسعه‌دهنده سوییچ کنید:
```bash
git checkout master
```
مستندات کامل توسعه و راهنمای بیلد در فایل‌های `README.md` و `README-fa.md` در شاخه `master` قرار دارد.
