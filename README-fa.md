# مرجع جامع و دانشنامه تخصصی فریم‌ورک‌های Timeloop ،Sparseloop و LoopTree

این سند به عنوان مرجع کامل و نهایی، شامل تشریح تمامی قابلیت‌ها، فرمولاسیون‌ها، کدهای نمونه، راهنمای کاوش فضای طراحی (**DSE**) همراه با ثبت لاگ کامل و رسم نمودار، تخمین توان و انرژی (**Power & Energy Estimation**) و معماری لایه‌های ادغام‌شده (**LoopTree**) تدوین شده است.

---

## فهرست مطالب
- [۱. تبارشناسی و معماری سه نسل (Timeloop -> Sparseloop -> LoopTree)](#۱-تبارشناسی-و-معماری-سه-نسل-timeloop---sparseloop---looptree)
- [۲. ساختار مخزن: فایل‌های اجرایی (`runner/`)، پوشه ورودی‌ها (`inputs/`) و خروجی‌ها (`outputs/`)](#۲-ساختار-مخزن-فایلهای-اجرایی-runner-پوشه-ورودیها-inputs-و-خروجیها-outputs)
- [۳. ماتریس جامع تمام قابلیت‌های قابل استفاده در این ابزار](#۳-ماتریس-جامع-تمام-قابلیتهای-قابل-استفاده-در-این-ابزار)
  - [قابلیت ۱: کاوش هوشمند فضای نگاشت (Heuristic Mapping Exploration)](#قابلیت-۱-کاوش-هوشمند-فضای-نگاشت-heuristic-mapping-exploration)
  - [قابلیت ۲: ثبت لاگ تمام حالات موفق و مصورسازی فضای سرچ (DSE Logging & Plotting)](#قابلیت-۲-ثبت-لاگ-تمام-حالات-موفق-و-مصورسازی-فضای-سرچ-dse-logging--plotting)
  - [قابلیت ۳: پیاده‌سازی و مقایسه انواع جریان داده (Dataflow Modeling: WS, OS, IS, RS)](#قابلیت-۳-پیادهسازی-و-مقایسه-انواع-جریان-داده-dataflow-modeling-ws-os-is-rs)
  - [قابلیت ۴: ارزیابی قطعی یک نگاشت مشخص بدون سرچ (`timeloop-model`)](#قابلیت-۴-ارزیابی-قطعی-یک-نگاشت-مشخص-بدون-سرچ-timeloop-model)
  - [قابلیت ۵: میکروبنچمارک مشخصات فیزیکی سخت‌افزار (`timeloop-metrics`)](#قابلیت-۵-میکروبنچمارک-مشخصات-فیزیکی-سخت‌افزار-timeloop-metrics)
  - [قابلیت ۶: کاوش فضای طراحی سخت‌افزار (Hardware DSE)](#قابلیت-۶-کاوش-فضای-طراحی-سخت‌افزار-hardware-dse)
  - [قابلیت ۷: تخمین دقیق توان پویا و ایستا (Power & Energy Estimation)](#قابلیت-۷-تخمین-دقیق-توان-پویا-و-ایستا-power--energy-estimation)
  - [قابلیت ۸: مدل‌سازی شتاب‌دهنده‌های اسپارس (Sparseloop: Zero-Skipping & Gating)](#قابلیت-۸-مدلسازی-شتابدهندههای-اسپارس-sparseloop-zero-skipping--gating)
  - [قابلیت ۹: مدل‌سازی شتاب‌دهنده‌های لایه‌-فیوز شده (LoopTree: Fused Layers & Tree Mapping)](#قابلیت-۹-مدلسازی-شتابدهندههای-لایه‌-فیوز-شده-looptree-fused-layers--tree-mapping)
  - [قابلیت ۱۰: مدل‌سازی شبکه ارتباطی درون‌تراشه‌ای و چندپخشی (NoC & Multicast)](#قابلیت-۱۰-مدلسازی-شبکه-ارتباطی-درونتراشهای-و-چندپخشی-noc--multicast)
- [۴. پروژه‌های نمونه عملی و اسکریپت‌های تست‌شده](#۴-پروژه‌های-نمونه-عملی-و-اسکریپت‌های-تست‌شده)
  - [پروژه ۱: شبیه‌سازی شتاب‌دهنده سخت‌افزاری واقعی MMA و مدل RiskMLP به همراه DSE (`inputs/mma_gemm_accelerator/`)](#پروژه-۱-شبیه‌سازی-شتاب‌دهنده-سخت‌افزاری-واقعی-mma-و-مدل-riskmlp-به-همراه-dse-inputsmma_gemm_accelerator)
  - [پروژه ۲: ارزیابی و کاوش با Accelergy و نگاشت بهینه](#پروژه-۲-ارزیابی-و-کاوش-با-accelergy-و-نگاشت-بهینه)
- [۵. راهنمای راه‌اندازی و استفاده بر اساس نقش (شاخه‌های `user` و `master`)](#۵-راهنمای-راهاندازی-و-استفاده-بر-اساس-نقش-شاخههای-user-و-master)
  - [۵.۱. راهنمای شاخه `user` (کاربر نهایی - آماده اجرا بدون نیاز به کامپایل)](#۵۱-راهنمای-شاخه-user-کاربر-نهایی---آماده-اجرا-بدون-نیاز-به-کامپایل)
  - [۵.۲. راهنمای شاخه `master` (توسعه‌دهنده معماری / سورس‌کد کامل LoopTree)](#۵۲-راهنمای-شاخه-master-توسعهدهنده-معماری--سورسکد-کامل-looptree)

---

## ۱. تبارشناسی و معماری سه نسل (Timeloop -> Sparseloop -> LoopTree)

سه نسخه این ابزار در یک مسیر تکاملی مستقیم قرار دارند و هر نسخه، سوپرمجموعه کامل نسخه قبل از خود است:

| نسل | نام و سال مقاله | دامنه مسئله | ساختار نگاشت | ویژگی متمایز |
| :---: | :---: | :---: | :---: | :---: |
| **نسل ۱** | **Timeloop** (ISPASS 2019) | تک‌لایه متراکم (Dense) | زنجیره خطی حلقه‌ها (Loop Nest) | ارزیابی تحلیلی سریع چرخه‌ها و انرژی |
| **نسل ۲** | **Sparseloop** (MICRO 2022) | تک‌لایه اسپارس و متراکم | زنجیره خطی حلقه‌ها (Loop Nest) | فشرده‌سازی، Zero-Skipping و Zero-Gating |
| **نسل ۳** | **LoopTree** (ISPASS 2023) | تک‌لایه و چندلایه ادغام‌شده | **درخت حلقه‌ها (Loop Tree)** | ادغام لایه‌ها (Layer Fusion)، استفاده مجدد بین‌لایه‌ای، محاسبه مجدد |

> **نکته کلیدی:** اگر در LoopTree یک مسئله تک‌لایه‌ای تعریف کنید، درخت فقط ۱ شاخه خواهد داشت و خروجی آن دقیقاً همان ارزیابی نسخه‌های ۱ و ۲ است.

---

## ۲. ساختار مخزن: فایل‌های اجرایی (`runner/`)، پوشه ورودی‌ها (`inputs/`) و خروجی‌ها (`outputs/`)

### ۲.۱. فایل‌های اجرایی (`runner/`) و تمیزسازی خودکار `build/`
* **پوشه `runner/` (محل استقرار باینری‌های اجرایی نهایی):**
  - تمامی فایل‌های باینری اجرایی نهایی پس از کامپایل در پوشه `runner/` قرار می‌گیرند. این پوشه یک ساختار توزیع تمیز است که مسیر آن مستقیماً در اسکریپت `env/setup-env.bash` به متغیر `PATH` اضافه شده است.
  - **ابزارهای اجرایی موجود در `runner/`:**
    * **`looptree-model`**: موتور شبیه‌ساز تحلیلی لایه‌های ادغام‌شده (LoopTree / Fused-Layers) که گراف جریان داده و استفاده مجدد تانسورها را ارزیابی می‌کند.
    * **`timeloop-mapper`**: ابزار اصلی کاوش خودکار فضای نگاشت با استفاده از الگوریتم‌های بهینه‌سازی ترکیبی.
    * **`timeloop-model`**: ارزیاب مستقیم عملکرد یک نگاشت مشخص بدون جست‌وجو (محاسبه دقیق چرخه‌ها، انرژی و توان).
    * **`timeloop-metrics`**: استخراج‌کننده مشخصات فیزیکی سخت‌افزار مستقل از بار کاری (مساحت سیلیکون و انرژی هر دسترسی).
    * **`timeloop-design-space`**: ابزار پیمایش و تحلیل ابعاد فضای طراحی سخت‌افزار.
    * **`timeloop-simple-mapper`**: نسخه سبک‌تر ماپر برای تست‌های سریع.
* **پوشه `build/` و پاک‌سازی خودکار:**
  - فرآیند کامپایل فایل‌های میانی را در `build/` ایجاد می‌کند؛ با این حال در پیکربندی سیستم بیلد، هوک خودکار پس از اتمام موفقیت‌آمیز ساخت اجرا شده و پوشه `build/` را پاک‌سازی می‌کند تا مخزن همواره تمیز و کم‌حجم باقی بماند.
* **کتابخانه‌های اشتراکی (`lib/`):** کتابخانه نهایی کامپایل‌شده `libtimeloop-model.so` در پوشه `lib/` نصب می‌شود و در زمان اجرا به `LD_LIBRARY_PATH` متصل است.

### ۲.۲. پوشه آرشیو نمونه‌ها (`examples/`) و کتابخانه (`tech/`)
پوشه `examples/` آرشیو مرجع معماری‌های مختلف سخت‌افزاری است که هر کدام در پوشه اختصاصی خود ایزوله شده‌اند:
1. **`examples/gemm/`**: معماری، قیود نگاشت و ابعاد ماتریسی ضرب عمومی GEMM به همراه فایل‌های تجمیعی آماده اجرا.
2. **`examples/eyeriss/`**: معماری‌های ۲۵۶ و ۱۰۲۴ واحدی آیریس، فایل‌های تفکیک‌شده و اتصال به انرژی.
3. **`examples/cnn_conv/`**: اشکال ۷ بعدی لایه‌های کانولوشن و الگوهای ارزیابی و کاوش مدل و ماپر.
4. **`examples/simba/`**: مشخصات معماری چندچیپلتی Simba انویدیا.
5. **`examples/mma_gemm_accelerator/`**: پروژه کامل DSE شتاب‌دهنده سخت‌افزاری MMA و مدل RiskMLP به همراه اسکریپت‌های تحلیلی پایتون.
6. **`tech/`**: آرشیو کتابخانه‌های فناوری سیلیکونی متن‌باز در ابعاد مختلف:
   * **`tech/2nm/`**: گره ۲ نانومتر Nanosheet GAAFET (پروژه GT2N گرجیا تک، ۲۰۲۶).
   * **`tech/7nm/`**: گره ۷ نانومتر FinFET پیش‌بینی‌کننده (پروژه ASAP7 دانشگاه آریزونا و آرم).
   * **`tech/45nm/`**: گره ۴۵ نانومتر بالک پلنار (Nangate FreePDK45).
   * **`tech/130nm/`**: گره ۱۳۰ نانومتر پرچگال قابل ساخت صنعتی (SkyWater SKY130 HD گوگل).
   * **`tech/180nm/`**: گره ۱۸۰ نانومتر صنعتی میکروکنترلر (GlobalFoundries GF180MCU گوگل).

### ۲.۳. پوشه فعال ورودی‌ها (`inputs/`)
پوشه `inputs/` فضای کاری فعال شبیه‌سازی است و به صورت پیش‌فرض با معماری شتاب‌دهنده **GEMM** پیکربندی شده است:
* `inputs/architectures/gemm_arch.yaml`: تعریف ساختار سخت‌افزاری واحدهای پردازشی و بافرها.
* `inputs/constraints/gemm_constraints.yaml`: قیود نگاشت مکانی و زمانی.
* `inputs/workloads/`: تعاریف تانسور و ابعاد مسئله (`gemm_ABZ.yaml`, `gemm_problem.yaml`, `gemm.cfg`).
* `inputs/mapper/gemm_mapper.yaml`: تنظیمات الگوریتم جستجو و شرایط همگرایی ماپر.
* `inputs/sample_gemm.yaml`: فایل یکپارچه آماده اجرای کاوش ماپر با یک دستور.
* `inputs/sample_model_gemm.yaml`: فایل یکپارچه آماده اجرای ارزیابی قطعی مدل با یک دستور.

> **نحوه تعویض معماری فعال:** برای شبیه‌سازی هر معماری دلخواه دیگر، کافیست فایل‌های نمونه مورد نظر را از پوشه `examples/` درون `inputs/` کپی نمایید (مثلاً `cp -r examples/eyeriss/* inputs/`).

### ۲.۴. پوشه خروجی‌ها (`outputs/`)
تمامی باینری‌های اجرایی به طور پیش‌فرض پیکربندی شده‌اند که کلیه نتایج شبیه‌سازی را مستقیماً در پوشه **`outputs/`** تولید کنند (در صورت عدم وجود پوشه، برنامه به صورت خودکار آن را ایجاد می‌کند):
* **`outputs/*.stats.txt`**: گزارش متنی تفکیک انرژی و توان اجزا، تعداد چرخه‌ها (Cycles)، و درصد بهره‌وری واحدها (Utilization).
* **`outputs/*.map.txt`**: نگاشت بهینه پیدا شده (تایلینگ زمانی، مکانی و ترتیب حلقه‌ها).
* **`outputs/*.map+stats.xml`**: خروجی ساخت‌یافته XML از تمامی آمارها برای پردازش‌های بعدی یا ترسیم پلات.
* **`outputs/*.log`**: لاگ خط‌به‌خط فرآیند شبیه‌سازی یا پیمایش ماپر.

### ۲.۴. جایگاه و معماری ادغام Accelergy در پروژه
* **ماهیت Accelergy**: فریم‌ورک متن‌باز مستقل بر پایه پایتون (توسعه‌یافته در دانشگاه MIT) است که برای تخمین سطح معماری انرژی و مساحت شتاب‌دهنده‌ها به کار می‌رود. کد منبع پایتونی Accelergy داخل این مخزن C++ کامپایل نمی‌شود، بلکه به صورت یک ابزار خارجی مستقل فراخوانی می‌گردد.
* **نقطه اتصال در سورس‌کد C++**:
  * پیاده‌سازی رابط: `src/util/accelergy_interface.cpp`
  * فایل هدر: `src/util/accelergy_interface.hpp`
* **مکانیزم فعال‌سازی و فراخوانی**:
  1. هنگام کامپایل با گزینه `scons --accelergy`، فلگ پیش‌پردازنده `USE_ACCELERGY` در کدهای C++ فعال می‌شود.
  2. در زمان اجرا، تایم‌لوپ ابزار Accelergy را به صورت یک پروسه خارجی از طریق متغیر محیطی `ACCELERGYPATH` یا دستور شل (`which accelergy`) اجرا می‌کند.
  3. تایم‌لوپ مشخصات معماری و کنش‌های سخت‌افزار (Actionها مثل خواندن/نوشتن حافظه، جمع و ضرب) را به Accelergy تحویل می‌دهد. Accelergy با بهره‌گیری از پلاگین‌های خود (مثل CACTI، Aladdin یا جداول تکنولوژی) جداول مرجع انرژی (**ERT**) و مساحت (**ART**) را تولید کرده و به موتور تایم‌لوپ بازمی‌گرداند.
  4. **مدل پیش‌فرض و جایگزین**: چنانچه پروژه بدون Accelergy کامپایل شود، تایم‌لوپ از مدل داخلی PAT موجود در پوشه `pat-public/` (Power-Area-Timing) برای برآورد مصرف انرژی استفاده می‌نماید.

---

## ۳. ماتریس جامع تمام قابلیت‌های قابل استفاده در این ابزار

در این بخش، تک‌تک قابلیت‌های فریم‌ورک همراه با نحوه پیکربندی، کلیدواژه‌های YAML، و دستورات اجرا تشریح شده‌اند:

---

### قابلیت ۱: کاوش هوشمند فضای نگاشت (Heuristic Mapping Exploration)

این قابلیت توسط ابزار `timeloop-mapper` انجام می‌شود. ماپر وظیفه دارد در میان میلیاردها حالت ممکن برای تایلینگ و جایگشت حلقه‌ها، بهینه‌ترین حالت را بیابد.

#### تنظیمات اصلی ماپر در YAML:
```yaml
mapper:
  # الگوریتم‌های در دسترس:
  # 1. exhaustive: پیمایش کامل (بسیار کند، فقط برای فضاهای بسیار کوچک)
  # 2. linear-pruned: پیمایش خطی با هرس حالات زاید جایگشت
  # 3. random: نمونه‌برداری تصادفی یکنواخت
  # 4. random-pruned: نمونه‌برداری تصادفی با قفل فاکتورها
  # 5. hybrid: ترکیب تصادفی و خطی (بهترین و سریع‌ترین - پیش‌فرض)
  algorithm: hybrid

  # شاخص‌های بهینه‌سازی بر حسب اولویت:
  # energy, delay, edp (Energy-Delay Product), last-level-accesses (ترافیک حافظه اصلی)
  optimization-metrics:
    - energy
    - delay

  # کنترل شرایط خاتمه و موازی‌سازی:
  num-threads: 8               # تعداد تردهای موازی پردازنده میزبان
  timeout: 1000                # سقف تحمل نگاشت‌های نامعتبر متوالی
  victory-condition: 100       # توقف ترد پس از ۱۰۰ نگاشت معتبر بدون بهبود
  search-size: 0               # حداکثر تعداد کل نگاشت‌های بررسی‌شده (0 = نامحدود)
  sync-interval: 10            # فواصل اشتراک‌گذاری بهترین نگاشت بین تردها
  live-status: false           # نمایش زنده مانیتورینگ متنی در ترمینال
```

#### نحوه اجرا:
```bash
timeloop-mapper config.yaml
# یا به صورت ماژولار:
timeloop-mapper arch.yaml problem.yaml mapspace.yaml mapper.yaml
```

---

### قابلیت ۲: ثبت لاگ تمام حالات موفق و مصورسازی فضای سرچ (DSE Logging & Plotting)

به طور پیش‌فرض، ماپر فقط نگاشت‌هایی را که رکورد قبلی را بهبود می‌دهند چاپ می‌کند. برای تحلیل عمیق فضای سرچ و مقایسه تمام حالات موفق، می‌توان لاگ تفصیلی را فعال کرد.

#### فعال‌سازی در کانفیگ:
```yaml
mapper:
  log-suboptimal: true         # ثبت تک‌تک نگاشت‌های معتبر ارزیابی‌شده در طول سرچ
  log-stats: true              # ثبت آمار لحظه‌ای شمارنده‌ها بعد از هر ارزیابی
  diagnostics: true            # در صورت بروز خطا، دلیل رد شدن نگاشت‌ها را تفکیک می‌کند
```

با فعال‌سازی این گزینه‌ها، هر نگاشت موفق در فایل لاگ به شکل زیر ثبت می‌شود:
```text
[Thread 2] Utilization = 0.77 | pJ/Algorithmic-Compute = 5.431 | pJ/Compute = 5.431 | L2[WIO] K2 P14 C2 R3 - L1[WIO] K2 Q14 S3 K16Y C16X - L0[O] C2 Q2 P2
```

#### اسکریپت استخراج لاگ‌ها و تولید نمودارها:
اسکریپت تخصصی [plot_mma_dse.py](file:///d:/UT.CE/NNA/04-TimeLoop/repo/inputs/mma_gemm_accelerator/plot_mma_dse.py) تمام لاگ‌ها را پارس کرده و فایل‌های تحلیلی، جبهه پارتو و نمودارهای جامع تفکیک توان را تولید می‌کند.
1. `all_evaluated_candidates.csv`: ثبت مشخصات تمام نگاشت‌های موفق در طول سرچ (بهره‌وری، انرژی، ساختار حلقه).
2. `dse_comprehensive_report.png`: نمودار ۴ قسمتی شامل جبهه پارتو، تفکیک توان، تریدآف بافر و TOPS/W.
3. `search_space_all_candidates.png`: نمودار پراکندگی (Scatter Plot) کل فضای سرچ.

---

### قابلیت ۳: پیاده‌سازی و مقایسه انواع جریان داده (Dataflow Modeling: WS, OS, IS, RS)

جریان داده مشخص می‌کند کدام تنسورها در کدام سطوح حافظه مقیم شوند و چگونه تایلینگ شوند.

#### ۱. جریان داده حاصل‌جمع ثابت (Output Stationary - OS):
خروجی‌ها تا پایان محاسبه در رجیسترها می‌مانند:
```yaml
mapspace:
  constraints:
    - target: Registers
      type: datatype
      keep: [ Outputs ]
      bypass: [ Weights, Inputs ]
```

#### ۲. جریان داده وزن ثابت (Weight Stationary - WS):
وزن‌ها در رجیسترها قفل می‌شوند و ورودی‌ها و خروجی‌ها در جریانند (مانند NVDLA):
```yaml
mapspace:
  constraints:
    - target: Registers
      type: datatype
      keep: [ Weights ]
      bypass: [ Inputs, Outputs ]
```

#### ۳. جریان داده ردیف ثابت (Row Stationary - RS):
ردیف‌های تصویر و وزن در رجیسترفایل‌های مجزا کش می‌شوند (مانند Eyeriss):
```yaml
mapspace:
  constraints:
    - target: WeightRegFile
      type: datatype
      keep: [ Weights ]
      bypass: [ Inputs, Outputs ]
    - target: InputRegFile
      type: datatype
      keep: [ Inputs ]
      bypass: [ Weights, Outputs ]
    - target: OutputRegFile
      type: datatype
      keep: [ Outputs ]
      bypass: [ Weights, Inputs ]
```

#### ۴. قید تایلینگ مکانی (Spatial Tiling):
توزیع ابعاد حلقه روی مش سخت‌افزاری دو بعدی ($X \times Y$):
```yaml
    - target: GlobalBuffer
      type: spatial
      factors: P1 Q1 R1 S1 C16 K16 N1   # بعد C روی ۱۶ ستون و K روی ۱۶ سطر
      permutation: NCPRS QK
      split: 5                         # تفکیک بعد X و Y
```

---

### قابلیت ۴: ارزیابی قطعی یک نگاشت مشخص بدون سرچ (`timeloop-model`)

اگر نگاشت شتاب‌دهنده توسط کامپایلر شما تولید شده یا می‌خواهید یک نگاشت مشخص را بدون اتلاف وقت ارزیابی کنید:

```yaml
# mapping.yaml
mapping:
  - target: Registers
    type: temporal
    factors: R1 S1 P8 Q1 C2 K16 N1
    permutation: PCKRSQN
  - target: GlobalBuffer
    type: spatial
    factors: R1 S1 P1 Q2 C1 K8 N1
    permutation: QKRSPCN
    split: 0
```

#### دستور اجرا:
```bash
timeloop-model arch.yaml problem.yaml mapping.yaml
```
این دستور مستقیماً فایل‌های آماری `timeloop-model.stats.txt` را در کسری از ثانیه تولید می‌کند.

---

### قابلیت ۵: میکروبنچمارک مشخصات فیزیکی سخت‌افزار (`timeloop-metrics`)

برای استخراج مشخصات ساختاری شتاب‌دهنده (مساحت سیلیکون و انرژی هر دسترسی خواندن/نوشتن) مستقل از هر لایه شبکه عصبی:

```bash
timeloop-metrics arch.yaml
```
خروجی در فایل `timeloop-metrics.stats.txt` ذخیره می‌شود.

---

### قابلیت ۶: کاوش فضای طراحی سخت‌افزار (Hardware DSE)

#### روش ۱: استفاده از ابزار بومی `timeloop-design-space`
تعریف بازه‌ای از متغیرهای معماری در فایل `arch_sweep.yaml`:
```yaml
arch-space-sweep:
  arch-spec: base_arch.yaml
  elements:
    - name: "GlobalBuffer.depth"
      min: 1024
      max: 8192
      step-size: 1024
    - name: "MACs.meshX"
      min: 8
      max: 32
      step-size: 8
```
اجرا:
```bash
timeloop-design-space arch_sweep.yaml problem.yaml
```

#### روش ۲: اتوماسیون پایتونی جامع
اسکریپت [run_mma_dse.py](file:///d:/UT.CE/NNA/04-TimeLoop/repo/inputs/mma_gemm_accelerator/run_mma_dse.py) پارامترها را در فایل کانفیگ تغییر داده، ماپر را اجرا کرده، فایل XML را پارس می‌کند و جداول مقایسه‌ای و نمودارها را تولید می‌نماید.

---

### قابلیت ۷: تخمین دقیق توان پویا و ایستا (Power & Energy Estimation)

#### فرمولاسیون ریاضی:
با داشتن دوره تناوب کلاک ($T_{\text{clk}} = 1 / f_{\text{clk}}$) و تعداد چرخه‌ها ($\text{Cycles}$):

$$T_{\text{exec}} (\text{ms}) = \frac{\text{Cycles} \times 1000}{f_{\text{clk}} (\text{MHz}) \times 10^6}$$

$$P_{\text{avg}} (\text{mW}) = \frac{E_{\text{total}} (\mu\text{J})}{T_{\text{exec}} (\text{ms})}$$

#### تفکیک توان مصرفی در قطعات:
$$P_{\text{total}} = P_{\text{MACs}} + P_{\text{Registers}} + P_{\text{SRAM\_Buffers}} + P_{\text{DRAM\_IF}} + P_{\text{NoC}}$$

#### کارایی و بازدهی:
$$\text{Throughput (GOPS)} = \frac{2 \times \text{MACs}}{T_{\text{exec}} (\text{s}) \times 10^9}$$

$$\text{Energy Efficiency (TOPS/W)} = \frac{\text{Throughput (GOPS)} / 1000}{P_{\text{total}} (\text{W})} = \frac{2 \times \text{MACs}}{E_{\text{total}} (\text{pJ})}$$

---

### قابلیت ۸: مدل‌سازی شتاب‌دهنده‌های اسپارس (Sparseloop: Zero-Skipping & Gating)

در شبکه‌های تنک (Sparse)، بسیاری از وزن‌ها یا مقادیر اکتیویشن صفر هستند.

#### ۱. تعریف چگالی تنسورها در بلوک `problem`:
```yaml
problem:
  shape: cnn-layer
  ...
  densities:
    Weights:
      distribution: fixed-hypergeometric
      density: 0.25          # ۷۵٪ وزن‌ها صفر هستند
    Inputs:
      distribution: fixed-hypergeometric
      density: 0.50          # ۵۰٪ ورودی‌ها صفر هستند
```

#### ۲. قالب‌های فشرده‌سازی در بلوک `arch`:
```yaml
    - name: GlobalBuffer
      sizeKB: 128
      format: compressed-sparse-row  # گزینه‌ها: CSR, CSC, CP, RLE, bitmask, UOP
```

#### ۳. بهینه‌سازی‌های اسپارس سخت‌افزاری (`sparse_optimizations`):
```yaml
sparse_optimizations:
  targets:
    - name: Registers
      action-optimizations:
        # پرش از صفر: کاهش تعداد چرخه‌ها و شتاب اجرای لایه
        - type: skipping
          options:
            - target: Weights
              condition-on: [ Inputs ]
        # گیتینگ ساعت: جلوگیری از سوئیچینگ خازنی و کاهش توان مصرفی
        - type: gating
          options:
            - target: Registers
              condition-on: [ Weights ]
```

---

### قابلیت ۹: مدل‌سازی شتاب‌دهنده‌های لایه‌-فیوز شده (LoopTree: Fused Layers & Tree Mapping)

ویژگی منحصربه‌فرد نسخه ۳ (برنچ `master`) برای از بین بردن ترافیک DRAM بین لایه‌های متوالی.

#### انواع گره‌های درخت نگاشت در LoopTree:
1. **`temporal`:** حلقه زمانی روی یکی از ابعاد با `factor`.
2. **`spatial`:** حلقه توزیع موازی روی مش پردازنده‌ها.
3. **`storage`:** تعیین بافر ذخیره‌سازی برای یک تنسور موقت با فلگ `exploits_reuse`.
4. **`sequential`:** اجرای ترتیبی کاشی‌های لایه ۱ و لایه ۲.
5. **`pipeline`:** اجرای همزمان خط‌لوله‌ای دو لایه با بافر اشتراکی پینگ‌پنگ.
6. **`compute`:** گره اتصال لایه محاسباتی به آرایه پردازنده‌ها.

#### مصالحه نگه‌داری در برابر محاسبه مجدد (Retention vs. Recomputation):
LoopTree محاسبه می‌کند که آیا نگه‌داری داده‌های حاشیه کاشی‌ها در بافر روی تراشه انرژی کمتری مصرف می‌کند، یا محاسبه دوباره آن‌ها در سخت‌افزار برای کوچک‌تر نگه‌داشتن اندازه بافر.

---

### قابلیت ۱۰: مدل‌سازی شبکه ارتباطی درون‌تراشه‌ای و چندپخشی (NoC & Multicast)

در معماری‌های فضایی، داده‌ها از بافر سراسری به چند PE فرستاده می‌شوند:
* **چندپخشی فضایی (Spatial Multicast):** اگر چند PE در یک سطر به داده وزن یکسانی نیاز داشته باشند، داده یک‌بار خوانده شده و روی باس مالتی‌کست می‌شود. تیم‌لوپ تعداد هاپ‌ها (`hops`) و فاکتور چندپخشی (`@multicast`) را در `stats.txt` محاسبه و در مصرف انرژی سیم‌ها اعمال می‌کند.
* **کاهش فضایی (Spatial Reduction):** جمع‌آوری خروجی‌های جزئی از روی آرایه PEها از طریق شبکه درخت جمع‌کننده (Adder Tree Network).

---

## ۴. پروژه‌های نمونه عملی و اسکریپت‌های تست‌شده

در این مخزن، پروژه‌های عملی و پیکربندی‌های آزموده‌شده برای استفاده مستقیم آماده شده است:

### پروژه ۱: شبیه‌سازی شتاب‌دهنده سخت‌افزاری واقعی MMA و مدل RiskMLP به همراه DSE (`inputs/mma_gemm_accelerator/`)

این پروژه مدلسازی دقیق سخت‌افزار واقعی شتاب‌دهنده ماتریسی پروژه بر اساس کد RTL واقعی ([nvdla_matrix_multiplication_4x4.v](file:///D:/UT.CE/MCI/GEMM-accelerator/HW/acc/nvdla_matrix_multiplication_4x4.v) و [matrix_multiplication_accelerator.v](file:///D:/UT.CE/MCI/GEMM-accelerator/HW/mma/matrix_multiplication_accelerator.v)) است:

* **مشخصات ساختاری**:
  * **هسته محاسباتی**: بلوک‌تری‌های موازی (`NUM_COL`) با ۴ ضرب‌کننده INT8 در بعد کاهش $K$ در هر بلوک‌تری به همراه درخت جمع ۴ ورودی.
  * **رجیسترهای اکومولاتور (`acc_reg`)**: رجیستر ۳۲ بیتی جهت انباشت بدون سرریز ضرب‌های جزئی در جریان داده Output Stationary.
  * **بافرهای پینگ‌پنگ**: بافر ورودی اشتراکی (`IA_BUF`) با برودکست به تمام ستون‌ها، بافر محلی وزن‌ها (`W_BUF`)، بافر محلی خروجی (`OA_BUF`).
  * **ارتباط با حافظه اصلی**: اتصال از طریق Dedicated DMA و گذرگاه اشتراکی ۶۴ بیتی به DRAM.
  * **تکنولوژی ساخت**: کتابخانه متن‌باز **45nm Nangate Open Cell** مستقر در [inputs/TechFile/45nm/](file:///d:/UT.CE/NNA/04-TimeLoop/repo/inputs/TechFile/45nm).
* **بار کاری واقعی**: ۳ لایه مدل تخمین ریسک خودروهای خودران (**RiskMLP**):
  * لایه ۱: $M=1, K=16, N=64$
  * لایه ۲: $M=1, K=64, N=64$
  * لایه ۳: $M=1, K=64, N=3$
* **فایل‌های پیاده‌سازی**:
  * [arch_mma.yaml](file:///d:/UT.CE/NNA/04-TimeLoop/repo/inputs/mma_gemm_accelerator/arch_mma.yaml): معماری سلسله‌مراتبی شتاب‌دهنده.
  * [constraints_mma.yaml](file:///d:/UT.CE/NNA/04-TimeLoop/repo/inputs/mma_gemm_accelerator/constraints_mma.yaml): قیود نگاشت مکانی و جریان داده Output Stationary.
  * [problem_risk_mlp.yaml](file:///d:/UT.CE/NNA/04-TimeLoop/repo/inputs/mma_gemm_accelerator/problem_risk_mlp.yaml): ابعاد لایه‌های RiskMLP.
  * [run_mma_dse.py](file:///d:/UT.CE/NNA/04-TimeLoop/repo/inputs/mma_gemm_accelerator/run_mma_dse.py): موتور DSE پارامتریک برای جاروب بلوک‌تری‌ها، سایز بافرها، فرکانس و لایه‌ها به همراه استخراج **تمام نقاط ساب‌اپتیمال**.
  * [plot_mma_dse.py](file:///d:/UT.CE/NNA/04-TimeLoop/repo/inputs/mma_gemm_accelerator/plot_mma_dse.py): رسم ۴ نمودار پیشرفته تحلیلی (پرتو، فضای جستجو، تفکیک توان، کارایی کل RiskMLP).

#### دستورات اجرا:
```bash
cd /mnt/d/UT.CE/NNA/04-TimeLoop/repo/inputs/mma_gemm_accelerator

# ۱. اجرای کامل کاوش فضای طراحی (۶۰ سناریو + ۲۰,۰۸۵ نقطه ساب‌اپتیمال)
python3 run_mma_dse.py --bts 2 4 8 16 32 --bufs 128 256 512 1024 --freq 500.0

# ۲. رسم تمام نمودارها
python3 plot_mma_dse.py
```

---

### پروژه ۲: ارزیابی و کاوش با Accelergy و نگاشت بهینه

برای ارزیابی و کاوش خودکار با استفاده از فناوری ۴۵ نانومتر و کتابخانه Accelergy:

#### ارزیابی قطعی با `timeloop-model`:
```bash
source env/setup-env.bash
timeloop-model inputs/sample_model_gemm.yaml
```

#### کاوش خودکار فضای نگاشت با `timeloop-mapper`:
```bash
source env/setup-env.bash
timeloop-mapper inputs/sample_gemm.yaml
```

#### کاوش با نمونه متصل به شبیه‌ساز انرژی Accelergy:
```bash
source env/setup-env.bash
timeloop-mapper examples/cnn_conv/samples/sample_accelergy.yaml
```
خروجی‌ها شامل گزارش تفکیک توان، جداول ERT و ART، و نگاشت‌های بهینه در پوشه `outputs/` ذخیره می‌شوند.

---

## ۵. راهنمای راه‌اندازی و استفاده بر اساس نقش (شاخه‌های `user` و `master`)

این پروژه منحصراً بر پایه آخرین نسخه پایدار و پیشرفته **LoopTree (Timeloop v3.0)** ساختاریافته است و در دو شاخه مجزا در گیت‌هاب عرضه می‌شود:

```
https://github.com/Mohammad-Khoshroo/timeloop
├── master  (نسخه توسعه‌دهنده: سورس‌کد کامل C++، ابزارهای کامپایل و سیستم بیلد SCons)
└── user    (نسخه کاربر نهایی: باینری‌های از پیش کامپایل‌شده، ورودی‌های آماده، بدون نیاز به کامپایل)
```

---

### ۵.۱. راهنمای شاخه `user` (کاربر نهایی - آماده اجرا بدون نیاز به کامپایل)

این شاخه مخصوص پژوهشگران و کاربرانی است که قصد دارند شبیه‌سازی‌ها و ارزیابی شتاب‌دهنده‌ها را بلافاصله و بدون مواجهه با فرآیند طولانی کامپایل یا پیش‌نیازهای سنگین C++ انجام دهند.

#### ۱. نیازمندی‌های زمان اجرا (Runtime Dependencies):
تنها کتابخانه‌های اشتراکی سیستم لینوکس مورد نیاز است:
```bash
sudo apt update && sudo apt install -y libyaml-cpp0.8 libboost-iostreams1.83.0 libncurses6 libtinfo6
```

#### ۲. کلون و تنظیم محیط:
```bash
# دریافت مستقیم شاخه سبک کاربر
git clone -b user https://github.com/Mohammad-Khoshroo/timeloop.git timeloop-user
cd timeloop-user

# فعال‌سازی متغیرهای مسیر و کتابخانه‌ها
source env/setup-env.bash
```

#### ۳. اجرای شبیه‌سازی:
```bash
# اجرای ارزیاب مستقیم GEMM با تولید خودکار خروجی در outputs/
timeloop-model inputs/sample_model_gemm.yaml

# اجرای کاوشگر خودکار ماپر برای GEMM
timeloop-mapper inputs/sample_gemm.yaml

# اجرای نمونه کاوش با Accelergy
timeloop-mapper examples/cnn_conv/samples/sample_accelergy.yaml

# مشاهده نتایج ذخیره‌شده
ls -lh outputs/
```

---

### ۵.۲. راهنمای شاخه `master` (توسعه‌دهنده معماری / سورس‌کد کامل LoopTree)

این شاخه شامل تمامی کدهای منبع C++ شبیه‌ساز تحلیلی، پایپ‌لاین LoopTree، سیستم بیلد SCons و اتصال به ابزار Accelergy است.

#### ۱. نصب ابزارهای ساخت و وابستگی‌ها:
```bash
sudo apt update && sudo apt install -y \
    scons build-essential libboost-all-dev libyaml-cpp-dev \
    libconfig++-dev libncurses-dev libtinfo-dev libgpm-dev \
    libgmp-dev libtool autoconf automake
```

#### ۲. نصب کتابخانه‌های محاسبات چندوجهی (NTL و Barvinok 0.41.6):
برای تحلیل هندسی زنجیره‌ها و حلقه‌های ادغام‌شده در LoopTree، نصب این دو کتابخانه ضروری است:
```bash
# ۱. کامپایل و نصب NTL
cd /tmp && wget https://libntl.org/ntl-11.5.1.tar.gz && tar -xzf ntl-11.5.1.tar.gz && cd ntl-11.5.1/src
./configure NTL_GMP_LIP=on SHARED=on && make -j$(nproc) && sudo make install

# ۲. کامپایل و نصب Barvinok
cd /tmp && wget https://barvinok.sourceforge.io/barvinok-0.41.6.tar.gz && tar -xzf barvinok-0.41.6.tar.gz && cd barvinok-0.41.6
./configure --enable-shared-barvinok --with-ntl=system && make -j$(nproc) && sudo make install && sudo ldconfig
```

#### ۳. راه‌اندازی و اتصال Accelergy (در صورت نیاز به برآورد انرژی دقیق):
```bash
pip install --break-system-packages accelergy accelergy-cacti-plug-in accelergy-table-based-plug-in
```

#### ۴. کامپایل شبیه‌ساز با SCons:
```bash
cd /path/to/timeloop
source env/setup-env.bash

# کامپایل با فعال‌سازی Accelergy
scons -j$(nproc) --accelergy
```
> **نکته:** پس از پایان موفقیت‌آمیز کامپایل، باینری‌های اجرایی به طور خودکار به پوشه `runner/` منتقل شده و پوشه موقت `build/` به صورت خودکار پاک‌سازی خواهد شد.
