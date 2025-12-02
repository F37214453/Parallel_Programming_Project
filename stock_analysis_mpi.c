/***************************************************************
 * ملف: stock_analysis_mpi.c
 *
 * الفكرة العامة للمشروع:
 *  - عندنا بيانات أسهم في ملف CSV (لكل يوم: open, high, low, close, volume).
 *  - في النسخة السيريال نحسب:
 *      * متوسط السعر اليومي لكل يوم
 *      * العائد اليومي بين كل يوم واليوم اللي بعده
 *      * التذبذب (الانحراف المعياري للعوائد)
 *
 * في هذه النسخة نستخدم MPI عشان:
 *  - نقسم الحسابات على أكثر من عملية (process)
 *  - نقلل وقت التنفيذ عند زيادة حجم البيانات
 *  - مع الحفاظ على نفس النتائج والمنطق الموجود في السيريال
 ***************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

#define MAX_LINE_LEN 256
#define MAX_DAYS     1000

/*--------------------------------------------------------------
 * struct StockData
 *
 * يمثل صف واحد من ملف CSV:
 *  - تاريخ اليوم
 *  - أسعار الافتتاح، الأعلى، الأدنى، الإغلاق
 *  - حجم التداول
 *
 * مهم جداً يكون هذا الـ struct مطابق تماماً
 * لما هو مستخدم في كود السيريال حتى تبقى النتائج متطابقة.
 --------------------------------------------------------------*/
typedef struct {
    char date[20];
    double open, high, low, close, volume;
} StockData;

/*--------------------------------------------------------------
 * daily_average
 *
 * تحسب متوسط سعر اليوم الواحد باستخدام نفس المعادلة
 * المستخدمة في السيريال:
 *
 *   (open + high + low + close) / 4
 *
 * نستخدم هذه القيمة لاحقاً لحساب المتوسط العام لكل الفترة.
 --------------------------------------------------------------*/
double daily_average(StockData s) {
    return (s.open + s.high + s.low + s.close) / 4.0;
}

/*--------------------------------------------------------------
 * daily_return
 *
 * تحسب العائد النسبي بين يومين متتاليين:
 *
 *   (سعر إغلاق اليوم الحالي - سعر إغلاق اليوم السابق)
 *   ---------------------------------------------------
 *                 سعر إغلاق اليوم السابق
 *
 * إذا كان سعر إغلاق اليوم السابق = 0 نتجنب القسمة على صفر
 * ونعيد 0 للعائد.
 --------------------------------------------------------------*/
double daily_return(double prev_close, double curr_close) {
    if (prev_close == 0.0) return 0.0;
    return (curr_close - prev_close) / prev_close;
}

/*--------------------------------------------------------------
 * read_csv
 *
 * مسؤولة عن:
 *  - فتح ملف CSV
 *  - تخطي سطر العناوين (Header)
 *  - قراءة كل صف بالشكل التالي:
 *      Date,Open,High,Low,Close,Volume
 *  - تخزين القيم في مصفوفة من نوع StockData
 *
 * ترجع عدد الصفوف الصحيحة التي تم قراءتها.
 *
 * ملاحظة:
 *  هذه الدالة تُستدعى فقط من العملية 0،
 *  ثم بقية العمليات تستقبل البيانات بعد ذلك من خلال MPI_Bcast.
 --------------------------------------------------------------*/
int read_csv(const char *filename, StockData *data, int max_days) {

    FILE *file = fopen(filename, "r");
    if (!file) return 0;

    char line[MAX_LINE_LEN];
    int count = 0;

    // تخطي سطر العناوين
    fgets(line, sizeof(line), file);

    // قراءة الصفوف سطر سطر
    while (fgets(line, sizeof(line), file) && count < max_days) {

        // نحاول نقرأ 6 قيم بالترتيب المحدد
        if (sscanf(line, "%19[^,],%lf,%lf,%lf,%lf,%lf",
                   data[count].date,
                   &data[count].open,
                   &data[count].high,
                   &data[count].low,
                   &data[count].close,
                   &data[count].volume) == 6)
        {
            count++;
        }
        // لو السطر ما طابق الفورمات نتجاهله ببساطة
    }

    fclose(file);
    return count;
}

int main(int argc, char *argv[]) {

    int rank, size;

    /*--------------------------------------------------------------
     * تهيئة بيئة MPI
     * بعد هذه النقطة جميع العمليات تصبح جاهزة للتنفيذ.
     --------------------------------------------------------------*/
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);  // رقم العملية
    MPI_Comm_size(MPI_COMM_WORLD, &size);  // عدد العمليات الكلي


    /*--------------------------------------------------------------
     * معالجة مدخلات البرنامج:
     *
     * الشكل المتوقع:
     *   ./prog [max_days] file1.csv file2.csv ...
     *
     * - إذا أول مدخل رقم → نستخدمه كحد أعلى لعدد الأيام max_days
     * - غير ذلك → نتعامل مع كل المدخلات على أنها أسماء ملفات.
     --------------------------------------------------------------*/
    int max_days = MAX_DAYS;
    int file_index = 1;

    if (argc >= 3) {
        char *end;
        long tmp = strtol(argv[1], &end, 10);
        if (*end == '\0' && tmp > 0) {
            max_days = (int)tmp;
            file_index = 2;
        }
    }

    if (file_index >= argc) {
        if (rank == 0)
            printf("Usage: %s [max_days] <file1.csv> <file2.csv> ...\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("\n*** Parallel Stock Analysis (MPI Version) ***\n\n");
    }

    /*--------------------------------------------------------------
     * هنا نكرر على كل ملف CSV تم تمريره للبرنامج.
     * هذا يسمح بتحليل أكثر من سهم في تشغيل واحد،
     * وهو نفس الأسلوب المنطقي تقريباً في نسخة السيريال.
     --------------------------------------------------------------*/
    for (int f = file_index; f < argc; f++) {

        const char *filename = argv[f];
        StockData *data = NULL;
        int n = 0;  // عدد الأيام المقروءة من الملف

        /*----------------------------------------------------------
         * الخطوة 1: عملية واحدة فقط (rank 0) تقرأ الملف من القرص.
         *
         * السبب:
         *  - عمليات الإدخال/الإخراج (I/O) ليست موازية بطبيعتها،
         *    وفتح نفس الملف من كل عملية يزيد التعقيد واحتمال الأخطاء.
         *  - أسهل وأوضح أن نقرأ الملف مرة واحدة في عملية واحدة
         *    ثم نرسل البيانات للعمليات الأخرى.
         ----------------------------------------------------------*/
        if (rank == 0) {
            data = malloc(sizeof(StockData) * max_days);
            n = read_csv(filename, data, max_days);
        }

        /*----------------------------------------------------------
         * الخطوة 2: بث عدد الصفوف n لجميع العمليات.
         *
         * الآن كل عملية تعرف:
         *  - كم عدد الأيام التي نملك بيانات لها
         *  - إذا n <= 1 ما فيه شغل كافي ونقدر نتخطى الملف.
         ----------------------------------------------------------*/
        MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);

        if (n <= 1) {
            if (rank == 0) free(data);
            continue;  // ننتقل للملف اللي بعده
        }

        /*----------------------------------------------------------
         * الخطوة 3: تجهيز مصفوفة data في بقية العمليات.
         *
         * في هذه المرحلة العملية 0 لديها البيانات،
         * وباقي العمليات تملك مصفوفة فاضية بنفس الحجم
         * وتنتظر استقبال نسخة من البيانات.
         ----------------------------------------------------------*/
        if (rank != 0) {
            data = malloc(sizeof(StockData) * n);
        }

        /*----------------------------------------------------------
         * الخطوة 4: بث كامل البيانات لكل العمليات.
         *
         * نرسل المصفوفة كـ bytes (MPI_BYTE) لأن struct يحتوي
         * على أنواع متعددة، وهذا أسلوب بسيط لتجاوز بناء
         * نوع مركّب بـ MPI.
         *
         * بعد هذه الخطوة:
         *  - كل عملية لديها نسخة كاملة من بيانات الأيام.
         *  - هذا مهم لأن حساب العائد يتطلب إغلاق يومين متتاليين.
         ----------------------------------------------------------*/
        MPI_Bcast(
            data,
            n * (int)sizeof(StockData),
            MPI_BYTE,
            0,
            MPI_COMM_WORLD
        );

        /*----------------------------------------------------------
         * الخطوة 5: تقسيم الأيام بين العمليات.
         *
         * الهدف:
         *  - توزيع العمل بالتساوي قدر الإمكان
         *  - كل عملية تحسب مجموع المتوسطات لجزء من الأيام
         *
         * التقسيم يتم بهذه الفكرة:
         *  - base_days = n / size  (الجزء الأساسي لكل عملية)
         *  - extra_days = n % size (الأيام المتبقية توزع واحدة واحدة
         *    على أول extra_days من العمليات)
         ----------------------------------------------------------*/
        int base_days = n / size;
        int extra_days = n % size;

        int local_n_days;
        int start_day;

        if (rank < extra_days) {
            local_n_days = base_days + 1;
            start_day = rank * (base_days + 1);
        } else {
            local_n_days = base_days;
            start_day = rank * base_days + extra_days;
        }
        int end_day = start_day + local_n_days;

        /*----------------------------------------------------------
         * الخطوة 6: تقسيم حساب العوائد اليومية.
         *
         * عدد العوائد أقل بواحد من عدد الأيام:
         *   num_returns = n - 1
         *
         * نطبق نفس فكرة التقسيم:
         *  - كل عملية تحسب عوائد مجموعة من الأيام المتتالية
         *  - بدون تكرار ولا تضارب بين العمليات
         ----------------------------------------------------------*/
        int num_returns = n - 1;

        int base_ret = num_returns / size;
        int extra_ret = num_returns % size;

        int local_n_ret;
        int start_ret;

        if (rank < extra_ret) {
            local_n_ret = base_ret + 1;
            start_ret = rank * (base_ret + 1);
        } else {
            local_n_ret = base_ret;
            start_ret = rank * base_ret + extra_ret;
        }
        int end_ret = start_ret + local_n_ret;

        /*----------------------------------------------------------
         * الخطوة 7: مزامنة البداية وحساب الزمن الموازي.
         *
         * - Barrier: نتأكد كل العمليات بدأت بنفس الوقت
         * - Wtime: لقياس الوقت الفعلي للتنفيذ المتوازي فقط
         ----------------------------------------------------------*/
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();

        /*----------------------------------------------------------
         * الخطوة 8: الحساب المحلي لكل عملية.
         *
         * - كل عملية تحسب مجموع المتوسطات اليومية لجزء محدد من الأيام.
         * - وكل عملية تحسب العوائد ومربعات العوائد لجزء من الفترات الزمنية.
         ----------------------------------------------------------*/
        double local_sum_avg = 0.0;
        for (int i = start_day; i < end_day; i++) {
            local_sum_avg += daily_average(data[i]);
        }

        double local_sum_ret = 0.0;
        double local_sum_ret_sq = 0.0;

        for (int j = start_ret; j < end_ret; j++) {
            double r = daily_return(data[j].close, data[j + 1].close);
            local_sum_ret    += r;
            local_sum_ret_sq += r * r;
        }

        /*----------------------------------------------------------
         * الخطوة 9: جمع نتائج كل العمليات في العملية 0.
         *
         * هذا بالضبط دور MPI_Reduce:
         *   - تجمع مجموعات الجزئيات من كل العمليات
         *   - تعطي النتيجة النهائية للعملية 0
         ----------------------------------------------------------*/
        double global_sum_avg = 0.0;
        double global_sum_ret = 0.0;
        double global_sum_ret_sq = 0.0;

        MPI_Reduce(&local_sum_avg,    &global_sum_avg,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_sum_ret,    &global_sum_ret,    1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(&local_sum_ret_sq, &global_sum_ret_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        MPI_Barrier(MPI_COMM_WORLD);
        double t1 = MPI_Wtime();

        /*----------------------------------------------------------
         * الخطوة 10: حساب النتائج النهائية وطباعتها (Rank 0 فقط).
         *
         * - متوسط السعر العام
         * - التذبذب باستخدام الانحراف المعياري
         * - الوقت الموازي المستغرق
         ----------------------------------------------------------*/
        if (rank == 0) {

            double avg_price = global_sum_avg / (double)n;

            double mean_ret = global_sum_ret / (double)num_returns;
            double mean_sq  = global_sum_ret_sq / (double)num_returns;
            double variance = mean_sq - (mean_ret * mean_ret);
            if (variance < 0) variance = 0;  
            double volatility = sqrt(variance);

            printf("File: %s\n", filename);
            printf("Days loaded: %d\n", n);
            printf("Average price: %.4f\n", avg_price);
            printf("Volatility: %.6f\n", volatility);
            printf("Parallel time: %.6f seconds\n", t1 - t0);
            printf("-------------------------------------------\n");
        }

        free(data);
    }

    MPI_Finalize();
    return 0;
}
