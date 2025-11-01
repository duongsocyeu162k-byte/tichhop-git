# 🚀 Job Market Analytics (Simplified - JSON-based)

## 📋 Tổng quan Dự án

Dự án phân tích dữ liệu thị trường việc làm từ 3 nguồn khác nhau với kiến trúc đơn giản dựa trên file JSON.

### 🎯 Mục tiêu
- ✅ Tích hợp và làm sạch dữ liệu từ 3 nguồn: CareerLink, Joboko, TopCV
- ✅ Phân tích toàn diện thị trường việc làm tại Việt Nam
- ✅ Xuất dữ liệu đã xử lý ra file JSON (không cần database)
- ✅ Cung cấp API và Dashboard để truy vấn và visualize dữ liệu

### ✨ Điểm khác biệt phiên bản mới
- **Không cần PostgreSQL hay MongoDB** - Tất cả dữ liệu được lưu trong file JSON
- **Đơn giản hơn** - Dễ setup và chạy
- **Linh hoạt** - Dữ liệu JSON dễ chia sẻ và xử lý
- **Nhanh hơn** - Không cần quản lý database

## 📊 Dataset

| Dataset | Nguồn | Kích thước | Mô tả |
|---------|-------|------------|-------|
| `data_careerlink.json` | CareerLink | ~15,772 records | Việc làm IT tại Việt Nam |
| `data_joboko.json` | Joboko | ~7,522 records | Việc làm đa dạng tại Việt Nam |
| `data_topcv.json` | TopCV | ~2,420 records | Việc làm IT/CNTT tại Việt Nam |

### Cấu trúc Dữ liệu
- **CareerLink**: Tên công việc, Tên công ty, Địa điểm, Mức lương, Kinh nghiệm, Mô tả, Kỹ năng yêu cầu
- **Joboko**: Tên công việc, Tên công ty, Địa điểm, Mức lương, Kinh nghiệm, Mô tả, Kỹ năng yêu cầu, Ngành nghề
- **TopCV**: Tên công việc, Tên công ty, Địa điểm, Mức lương, Kinh nghiệm, Mô tả, Kỹ năng yêu cầu, Quyền lợi

## 🏗️ Kiến trúc Hệ thống (Simplified)

```
📥 DATA SOURCES (JSON Files)
├── data/data_careerlink.json
├── data/data_joboko.json
└── data/data_topcv.json
         ⬇️
🔄 ETL PIPELINE (json_export_pipeline.py)
├── 1. Load: DataLoader
├── 2. Clean & Standardize: DataCleaner (LÀM TRƯỚC)
├── 3. Schema Matching: SchemaMatcher (trên cleaned data)
├── 4. Data Matching: DataMatcher (trên cleaned data)
├── 5. Combine: Merge tất cả sources
└── 6. Export: JSONExporter
         ⬇️
💾 OUTPUT (JSON Files)
├── output/export-TIMESTAMP.json               (Dữ liệu đã xử lý)
├── output/matching_report_TIMESTAMP.json      (Schema & Data Matching)
└── output/pipeline_summary.json                (Tóm tắt pipeline)



## 🛠️ Công nghệ Sử dụng

### Core Technologies
- **Python 3.8+**: Ngôn ngữ chính
- **Pandas**: Data manipulation và analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning

### Storage
- **JSON Files**: Lưu trữ dữ liệu đơn giản, dễ chia sẻ

### Analytics & Visualization
- **Plotly**: Interactive charts
- **Streamlit**: Web dashboard
- **FastAPI**: REST API
- **Jupyter**: Interactive analysis

### Infrastructure
- **Docker** (optional): Containerization
- **Git**: Version control

## 🚀 Cài đặt và Chạy Dự án

### Yêu cầu Hệ thống
- Python 3.8+ 
- pip (Python package manager)
- (Optional) Docker & Docker Compose

### Cách 1: Chạy trực tiếp (Khuyên dùng - Đơn giản nhất)

**Bước 1: Cài đặt dependencies**

```bash
# Tạo virtual environment (khuyên dùng)
python -m venv venv

# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Cài đặt thư viện
pip install --upgrade pip
pip install -r requirements.txt
```

**Bước 2: Chuẩn bị dữ liệu**

Đảm bảo các file dữ liệu có trong thư mục `data/`:
- `data/data_careerlink.json`
- `data/data_joboko.json`
- `data/data_topcv.json`

**Bước 3: Chạy ETL Pipeline**

```bash
python json_export_pipeline.py
```

Pipeline sẽ thực hiện theo thứ tự:
1. ✅ **Load Data**: Đọc dữ liệu từ 3 nguồn JSON (CareerLink, Joboko, TopCV)
2. ✅ **Clean & Standardize** ⬅️ **LÀM TRƯỚC**: Làm sạch và chuẩn hóa dữ liệu
   - Normalize text, locations, salaries
   - Extract structured data (experience, skills)
   - Standardize column names
   - Tạo standard columns (job_title_clean, company_name, location_clean, etc.)
3. ✅ **Schema Matching**: Phân tích tương thích schema giữa các nguồn (trên cleaned data)
   - Detect schema đã standardized
   - So sánh schema giữa các nguồn
   - Tạo unified schema
   - Validate compatibility
4. ✅ **Data Matching**: Tìm duplicates và entity resolution (trên cleaned data)
   - Duplicate detection (chính xác hơn với normalized data)
   - Similarity analysis
   - Entity resolution (companies, job titles)
5. ✅ **Combine**: Kết hợp dữ liệu từ tất cả nguồn
6. ✅ **Export**: Xuất kết quả ra thư mục `output/`:
   - `export-TIMESTAMP.json` - Dữ liệu đã xử lý
   - `matching_report_TIMESTAMP.json` - Schema & Data Matching results
   - `pipeline_summary.json` - Tóm tắt pipeline

### Cách 2: Chạy với Docker

**Bước 1: Build và chạy containers**

```bash
# Chạy pipeline bên ngoài container
python json_export_pipeline.py

# Hoặc chạy trong container
docker-compose exec api python json_export_pipeline.py
```

## 📁 Cấu trúc Thư mục

```
tichhop-git/
├── data/                           # 📥 Dữ liệu nguồn (JSON)
│   ├── data_careerlink.json
│   ├── data_joboko.json
│   └── data_topcv.json
│
├── output/                         # 💾 Dữ liệu đã xử lý (JSON)
│   ├── export-*.json               # Dữ liệu đã làm sạch và standardized
│   ├── matching_report_*.json      # Schema & Data Matching results
│   └── pipeline_summary.json       # Tóm tắt pipeline
│
├── src/                           # 📦 Source code
│   ├── etl/                      # ETL modules
│   │   ├── data_loader.py        # Load dữ liệu từ JSON
│   │   ├── data_cleaner.py       # Làm sạch & chuẩn hóa dữ liệu
│   │   └── schema_matcher.py     # Schema matching & Data matching
│   └── analytics/                # Analytics modules
│       ├── comprehensive_analyzer.py
│       ├── trend_analyzer.py
│       ├── salary_predictor.py
│       └── ...

├── notebooks/                     # 📓 Jupyter notebooks
│   └── 01_data_exploration.ipynb
│
├── config/                        # ⚙️ Configuration files
│   └── config.yaml
│
├── docker/                        # 🐳 Docker configurations
│   ├── Dockerfile.api
│   ├── Dockerfile.dashboard
│   └── Dockerfile.jupyter
│
├── json_export_pipeline.py        # 🚀 Main ETL pipeline
├── requirements.txt               # 📋 Python dependencies
├── docker-compose.yml             # 🐳 Docker compose
└── README.md                      # 📖 This file
```

## 📊 Kết quả và Tính năng

### Pipeline Output
Sau khi chạy `json_export_pipeline.py`, bạn sẽ có:

1. **export-TIMESTAMP.json**: Dữ liệu đã làm sạch và standardized với các trường:
   - `source`: Nguồn dữ liệu (careerlink, joboko, topcv)
   - `job_title_clean`: Tên công việc đã chuẩn hóa
   - `company_name`: Tên công ty
   - `location_clean`, `city`, `country`: Địa điểm đã chuẩn hóa
   - `salary_min`, `salary_max`, `salary_currency`: Mức lương
   - `skills`: Kỹ năng yêu cầu
   - `experience`: Số năm kinh nghiệm
   - `industry`, `job_type`, `job_description`: Thông tin chi tiết
   - Và nhiều trường khác...

2. **matching_report-TIMESTAMP.json**: Kết quả Schema & Data Matching:
   - **Schema Analysis**:
     - Schema compatibility giữa các nguồn
     - Unified schema definition
     - Field coverage analysis
   - **Data Matching**:
     - Duplicate records & groups
     - Similar records analysis
     - Entity resolution (companies, job titles)

3. **pipeline_summary.json**: Tóm tắt kết quả chạy pipeline:
   - Pipeline info (version, run date)
   - Data sources statistics
   - Schema matching metrics
   - Data matching metrics
   - Files generated

### Quy trình Pipeline

```
1. Load Data (DataFrame in memory)
   ↓
2. Clean & Standardize ⬅️ LÀM TRƯỚC (Pandas operations - NO MongoDB)
   - Normalize text, locations, salaries
   - Extract structured data (experience, skills)
   - Standardize column names
   - Tạo standard columns
   ↓
3. Schema Matching (Pandas operations - NO MongoDB)
   - Detect schema từ cleaned data
   - So sánh schema đã standardized
   - Tạo unified schema
   - Validate compatibility
   ↓
4. Data Matching (Pandas operations - NO MongoDB)
   - Tìm duplicates trên cleaned data (chính xác hơn)
   - Entity resolution (companies, job titles)
   - Similarity analysis với normalized values
   ↓
5. Combine Data
   - Merge tất cả sources
   - Remove unnecessary columns
   ↓
6. Export to export-{timestamp}.json
```

**Lưu ý quan trọng:**
- ✅ **Clean & Standardize được làm TRƯỚC** để normalize dữ liệu
- ✅ Schema Matching và Data Matching chạy trên **cleaned data** (sau clean)
- ✅ Sử dụng Pandas operations, **KHÔNG dùng MongoDB**
- ✅ Data Matching chính xác hơn với dữ liệu đã normalized
- ✅ Không cần map columns thủ công (đã có standard columns)

### Tính năng Pipeline

**Schema Matching:**
- ✅ Tự động detect schema từ cleaned data
- ✅ So sánh tương thích schema giữa các nguồn (đã standardized)
- ✅ Tạo unified schema definition
- ✅ Field coverage analysis

**Data Matching:**
- ✅ Duplicate detection (exact & fuzzy matching)
- ✅ Similarity analysis với Levenshtein distance
- ✅ Entity resolution cho companies và job titles
- ✅ Automatic column mapping từ raw → standard

**Data Cleaning:**
- ✅ Text normalization (remove special chars, normalize unicode)
- ✅ Location extraction (city, province, country)
- ✅ Salary extraction và conversion (VND, USD)
- ✅ Experience extraction từ text
- ✅ Skills extraction và normalization


## 🔍 Schema Matching & Data Matching

### Schema Matching
- **Mục đích**: Phát hiện và so sánh schema giữa các nguồn dữ liệu
- **Input**: Cleaned & standardized data
- **Output**:
  - Schema compatibility score
  - Unified schema definition
  - Field coverage analysis
- **Technology**: Pandas operations, Levenshtein distance cho fuzzy matching
- **Lợi ích**: Schema đã standardized nên so sánh dễ dàng và chính xác hơn

### Data Matching
- **Mục đích**: Tìm duplicates và entity resolution
- **Input**: Cleaned & standardized data (đã có standard columns)
- **Output**:
  - Duplicate records & groups
  - Similar records analysis
  - Entity resolution (unique companies, job titles)
- **Technology**: Pandas operations, Levenshtein ratio cho similarity
- **Features**:
  - ✅ Không cần map columns (đã có standard columns)
  - ✅ Fuzzy matching chính xác hơn với normalized values
  - ✅ Entity deduplication hiệu quả hơn
- **Lợi ích**: Matching chính xác hơn vì values đã normalized (ví dụ: "Hà Nội" → "Ha Noi")

### Lý do thứ tự hiện tại (Clean → Schema Match → Data Match)
Pipeline chạy **Clean & Standardize TRƯỚC** Schema Matching và Data Matching:
- ✅ **Data Matching chính xác hơn**: Values đã normalized → matching tốt hơn
- ✅ **Không cần map columns**: Đã có standard columns (job_title_clean, company_name, etc.)
- ✅ **Code đơn giản hơn**: Không cần logic map columns thủ công
- ✅ **Schema Matching dễ dàng**: So sánh schema đã standardized
- ✅ **Performance tốt**: Không cần xử lý nhiều lần

## 🎯 Use Cases

### 1. Phân tích thị trường việc làm
- Xem xu hướng tuyển dụng theo ngành, địa điểm
- So sánh mức lương giữa các vị trí
- Phân tích kỹ năng được yêu cầu nhiều nhất

### 2. Dự đoán lương
- API `/api/analytics/salary-prediction` cung cấp dự đoán lương dựa trên:
  - Tên công việc
  - Địa điểm
  - Kinh nghiệm
  - Ngành nghề

### 3. Tìm kiếm và lọc công việc
- API `/api/jobs` cho phép filter theo nhiều tiêu chí
- Dashboard cung cấp UI trực quan để explore data

### 4. Export và share data
- Dữ liệu JSON dễ dàng chia sẻ
- Dashboard có chức năng download CSV/JSON

## 🔧 Customization

### Thêm nguồn dữ liệu mới
1. Thêm file JSON vào thư mục `data/`
2. Cập nhật `config/config.yaml`
3. (Nếu cần) Thêm cleaning method trong `src/etl/data_cleaner.py`

### Thêm phân tích mới
1. Tạo module mới trong `src/analytics/`
2. Import và sử dụng trong `json_export_pipeline.py`

## 📈 Performance

- **Processing Speed**: 
  - Load Data: ~1-2 giây
  - Schema Matching: ~2-3 giây
  - Data Matching: ~5-10 giây (depends on dataset size)
  - Clean & Standardize: ~5-10 giây
  - **Total**: ~15-30 giây cho 25,000+ records
- **File Size**: 
  - Export data: ~10-20 MB
  - Matching report: ~2-5 MB
  - Pipeline summary: ~50-100 KB
- **Memory Usage**: ~500MB - 1GB cho dataset 25K records

## ❓ FAQ

**Q: Tại sao không dùng database?**
A: Để đơn giản hóa dự án. File JSON đủ cho dataset cỡ nhỏ-trung bình (<100K records), dễ chia sẻ và không cần setup database.

**Q: Làm sao để update dữ liệu?**
A: Chạy lại `python json_export_pipeline.py`. Dashboard và API sẽ tự động load file JSON mới nhất.

**Q: Schema Matching và Data Matching có sử dụng MongoDB không?**
A: **KHÔNG**. Cả Schema Matching và Data Matching đều chạy trên Pandas DataFrames trong memory, không sử dụng MongoDB. MongoDB chỉ được dùng để lưu trữ kết quả (nếu dùng database pipeline), không phải trong json_export_pipeline.

**Q: Tại sao Clean & Standardize làm TRƯỚC Schema Matching và Data Matching?**
A: Pipeline hiện tại làm Clean TRƯỚC vì:
- ✅ **Data Matching chính xác hơn**: Values đã normalized → "Hà Nội", "Ha Noi", "HN" → "Ha Noi" (match được)
- ✅ **Không cần map columns**: Đã có standard columns sẵn
- ✅ **Code đơn giản**: Không cần logic map columns thủ công
- ✅ **Schema Matching dễ dàng**: So sánh schema đã standardized

**Q: Có thể đổi thứ tự không?**
A: Có thể, nhưng không khuyến nghị:
- Nếu làm Matching trước Clean: cần map columns thủ công, matching kém chính xác
- Clean trước Matching (hiện tại): Matching chính xác hơn, code đơn giản hơn

**Q: Có thể dùng với dữ liệu lớn hơn không?**
A: Với >100K records, nên cân nhắc dùng database (PostgreSQL) hoặc Parquet files để tối ưu performance.

**Q: Làm sao để deploy lên production?**
A: 
1. Sử dụng Docker: `docker-compose up -d`
2. Hoặc deploy trên cloud (Heroku, AWS, GCP, Azure)
3. Setup cron job để chạy pipeline định kỳ

**Q: File cũ trong output/ có bị ghi đè không?**
A: Không! Mỗi lần chạy pipeline sẽ tạo file mới với timestamp khác nhau.

## 🔄 Workflow Chi tiết

### Step-by-Step Pipeline

1. **Load Data** (1-2s)
   - Đọc JSON files từ `data/`
   - Convert to pandas DataFrames
   - Total: ~6,000-25,000 records

2. **Clean & Standardize**(5-10s)
   - Clean text (normalize, remove special chars)
   - Extract structured data (salary, experience)
   - Standardize column names
   - Tạo standard columns (job_title_clean, company_name, location_clean, etc.)
   - Output: Standardized DataFrames

3. **Schema Matching** (2-3s)
   - Detect schema từ cleaned data
   - So sánh schema đã standardized
   - Calculate compatibility score
   - Output: Schema analysis report

4. **Data Matching** (5-10s)
   - Find exact duplicates (chính xác hơn với normalized data)
   - Find similar records (fuzzy matching với normalized values)
   - Entity resolution (companies, job titles)
   - Output: Matching report

5. **Combine Data** (1-2s)
   - Concatenate all sources
   - Remove unnecessary columns
   - Add source identifiers
   - Output: Combined DataFrame

6. **Export** (1-2s)
   - Convert DataFrame → JSON
   - Write to `output/export-TIMESTAMP.json`
   - Write matching report
   - Write pipeline summary

## 📄 License

MIT License

## 📞 Contact

- **Author**: Nguyễn Thuỳ Dương
- **Email**: Duong.NT252022M@sis.hust.edu.vn

---

## 📚 Tech Stack References

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plotly Documentation](https://plotly.com/python/)

---

**✨ Dự án phân tích thị trường việc làm - Phiên bản đơn giản với JSON**

*Last updated: 2025*
