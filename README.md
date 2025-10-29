# 🚀 Big Data Integration Project: Job Market Analytics

## 📋 Tổng quan Dự án

Dự án tích hợp và phân tích dữ liệu lớn từ 3 nguồn việc làm khác nhau để tạo ra insights về thị trường lao động toàn cầu.

### 🎯 Mục tiêu
- Tích hợp dữ liệu từ 3 nguồn: CareerLink, Joboko, TopCV
- Phân tích xu hướng thị trường việc làm
- Dự đoán mức lương và yêu cầu kỹ năng
- Tạo dashboard trực quan hóa dữ liệu

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

## 🏗️ Kiến trúc Hệ thống

```
📥 DATA SOURCES
├── CareerLink (Vietnam IT-focused)
├── Joboko (Vietnam diverse)
└── TopCV (Vietnam IT/CNTT)

🔄 DATA INGESTION LAYER
├── Batch Processing: Apache Spark/PySpark
├── Data Cleaning & Standardization
└── Schema Mapping & Transformation

💾 STORAGE LAYER
├── Raw Data: HDFS/S3
├── Processed Data: Hive/Delta Lake
└── Analytics Ready: PostgreSQL/MongoDB

⚡ PROCESSING LAYER
├── Batch Analytics: Spark SQL, Pandas
├── Real-time: Kafka + Spark Streaming
└── ML Pipeline: Scikit-learn, TensorFlow

📊 ANALYTICS LAYER
├── Dashboard: Streamlit/Dash
├── API: FastAPI/Flask
└── Reports: Jupyter Notebooks
```

## 🛠️ Công nghệ Sử dụng

### Core Technologies
- **Python 3.8+**: Ngôn ngữ chính
- **Apache Spark**: Xử lý dữ liệu lớn
- **Pandas**: Data manipulation
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning

### Database & Storage
- **PostgreSQL**: Structured data storage
- **MongoDB**: Semi-structured data
- **HDFS**: Distributed file system
- **Delta Lake**: Data lake storage

### Analytics & ML
- **NLTK/spaCy**: Natural language processing
- **TensorFlow/PyTorch**: Deep learning
- **Prophet**: Time series forecasting
- **Plotly/Matplotlib**: Data visualization

### Infrastructure
- **Docker**: Containerization
- **Jupyter**: Interactive analysis
- **Streamlit**: Web dashboard
- **FastAPI**: REST API

## 📋 Kế hoạch Triển khai

### Phase 1: Data Preparation (2-3 tuần)
- [ ] **Data Cleaning & Standardization**
  - Xử lý missing values, duplicates
  - Standardize job titles, locations, skills
  - Create unified schema

- [ ] **Data Integration**
  - Map common fields across datasets
  - Create master data dictionary
  - Implement data quality checks

### Phase 2: Infrastructure Setup (1-2 tuần)
- [ ] **Environment Setup**
  - Docker containers for reproducibility
  - Jupyter notebooks for analysis
  - Database setup (PostgreSQL/MongoDB)

- [ ] **Data Pipeline**
  - ETL scripts (Python/PySpark)
  - Data validation framework
  - Monitoring and logging

### Phase 3: Analytics Implementation (3-4 tuần)
- [ ] **Exploratory Data Analysis**
  - Statistical summaries
  - Data visualization
  - Correlation analysis

- [ ] **Machine Learning Models**
  - Salary prediction model
  - Skills clustering
  - Sentiment analysis

### Phase 4: Visualization & Reporting (1-2 tuần)
- [ ] **Dashboard Development**
  - Interactive dashboards
  - Real-time analytics
  - Export capabilities

- [ ] **Documentation & Presentation**
  - Technical documentation
  - Business insights report
  - Demo preparation

## 🎯 Các Bài toán Phân tích

### 1. **Phân tích Xu hướng Thị trường Việc làm**
- **Mục tiêu**: Phân tích xu hướng việc làm theo thời gian, địa lý, ngành nghề
- **Phương pháp**: Time series analysis, Geographic analysis
- **Kết quả**: Dashboard hiển thị hot jobs, declining jobs, regional trends

### 2. **Phân tích Mức lương và Yếu tố Ảnh hưởng**
- **Mục tiêu**: Dự đoán mức lương dựa trên skills, experience, location
- **Phương pháp**: Regression analysis, Feature engineering
- **Kết quả**: Salary prediction model, compensation insights

### 3. **Phân tích Kỹ năng và Yêu cầu**
- **Mục tiêu**: Xác định skills quan trọng nhất cho từng vị trí
- **Phương pháp**: NLP, Text mining, Clustering
- **Kết quả**: Skills taxonomy, skill gap analysis

### 4. **Phân tích Cảm xúc và Mô tả Công việc**
- **Mục tiêu**: Phân tích sentiment trong job descriptions
- **Phương pháp**: NLP, Sentiment analysis
- **Kết quả**: Company culture insights, job attractiveness score

### 5. **Phân tích Cạnh tranh và Thị trường**
- **Mục tiêu**: So sánh thị trường việc làm giữa các quốc gia
- **Phương pháp**: Comparative analysis, Statistical modeling
- **Kết quả**: Market comparison dashboard

## 🚀 Cài đặt và Chạy Dự án

### Yêu cầu Hệ thống
- Python 3.8+
- Docker & Docker Compose
- PostgreSQL 12+
- Apache Spark 3.0+


1) Khởi động cơ sở dữ liệu bằng Docker

```bash
docker-compose up -d postgres mongodb

# Kiểm tra container
docker ps --filter "name=job_analytics_"
```

- PostgreSQL sẽ chạy tại `localhost:5432` với DB `job_analytics`, user `admin`, password `password123`.
- MongoDB sẽ chạy tại `localhost:27017` với user `admin`, password `password123`.

2) Xác nhận PostgreSQL đã khởi tạo bảng (tùy chọn)

```bash
# Cài psql nếu cần, rồi chạy:
PGPASSWORD=password123 psql -h 127.0.0.1 -p 5432 -U admin -d job_analytics -c "\dt"
```

Bạn sẽ thấy các bảng như `processed_jobs`, `salary_analysis`, `skills_analysis`, `market_trends` được tạo từ `config/init.sql`.

3) Lưu ý xác thực MongoDB (tránh lỗi auth)

- Người dùng `admin` mặc định nằm ở database `admin`. Nếu bạn gặp lỗi xác thực khi chạy pipeline, dùng chuỗi kết nối có `authSource=admin`.
- Ví dụ chuỗi kết nối an toàn:

```text
mongodb://admin:password123@localhost:27017/job_analytics?authSource=admin
```

Nếu cần, có thể sửa trong file `database_analytics_pipeline.py` (biến `mongodb_connection_string`).

4) Tạo môi trường Python và cài dependency

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux

pip install --upgrade pip
pip install -r requirements.txt
```

5) Chuẩn bị dữ liệu đầu vào

- Đảm bảo các file trong thư mục `data/` tồn tại: `data_careerlink.json`, `data_joboko.json`, `data_topcv.json`.
- Module `src/etl/data_loader.py` sẽ đọc các nguồn này khi pipeline chạy.

6) Chạy pipeline

```bash
python database_analytics_pipeline.py
```

Pipeline sẽ:
- Nạp và làm sạch dữ liệu (ETL) bằng `DataLoader` và `DataCleaner`
- Lưu dữ liệu đã chuẩn hóa vào PostgreSQL bảng `processed_jobs`
- Lưu dữ liệu thô/đã xử lý vào MongoDB (nếu bật)
- Tạo báo cáo phân tích toàn diện và lưu kết quả vào các bảng analytics

7) Kiểm tra nhanh dữ liệu sau khi chạy (tùy chọn)

```bash
# Đếm bản ghi processed_jobs
PGPASSWORD=password123 psql -h 127.0.0.1 -p 5432 -U admin -d job_analytics -c "SELECT COUNT(*) FROM processed_jobs;"
```

8) Xử lý sự cố thường gặp

- Cổng 5432/27017 đã bận: dừng dịch vụ khác hoặc đổi cổng ánh xạ trong `docker-compose.yml`.
- Lỗi MongoDB Authentication: dùng chuỗi kết nối có `?authSource=admin` như hướng dẫn ở bước 3.
- Thiếu thư viện Python: đảm bảo đã cài `requirements.txt` trong đúng virtualenv.
- Lỗi kết nối DB: kiểm tra container đang chạy và network nội bộ Docker hoạt động (`docker ps`).

## 📁 Cấu trúc Thư mục

```
tichhop-git/
├── data/                    # Raw datasets
│   ├── data_careerlink.json
│   ├── data_joboko.json
│   └── data_topcv.json
├── src/                     # Source code
│   ├── etl/                # ETL pipelines
│   ├── analytics/          # Analytics modules
│   ├── models/             # ML models
│   └── utils/              # Utility functions
├── notebooks/              # Jupyter notebooks
├── dashboard/              # Streamlit dashboard
├── api/                    # FastAPI endpoints
├── config/                 # Configuration files
├── tests/                  # Unit tests
├── docs/                   # Documentation
├── docker/                 # Docker configurations
├── requirements.txt        # Python dependencies
├── docker-compose.yml      # Docker compose
└── README.md              # This file
```

## 📊 Kết quả Mong đợi

### Dashboard Features
- **Job Market Trends**: Biểu đồ xu hướng việc làm theo thời gian
- **Salary Analysis**: Phân tích mức lương theo ngành, địa điểm
- **Skills Demand**: Top skills được yêu cầu nhiều nhất
- **Geographic Insights**: So sánh thị trường việc làm theo vùng
- **Company Analysis**: Phân tích công ty và văn hóa làm việc

### API Endpoints
- `GET /api/jobs` - Lấy danh sách việc làm
- `GET /api/salary/predict` - Dự đoán mức lương
- `GET /api/trends` - Xu hướng thị trường
- `GET /api/skills/analysis` - Phân tích kỹ năng

## 🔧 Development

### Code Style
- PEP 8 compliance
- Type hints
- Docstrings
- Unit tests

### Git Workflow
```bash
# Tạo feature branch
git checkout -b feature/new-analysis

# Commit changes
git add .
git commit -m "Add new analysis module"

# Push và tạo PR
git push origin feature/new-analysis
```

## 📈 Metrics & KPIs

- **Data Quality**: >95% data completeness
- **Processing Speed**: <5 minutes for full pipeline
- **Model Accuracy**: >85% for salary prediction
- **Dashboard Load Time**: <3 seconds

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create Pull Request

## 📄 License

MIT License - see LICENSE file for details

## 📞 Contact

- **Author**: [Nguyễn Thuỳ Dương]
- **Email**: [Duong.NT252022M@sis.hust.edu.vn]

---

## 📚 References

- [Apache Spark Documentation](https://spark.apache.org/docs/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

*Dự án này được phát triển cho môn Tích hợp Dữ liệu Lớn - Final Project*
