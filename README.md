# 🚀 Big Data Integration Project: Job Market Analytics

## 📋 Tổng quan Dự án

Dự án tích hợp và phân tích dữ liệu lớn từ 3 nguồn việc làm khác nhau để tạo ra insights về thị trường lao động toàn cầu.

### 🎯 Mục tiêu
- Tích hợp dữ liệu từ 3 nguồn: Glassdoor, Monster.com, Naukri.com
- Phân tích xu hướng thị trường việc làm
- Dự đoán mức lương và yêu cầu kỹ năng
- Tạo dashboard trực quan hóa dữ liệu

## 📊 Dataset

| Dataset | Nguồn | Kích thước | Mô tả |
|---------|-------|------------|-------|
| `DataAnalyst.csv` | Glassdoor | ~73,583 records | Việc làm Data Analyst tại Mỹ |
| `monster_com-job_sample.csv` | Monster.com | ~22,000 records | Việc làm đa dạng toàn cầu |
| `naukri_com-job_sample.csv` | Naukri.com | ~22,000 records | Việc làm tại Ấn Độ |

### Cấu trúc Dữ liệu
- **Glassdoor**: Job Title, Salary, Company, Location, Industry, Rating
- **Monster**: Job Title, Description, Location, Organization, Salary, Sector
- **Naukri**: Job Title, Company, Education, Experience, Skills, Salary

## 🏗️ Kiến trúc Hệ thống

```
📥 DATA SOURCES
├── Glassdoor (US-focused)
├── Monster.com (Global)
└── Naukri.com (India-focused)

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

### Cài đặt Dependencies
```bash
# Clone repository
git clone <repository-url>
cd tichhop-git

# Tạo virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# hoặc
venv\Scripts\activate  # Windows

# Cài đặt dependencies
pip install -r requirements.txt
```

### Chạy với Docker
```bash
# Build và chạy containers
docker-compose up -d

# Chạy Jupyter notebook
docker-compose exec jupyter jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
```

## 📁 Cấu trúc Thư mục

```
tichhop-git/
├── data/                    # Raw datasets
│   ├── DataAnalyst.csv
│   ├── monster_com-job_sample.csv
│   └── naukri_com-job_sample.csv
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
