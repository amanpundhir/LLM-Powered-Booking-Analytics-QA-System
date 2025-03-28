# LLM-Powered Booking Analytics & QA System

An interactive system for analyzing hotel booking data and answering questions via a RAG-powered LLM.
<p>
  <strong>Test the application here:</strong>
  <a href="https://llm-powered-booking-analytics-app-system.streamlit.app/">
    <img src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg" alt="Streamlit App" width="150" height="30">
  </a>
</p>


## Features

### Data Collection & Preprocessing
- Uses `cleaned_hotel_bookings.csv` dataset
- Data cleaning and date parsing
- Revenue calculation

### Analytics & Reporting (10 Visualizations)
1. Revenue trends over time
2. Cancellation rate analysis
3. Geographical distribution map
4. Booking lead time distribution
5. Market segment distribution
6. ADR by hotel type
7. Booking changes analysis
8. Special requests impact
9. Cancellation by deposit type
10. Repeated guests analysis

### Retrieval-Augmented QA (RAG)
- Pinecone vector database integration
- LLM-powered answers using document embeddings
- Context-aware responses to booking questions

### User Interface
- Streamlit-powered dashboard
- Sidebar navigation between analytics and QA
- Interactive visualizations

## Installation

1. **Clone repository**:
```bash
git clone https://github.com/amanpundhir/LLM-Powered-Booking-Analytics-QA-System.git
cd LLM-Powered-Booking-Analytics-QA-System
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
# venv\Scripts\activate  # Windows
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Configure API keys**:
- Replace `"your_google_api_key"` with your Google API key
- Replace `"your_pinecone_api_key"` with your Pinecone API key
- Update index name and file paths as needed

## Usage

**Launch the app**:
```bash
streamlit run main.py
```

Access via:
- Local URL: `http://localhost:8501`
- Network URL: `http://<your-ip>:8501`


## Sample Test Queries & Expected Answers

Below are some example queries you can use to test the system, along with their expected responses:

> **Query 1:** Show me total revenue for July 2017.

**Expected Answer:**
 The total revenue for July 2017 was 3,132,959.07.


> **Query 2:** What is the average price of a hotel booking?

**Expected Answer:**
 The average price of a hotel booking is $101.83.
