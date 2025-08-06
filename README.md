## 📄 TrafficMGS – Traffic Management System for Metropolitan Lagos

### 🚦 Overview
TrafficMGS is a web-based, smart traffic management solution built for Lagos, Nigeria. It leverages machine learning (YOLOv11), camera feeds, and community reporting to analyze and visualize real-time traffic patterns. The system was designed during my MSc research and aims to support urban mobility, congestion forecasting, and transport policy.

---

### 🛠️ Features

- 🎯 Real-time traffic data capture using camera + YOLOv11
- 🧠 Object detection for vehicle classification (cars, trucks, bikes, mini-bus)
- 🗃️ PostgreSQL integration for structured traffic data storage
- 🌐 Django-based dashboard with traffic heatmaps and zone statistics
- 📸 User blog system for crowd-sourced reports (images + descriptions)
- 🔒 Role-based access for Admin, Analyst, and Traffic Officers
- 🚀 Hosted locally with Ngrok for external access

---

### 📸 Screenshots


> dashboard 
>  <img width="1600" height="900" alt="Tmg" src="https://github.com/user-attachments/assets/1077aa59-4e19-4883-8592-056767d0e0c5" />

>  <img width="1920" height="1080" alt="Screenshot from 2025-05-31 23-22-19" src="https://github.com/user-attachments/assets/2e312a33-b666-4f5e-a0d1-06e93c74126b" />



> blog,
>  <img width="1600" height="900" alt="Blog" src="https://github.com/user-attachments/assets/77964409-54d0-4a9b-bf8d-563fc8f281da" />

> label and detection
> <img width="1912" height="548" alt="label" src="https://github.com/user-attachments/assets/596548d4-ed4d-4567-8192-53865909a30c" />
> <img width="1912" height="548" alt="Screenshot from 2025-06-05 23-33-35" src="https://github.com/user-attachments/assets/181449cd-b757-4888-9fe6-08ad589979fa" />

> database
> <img width="1920" height="1080" alt="Screenshot from 2025-06-06 15-05-39" src="https://github.com/user-attachments/assets/62232918-3245-4834-90b4-b58355975426" />


> Model results here
> <img width="2250" height="1500" alt="PR_curve" src="https://github.com/user-attachments/assets/a96e352b-2437-42f4-8558-f3050403a27b" />
)

---

### 🧱 Technologies Used

| Tool/Library     | Purpose                               |
|------------------|----------------------------------------|
| *Django*       | Web framework, backend logic           |
| *YOLOv11*      | Object detection from video feeds      |
| *PostgreSQL*   | Traffic data storage                   |
| *Ngrok*        | Secure tunnel from local machine       |
| *OpenCV*       | Frame capture and preprocessing        |
| *Leaflet.js*   | Mapping and visualization              |
| *Bootstrap*    | Frontend styling                       |

---

### 🚧 Installation

bash
# Clone repo
git clone https://github.com/zumerhub/trafficmgs

# Set up virtualenv
python -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt

# Run migrations
python manage.py makemigrations
python manage.py migrate

# Start development server
python manage.py runserver


---

### 🔍 How It Works

1. *Camera Feed* captures traffic images in real-time  
2. *YOLOv11* processes frames for vehicle detection  
3. *Detected data* stored in PostgreSQL (timestamp + object type)  
4. *Dashboard* displays zone-wise congestion stats  
5. *Users* report incidents via blog interface  
6. *Admins* moderate posts and oversee traffic updates  

---

### 📂 Project Structure


trafficmgs/
├── core/              # Base config & homepage
├── detection/         # YOLOv11 integration
├── blog/              # Traffic reports from users
├── dashboard/         # Analytics and zone stats
├── media/             # Uploaded images
├── static/            # CSS/JS assets
├── templates/         # HTML templates


---

### 📈 Future Enhancements

- Add predictive modeling using LSTM for congestion forecasting  
- Mobile-first responsive design  
- SMS/Email alerts for accident reports  
- API layer for external data sources and government usage  
- Integration with weather & calendar for smart prediction

---

### 🧠 Research Relevance

TrafficMGS was built as part of MSc research titled:  
*“Adoption of Machine Learning Techniques in Traffic Management in Metropolitan Lagos: A Case Study of Mile 2–Badagry Highway”*

It contributes to the evolution of African smart cities by offering scalable, real-time traffic solutions powered by AI and citizen collaboration.

---

### 🤝 Contribution

This project welcomes collaborators in:
- Transport research  
- Urban analytics  
- Machine learning  
- Civic tech  

Feel free to fork or reach out via [GitHub](https://github.com/zumerhub) or [adekunlesamuel123@gmail.com].
