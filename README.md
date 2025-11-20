<div align="center">

  <br/>
  <img width="1536" height="1024" alt="53ab72ac-e4c6-44ec-9675-a7fe43f79799" src="https://github.com/user-attachments/assets/33a30b2c-ad18-4b63-9a77-d1e3e83a595d" />

  <p><h3>🚀 AI • ESP32-CAM • ESP32-WROOM-38 • Health Monitoring • LED Alerts • Bus Safety</h3></p>
  
  <img src="https://img.shields.io/badge/Main%20Processor-ESP32--CAM-D81B60?style=for-the-badge&logo=espressif&logoColor=white" />
  <img src="https://img.shields.io/badge/Communication-ESP32--WROOM--38-6A1B9A?style=for-the-badge&logo=espressif&logoColor=white" />
  <img src="https://img.shields.io/badge/Feature-AI%20Drowsiness%20Detection-FFC107?style=for-the-badge&logo=tensorflow&logoColor=white" />
  <img src="https://img.shields.io/badge/Output-LED%20Alerts-FF5722?style=for-the-badge&logo=arduino&logoColor=white" />
  <img src="https://img.shields.io/badge/Domain-Smart%20Transportation-00BCD4?style=for-the-badge&logo=smartthings&logoColor=white" />
</div>

# HỆ THỐNG GIÁM SÁT DỊCH VỤ ĐƯA ĐÓN HỌC SINH

📘  Giới thiệu dự án
  
  Hệ thống giám sát học sinh qua camera và điểm danh học sinh khi học sinh di chuyển đến trường bằng xe đưa đón nhắm giúp phụ huynh yên tâm hơn về thời gian và an toàn di chuyển của con em.

🔧  Linh kiện sử dụng
  - ESP 32 cam
  - ESP32 wroom
  - Lcd screen
  - Led RGB
  - Công tắc hành trình
  - RFID
  - BUZZER

🔌  Sơ đồ kết nối

<img width="1320" height="855" alt="image" src="https://github.com/user-attachments/assets/0ce72ea6-8848-4b94-9d3a-ab813306b079" />

✨ Chức năng chính

 -Điểm danh RFID trước khi lên xe để nhận diện học sinh. 
 
 -ESP32 cam dùng để phát hiện  trạng thái học sinh .(DROWSINESS,UNHEALTHY,HEALTHY,DETECTED,UNDETECTED).
 
 -Sau khi phát hiện thì sẽ gửi thông báo lên led thông qua ESP32 WROOM 38.
 
-Công tắc hành trình dùng để phát hiện học sinh có ngồi trên ghế của mình hay không cũng đồng thời nhận diện sức khỏe khi ra khỏi khung hình.
