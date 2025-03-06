# Hệ thống Nhận Diện Xâm Nhập

## Giới thiệu
Hệ thống này sử dụng mô hình YOLOv8 để phát hiện đối tượng và ByteTrack để theo dõi đối tượng theo thời gian thực. Ứng dụng có thể được sử dụng trong giám sát an ninh để nhận diện và cảnh báo các hành vi xâm nhập trái phép. Thông tin chi tiết về hệ thống nằm trong file Document.pdf.

## Thành phần chính
1. **YOLOv8**: Mô hình deep learning hiện đại để phát hiện đối tượng với độ chính xác cao và tốc độ nhanh.
2. **ByteTrack**: Thuật toán theo dõi đối tượng sử dụng thông tin từ YOLOv8 để theo dõi chuyển động một cách chính xác.
3. **OpenCV**: Hỗ trợ xử lý hình ảnh và hiển thị kết quả.
4. **Ultralytics YOLO**: Thư viện cung cấp các mô hình YOLOv8 được huấn luyện sẵn và công cụ hỗ trợ inference.

## Cài đặt
### Yêu cầu hệ thống
- Python >= 3.8
- GPU (tùy chọn, để tăng tốc xử lý)

## Ứng dụng
- Giám sát an ninh trong khu vực nhạy cảm
- Phát hiện hành vi xâm nhập trong nhà máy, công trình, khu vực cấm


