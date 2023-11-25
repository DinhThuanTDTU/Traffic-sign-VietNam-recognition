<h3 align="center" font-size= 36px;><b>Trường Đại Học Tôn Đức Thắng</b></h3>
<p align="center">
  <a href="https://tdtu.edu.vn/" title="Trường Đại học Tôn Đức Thắng" style="border: 5;">
    <img src="https://ted.com.vn/wp-content/uploads/2022/08/tdt50.jpg" alt="Trường Đại học Tôn Đức Thấng">
  </a>
</p>
<h1 align="center"><b>Đồ án tổng hợp - 2023-2024 </b></h1>
<h2 align="center"><b> NHẬN DIỆN BIỂN BÁO GIAO THÔNG VIỆT NAM SỬ DỤNG DEEPLEARNING
 </br></h2>

## Giảng viên hướng dẫn
Họ tên | Email
--- | --- 
TS. Đặng Ngọc Minh Đức | Tg_dangngocminhduc@tdtu.edu.vn



## Giảng viên phản biện
Họ tên | Email
--- | --- 
TS. Trần Thanh Phương | tranthanhphuong@tdtu.edu.vn

## Sinh viên thực hiện
Họ tên | MSSV | Email |
--- | --- | -- | 
Nguyễn Đình Thuần | 41900683 | 41900683@student.tdtu.edu.vn

## 1. Tổng Quan Về đề tài
*   Đặt Vấn Đề: Trong khuôn khổ của sự phát triển nhanh chóng của công nghệ và đô thị hóa, việc đảm bảo an toàn giao thông trở thành một thách thức lớn. Biển báo giao thông đóng vai trò quan trọng trong việc hướng dẫn và bảo vệ người tham gia giao thông. 

*   Tầm Quan Trọng của Đề Tài: Nhận diện biển báo giao thông không chỉ cần thiết cho việc tuân thủ luật lệ giao thông mà còn là một yếu tố cốt lõi trong việc phát triển xe tự hành và các hệ thống hỗ trợ lái xe hiện đại.

*   Mục tiêu của Đồ Án: Mục tiêu của đồ án này là phát triển một hệ thống nhận diện biển báo giao thông chính xác và kịp thời sử dụng công nghệ deep learning, đặc biệt tập trung vào dữ liệu từ môi trường giao thông Việt Nam.

*   Ý Nghĩa Ứng Dụng:Ứng dụng của hệ thống này không chỉ giới hạn trong việc nâng cao an toàn giao thông mà còn mở rộng sang các lĩnh vực như hỗ trợ lái xe tự động và quản lý giao thông thông minh.

Khu vực được lựa chọn để thu thập data: Hầu hết các quận ở thành phô Hồ Chí Minh

Input và Output bài toán 
*   Input( được chia làm 3 trường hợp)
    * Ảnh 
    * Video
    * Webcam
*   Output( được chia làm 3 trường hợp)
    * Bouding box quanh biển báo
    * code của biển báo


## 2. Xây dựng bộ dữ liệu - 46 biển báo giao thông
### 2.1. Thu thập dữ liệu

*   cách thu thập dữ liệu trong đề tài này là sử dụng Gopro 9 hero để đi quay khắp thành phố Hồ Chí Minh trong 3 ngày và 3 đêm để đảm bảo sự đa dạng của dữ liệu. Video được quay ở định dạng 1920 x 1080 và 30 FPS
* dữ liệu sau khi quay thành công sẽ được cắt các đoạn video không chứa biển báo giao thông và giữ lại những khung hình chứa biển báo giao thông. Thu được 1 video dài 1 tiếng 12 phút
* Tách hình ảnh từ video đã cắt ghép với mỗi giây là ba bức ảnh
* Lọc các tấm ảnh xấu hoặc bị nhiểu.
### 2.2. Đánh nhãn cho dữ liệu
để đánh nhãn cho dữ liệu lần này thì em đã sử dụng một Tool là [LabelImg](https://github.com/tzutalin/labelImg) vì các lý do sau:
*   giao diện dễ dạng sử dụng
* hỗ trợ định dạng YOLO
* Nhiều chức năng như tự động lưu, thêm các class dễ dàng ...

 * Quy tắc khi label: 
    *	Label bounding box ôm gọn biển báo, tránh label rộng hơn, hay không label hết phần biển báo.
    *	Label những biển báo cách vị trí xe từ 0- 20m. Vì khi nhận diện để thông báo cho người tham gia giao thông, ta cần nhận diện và thông báo trước khi đi qua biển báo đó, để người đi điều chỉnh tốc độ hay chú ý hơn.
    *	Đối với những biển báo bị mất góc hay bị che thì sẽ bỏ qua. Lí do là ảnh đó có thể đánh mất một số features quan trọng, có thể làm cho model học sai.

![Minh họa ảnh được Label](https://github.com/thuanvipghe/Traffic-sign-VietNam-recognition/blob/main/traffic_labels.jpg)

### 2.3. Tổng quan dữ liệu

bộ dữ liệu dữ bao biển báo giao thông trong đề tài này sẽ gồm hơn 46 classes và 11577 ảnh chứa các loại biển báo như sau.

![46 biển báo giao thông và code của mỗi loại biển báo ](https://github.com/thuanvipghe/Traffic-sign-VietNam-recognition/blob/main/Picture/46traffic.jpg)


