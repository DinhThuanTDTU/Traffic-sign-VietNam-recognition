<h0 align="center" font-size= 36px;><b>Trường Đại Học Tôn Đức Thắng</b></h3>
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

và đây là biểu đồ số lần xuất hiện của biển báo trong bộ dữ liệu.
![46 biển báo giao thông và code của mỗi loại biển báo ](https://github.com/thuanvipghe/Traffic-sign-VietNam-recognition/blob/main/Picture/phanbo_class.jpg)

## 3. Huấn luyện model
### 3.1. Faster R-CNN
Thuật toán  [Faser R-CNN ]( https://arxiv.org/abs/1506.01497) được phát triển bởi Shaoqing Ren và cộng sự.

Faster RCNN là một mô hình phát hiện đối tượng hai giai đoạn. Đầu tiên, nó tạo ra các đề xuất vùng mà có thể chứa đối tượng, sau đó sử dụng một mạng phân loại để xác định lớp và vị trí chính xác của đối tượng và nó hoạt động theo các bước sau:
![Mô hình Faster R-CNN](https://media.geeksforgeeks.org/wp-content/uploads/20230823154315/Region-Proposal-Network-RPN-2.png)
*   Rút Trích Đặc Trưng: Đầu tiên, một hình ảnh được đưa qua mạng CNN (ví dụ: VGG hoặc ResNet) để rút trích đặc trưng.

*   Region Proposal Network (RPN): Sau đó, một mạng con gọi là Region Proposal Network sử dụng đặc trưng này để xác định các vùng (regions) mà có thể chứa đối tượng. RPN tạo ra các đề xuất về vị trí và kích thước của hộp giới hạn tiềm năng.

*   ROI Pooling: Các hộp đề xuất từ RPN sau đó được đưa qua một quá trình gọi là ROI Pooling, nơi mỗi hộp được chuyển hóa thành một kích thước cố định để có thể xử lý được.

*   Phân Loại và Định Vị: Cuối cùng, các đặc trưng từ ROI Pooling được sử dụng bởi hai lớp đầu ra: một lớp phân loại để xác định lớp của đối tượng trong mỗi hộp, và một lớp regression để tinh chỉnh vị trí của hộp đề xuất. 

### 3.2. YOLO
Thuật Toán  [YOLO ]( https://arxiv.org/abs/1506.02640) được phát triển bởi Joseph Redmon và cộng sự.
YOLO là một mô hình phát hiện đối tượng nhanh và hiệu quả, hoạt động theo các bước sau:

![Mô hình YOLO ](https://www.labellerr.com/blog/content/images/2023/01/yolo-algorithm-1.webp)

*   Xử Lý Toàn Bộ Hình Ảnh: Đầu tiên, YOLO xem xét toàn bộ hình ảnh một cách tổng thể, không giống như các mô hình phát hiện đối tượng truyền thống.

*   Chia Hình Ảnh thành Lưới: Hình ảnh được chia thành một lưới có kích thước cố định (ví dụ: 13x13).

* Dự Đoán Đối Tượng và Lớp: Mỗi ô trong lưới đưa ra dự đoán về hộp giới hạn và xác suất lớp. Hộp giới hạn bao gồm thông tin về vị trí và kích thước của đối tượng tiềm năng, trong khi xác suất lớp biểu thị khả năng đối tượng thuộc về một lớp cụ thể.

* Lọc và Tinh Chỉnh: Cuối cùng, YOLO áp dụng các kỹ thuật như non-maximum suppression để loại bỏ các hộp giới hạn chồng chéo và giữ lại những hộp với xác suất cao nhất.


3.3. So Sánh hai mô hình:

Faster RCNN: 
YOLO: Nhanh hơn nhưng có thể kém chính xác hơn trong một số trường hợp. Phù hợp cho các ứng dụng cần tốc độ xử lý nhanh.

Faster RCNN thực hiện phát hiện đối tượng theo hai giai đoạn: đầu tiên là đề xuất các hộp giới hạn, sau đó là phân loại và tinh chỉnh. Điều này thường đảm bảo độ chính xác cao hơn nhưng tốc độ xử lý chậm hơn.Phức tạp hơn, chính xác hơn nhưng chậm hơn. Phù hợp cho các ứng dụng cần độ chính xác cao.

YOLO thực hiện tất cả trong một giai đoạn, xử lý nhanh hơn nhưng đôi khi kém chính xác hơn so với Faster RCNN, đặc biệt trong việc phát hiện các đối tượng nhỏ hoặc chồng chéo.


