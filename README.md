<h1 align="center" font-size= 36px;><b>Trường Đại Học Tôn Đức Thắng</b></h3>
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

vvấu đây là biểu đồ số lần xuất hiện của biển báo trong bộ dữ liệu.
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
![Mô hình YOLO ](https://www.labellerr.com/blog/content/images/2023/01/yolo-algorithm-1.webp
)
*   Xử Lý Toàn Bộ Hình Ảnh: Đầu tiên, YOLO xem xét toàn bộ hình ảnh một cách tổng thể, không giống như các mô hình phát hiện đối tượng truyền thống.

*   Chia Hình Ảnh thành Lưới: Hình ảnh được chia thành một lưới có kích thước cố định (ví dụ: 13x13).

* Dự Đoán Đối Tượng và Lớp: Mỗi ô trong lưới đưa ra dự đoán về hộp giới hạn và xác suất lớp. Hộp giới hạn bao gồm thông tin về vị trí và kích thước của đối tượng tiềm năng, trong khi xác suất lớp biểu thị khả năng đối tượng thuộc về một lớp cụ thể.

* Lọc và Tinh Chỉnh: Cuối cùng, YOLO áp dụng các kỹ thuật như non-maximum suppression để loại bỏ các hộp giới hạn chồng chéo và giữ lại những hộp với xác suất cao nhất.


### 3.3. So Sánh hai mô hình:

Faster RCNN: 
YOLO: Nhanh hơn nhưng có thể kém chính xác hơn trong một số trường hợp. Phù hợp cho các ứng dụng cần tốc độ xử lý nhanh.

Faster RCNN thực hiện phát hiện đối tượng theo hai giai đoạn: đầu tiên là đề xuất các hộp giới hạn, sau đó là phân loại và tinh chỉnh. Điều này thường đảm bảo độ chính xác cao hơn nhưng tốc độ xử lý chậm hơn.Phức tạp hơn, chính xác hơn nhưng chậm hơn. Phù hợp cho các ứng dụng cần độ chính xác cao.

YOLO thực hiện tất cả trong một giai đoạn, xử lý nhanh hơn nhưng đôi khi kém chính xác hơn so với Faster RCNN, đặc biệt trong việc phát hiện các đối tượng nhỏ hoặc chồng chéo.

### Huấn luyện mô hình

Trong quá trình đào tạo của mình, tôi đã áp dụng Gradient Descent
optimizer và đào tạo tất cả các mô hình trong 100 epochs với bathch_size là 32. Chúng tôi đã duy trì các cài đặt tham số tương tự trong các model với nhau. Bằng cách sử dụng phương pháp này, chúng tôi đảm bảo tính nhất quán và cho phép so sánh trực tiếp.

#### Cách huấn luyện model 
Sau đây là cách tổ chức thư mục để huấn luyện model:

* DATASET 
    *    Train
          *   Images
          *   Labels
    *    Val
          *   Images
          *   Labels

source code train model Faster R-CNN: [Here](https://github.com/thuanvipghe/Traffic-sign-VietNam-recognition/blob/main/Faster%20R-CNN%20MobileNetV3/train.ipynb)

source code train model YOLO: [Here](https://docs.ultralytics.com/modes/train/#usage-examples)

hoặc bạn cũng có thể tải sẳn model tôi đã để ở phía trên.





## 4.1 Đánh giá Model
Để đánh giá một model Object Detection có tốt hay không thì các nhà khoa học thường sử dụng hay thông số chính đó là mAP và FPS. Đầu tiên để hiểu hai thông số này thì ta cần làm rõ các thông số liên quan.
#### 4.1.1 IOU
IOU (Intersection Over Union): IOU là thước đo mức độ chồng chéo giữa hai hình dạng. Trong bối cảnh phát hiện đối tượng trong thị giác máy tính, IOU thường được sử dụng để đánh giá mức độ chồng chéo giữa hai hình chữ nhật: hộp giới hạn được dự đoán và hộp giới hạn thực tế trên mặt đất.
Công thức:

![Công thức tính IOU](https://github.com/thuanvipghe/Traffic-sign-VietNam-recognition/blob/main/Picture/IOU.png)

#### 4.1.3. Presicion 
Presicion: đo lường mức độ mô hình có thể xác định nhãn hoặc lớp chính xác cho dữ liệu. độ chính xác được tính bằng cách chia số lượng kết quả dương tính thật (𝑇𝑃) cho tổng số lượng kết quả dương tính thật và dương tính giả (𝐹𝑃) 
Công thức:
![Công thức tính Precision](https://github.com/thuanvipghe/Traffic-sign-VietNam-recognition/blob/main/Picture/precision%20.png)

#### 4.1.3. Recall
Recall: Đo lường mức độ một mô hình có thể xác định chính xác các điểm dữ liệu có liên quan hoặc lớp tích cực trong số tất cả các điểm dữ liệu thuộc lớp đó. Nó được tính bằng cách chia số 𝑇𝑃 cho tổng số dương tính thật (TP) và âm tính giả (𝐹𝑁)
Công thức: 
![Công thức tính Precision](https://github.com/thuanvipghe/Traffic-sign-VietNam-recognition/blob/main/Picture/recall.png)

#### 4.1.4. Precision-Recall Curve (PRC):
PRC là biểu diễn đồ họa minh họa mối quan hệ giữa Presicion và Recall ở các ngưỡng khác nhau của mô hình phát hiện đối tượng. Bằng cách điều chỉnh ngưỡng, PRC cho phép quan sát các mức độ Presicion và Recall khác nhau, dẫn đến một đường cong mô tả sự cân bằng giữa hai số liệu này.

![PRC ](https://github.com/thuanvipghe/Traffic-sign-VietNam-recognition/blob/main/Picture/prc.jpg)

#### 4.1.5 Average Precision (AP) và mean Average Precision (mAP)

Trong các bài toán Object Detection, số liệu AP được tính bằng diện tích dưới đường cong PRC. Giá trị này cho biết mức hiệu suất của mô hình trên một lớp tính năng cụ thể. AP cung cấp một giá trị duy nhất để đánh giá sự cân bằng giữa độ chính xác và khả năng thu hồi, đồng thời phản ánh mức độ chính xác của mô hình trong việc định vị đối tượng. Sau khi tính AP, chỉ số Mean Average Precision (mAP) được sử dụng để tính giá trị trung bình của AP.

#### 4.1.6 FPS

FPS cho biết số lượng hình ảnh mà mô hình có thể xử lý trong một giây. Chỉ số này rất quan trọng khi đánh giá khả năng xử lý thời gian thực của mô hình. FPS cao cho thấy model có khả năng xử lý nhanh, phù hợp với các ứng dụng cần phản hồi tức thời.

### 4.2 Đánh giá model

bảng kết quả huấn luyện model được đánh giá trên tập VAL 

Model | mAP50 | mAP50:95 | FPS  
--- | --- | --- | --- 
Faster R-CNN | 0.910 | 0.668 | 39 
YOLOv8n | 0.956 | 0.730 | 84.03
YOLOv8s | 0.981 | 0.790 | 74.07
YOLOv8m | 0.984 | 0.839 | 66.67

Bảng trên cung cấp sự so sánh toàn diện về các đối tượng khác nhau
các mô hình phát hiện, bao gồm các biến thể khác nhau của YOLOv8 và Faster R-CNN. Các thử nghiệm của chúng tôi cho thấy rằng tất cả các mô hình này đều đạt được điểm mAP50 vượt quá 90%. Khi xem xét kỹ hơn, YOLOv8 nổi bật là mô hình chính xác nhất, vượt trội so với các mô hình khác. Tuy nhiên, điều đáng chú ý là Faster R-CNN MobileNetV3 có độ chính xác tương đối thấp hơn trong các đánh giá của chúng tôi. Mặt khác, YOLOv8n đã thể hiện số khung hình trên giây cao nhất (84 FPS), khiến nó trở thành lựa chọn thiết thực hơn cho các ứng dụng phát hiện đối tượng theo thời gian thực yêu cầu xử lý hiệu quả.





