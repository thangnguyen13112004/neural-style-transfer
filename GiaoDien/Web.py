import torch

from ThuVien import image_loader, st_imshow, get_style_model_and_losses, run_style_transfer

#Xây dựng web đơn giản với streamlit
import streamlit as st

import torchvision.models as models

# ----------------------------------------------------BIẾN KHỞI TẠO--------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# kích thước 512 nếu có GPU, else 128
imsize = 512 if torch.cuda.is_available() else 128 

# biến điều chỉnh hàm run_style_transfer
num_steps = 300

# khởi tạo Giá trị chuẩn hóa mô hình VGG19 từ ImageNet.
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)

# tải mô hình VGG 19 để trích xuất đặc trưng từ ảnh (Convolutional + ReLU + Pooling).
cnn = models.vgg19_bn(pretrained=True).features.to(device).eval()


# Code Streamlit
def main():
    st.title("Neural Style Transfer")
    st.write("Tải lên ảnh nội dung và ảnh phong cách để tạo ảnh mới")
    
    # Tạo hai cột cho phần upload ảnh
    col_upload1, col_upload2 = st.columns(2)
    
    with col_upload1:
        # Upload ảnh nội dung
        content_file = st.file_uploader("Ảnh nội dung", type=["jpg", "jpeg", "png"])
    
    with col_upload2:
        # Upload ảnh phong cách
        style_file = st.file_uploader("Ảnh phong cách", type=["jpg", "jpeg", "png"])

    # Xử lý khi người dùng đã tải lên cả 2 ảnh
    if content_file and style_file:
        content_img = image_loader(content_file, imsize)
        style_img = image_loader(style_file, imsize)

        # khởi tạo ảnh nhiễu (input img)
        input_img = torch.randn(content_img.data.size(), device=device)

        # Kiểm tra kích thước
        if content_img.size() != style_img.size():
            st.error("Cần cùng kích thước giữa 2 ảnh!")
            return
        
        # Hiển thị ảnh gốc - chia làm 3 cột: style, content, output
        col1, col2, col3 = st.columns([1, 1, 2])  # Tỉ lệ 1:1:2 giúp phần output lớn hơn

        with col1:
            st_imshow(content_img, "Ảnh nội dung")
        with col2:
            st_imshow(style_img, "Ảnh phong cách")
        with col3:
            st.subheader("Kết quả")
            # Placeholder cho kết quả và trạng thái xử lý
            result_placeholder = st.empty()
            status_placeholder = st.empty()
        
        # Căn giữa button bắt đầu chuyển đổi
        _, button_col, _ = st.columns([1, 2, 1])
        with button_col:
            start_button = st.button("Bắt đầu chuyển đổi phong cách", use_container_width=True)

        # Nút để bắt đầu chuyển đổi phong cách
        if start_button:
            # Hiển thị biểu tượng xử lý trong column 3
            with col3:
                status_placeholder.markdown("""
                <div style='display: flex; justify-content: center; align-items: center; height: 100px;'>
                    <div class="stAudio">
                        <div class="stAudioProgress">
                            <div class="audioWave">
                                <div class="audioWaveItem"></div>
                                <div class="audioWaveItem"></div>
                                <div class="audioWaveItem"></div>
                                <div class="audioWaveItem"></div>
                                <div class="audioWaveItem"></div>
                            </div>
                        </div>
                    </div>
                    <p style='margin-left: 10px;'>Đang xử lý...</p>
                </div>
                <style>
                    .audioWave {
                        display: flex;
                        align-items: center;
                        gap: 3px;
                    }
                    .audioWaveItem {
                        width: 4px;
                        height: 20px;
                        background: #ff4b4b;
                        animation: audioWave 1.5s infinite ease-in-out;
                    }
                    .audioWaveItem:nth-child(1) { animation-delay: 0.0s; }
                    .audioWaveItem:nth-child(2) { animation-delay: 0.2s; }
                    .audioWaveItem:nth-child(3) { animation-delay: 0.4s; }
                    .audioWaveItem:nth-child(4) { animation-delay: 0.6s; }
                    .audioWaveItem:nth-child(5) { animation-delay: 0.8s; }
                    @keyframes audioWave {
                        0%, 40%, 100% { transform: scaleY(0.4); }
                        20% { transform: scaleY(1); }
                    }
                </style>
                """, unsafe_allow_html=True)
            
            # lấy các giá trị cần thiết
            model, style_losses, content_losses = get_style_model_and_losses(cnn,
                    cnn_normalization_mean, cnn_normalization_std, style_img, content_img)
            
            # khởi chạy mô hình 
            output = run_style_transfer(model, style_losses, content_losses,
                                        input_img, num_steps=num_steps)
            
            # Xóa biểu tượng xử lý
            status_placeholder.empty()
            
            # Hiển thị kết quả trong cột 3
            with col3:
                result_placeholder.image(
                    output.cpu().squeeze(0).permute(1, 2, 0).detach().numpy(),
                    caption="Ảnh kết quả",
                    use_column_width=True
                )
                st.success("Hoàn thành!")
    else:
        st.info("Vui lòng tải lên cả ảnh nội dung và ảnh phong cách để tiếp tục")

if __name__ == "__main__":
    main()